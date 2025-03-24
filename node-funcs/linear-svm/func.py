import json
import pandas as pd
from minio import Minio
from minio.error import S3Error
from io import BytesIO
import pickle
from sklearn.svm import SVC
from chart_manager.generate_charts import auto_generate_charts_and_report


def linear_svm(req):
    """
    Use the Linear SVM algorithm to classify a dataset read from MinIO and upload the results back to MinIO.
    """
    try:
        # Extract configuration from parsed JSON input
        access_key = req.json['config']['access_key']['value']
        secret_key = req.json['config']['secret_key']['value']
        bucket_name = req.json['config']['bucket_name']['value']

        # Create a MinIO client
        client = Minio(
            "minio:24001",
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )

        # Extract input and output file names directly
        input_file_name = req.json['inputs']['Dataframe']['value']
        output_file_name = req.json['outputs']['Dataframe']['destination']
        target_column = req.json['inputs']['TargetColumn']['value']

        # Read the pickled DataFrame from MinIO
        obj = client.get_object(bucket_name, input_file_name)
        pickled_data = obj.read()

        # Unpickle the DataFrame
        df = pickle.loads(pickled_data)

        print('Cleaning Dataset')
        # Remove the first row (headers if mistakenly included as data)
        df = df.drop(index=0).reset_index(drop=True)

        # Clean the target column
        df[target_column] = df[target_column].apply(lambda x: ''.join(e for e in str(x) if e.isprintable())).str.strip()
        df[target_column] = df[target_column].fillna('Unknown')  # Replace NaNs with 'Unknown'

        # Drop rows with any NaN values
        df_clean = df.dropna(subset=[target_column]).dropna(axis=0, how='any')

        print('Preparing Features')
        # Prepare features and target
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]

        print('Fitting Linear SVM Model')
        # Fit the Linear SVM model
        svm_model = SVC(kernel='linear')  # Linear kernel for linear SVM
        svm_model.fit(X, y)

        # Predictions
        y_pred = svm_model.predict(X)

        # Create a DataFrame for the predictions
        results_df = pd.DataFrame(
            {
                "Index": range(len(df)),
                "Predictions": y_pred.tolist(),
            }
        )

        # Optionally use indices or distances as `y_pred` (modify as needed based on your use case)
        y_pred = y_pred.flatten()

        print('Uploading dataframe')
        # Pickle the results DataFrame and upload it back to MinIO
        with BytesIO() as bytes_buffer:
            pickle.dump(results_df, bytes_buffer)
            bytes_buffer.seek(0)
            client.put_object(bucket_name, output_file_name, bytes_buffer, len(bytes_buffer.getvalue()))

        print('Creating charts')
        # Generate charts and reports
        auto_generate_charts_and_report(
            df=results_df,
            y_pred=y_pred,
            output_folder_name=output_file_name.split('/')[0] + "/visualization_reports",
            client=client,
            bucket_name=bucket_name
        )

        # Return the output file path
        return output_file_name
    except KeyError as exc:
        raise RuntimeError(f"Missing key in JSON input: {exc}")
    except S3Error as exc:
        raise RuntimeError(f"An error occurred while reading from or uploading to MinIO: {exc}")
    except Exception as exc:
        raise RuntimeError(f"Error processing file: {exc}")


# Main function to handle incoming requests.
def main(context):
    """
    Entry point for handling incoming HTTP requests for the linear_svm function.
    """
    if 'request' in context.keys():
        if context.request.method == "POST":
            output_file = linear_svm(context.request)
            print(output_file)
            return context.request.method, 200
        elif context.request.method == "GET":
            return context.request.method, 200
    else:
        print("Empty request", flush=True)
        return "{}", 200
