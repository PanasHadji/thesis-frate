import json
import pandas as pd
from minio import Minio
from minio.error import S3Error
from io import BytesIO
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from chart_manager.generate_charts import auto_generate_charts_and_report


def rbf_svm(req):
    """
    Use the RBF SVM algorithm to classify a dataset read from MinIO and upload the results back to MinIO.
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

        print('Fitting Model')
        # Fit RBF SVM model
        rbf_svm_model = SVC(kernel='rbf')
        rbf_svm_model.fit(X, y)

        # Make predictions
        y_pred = rbf_svm_model.predict(X)

        # Save results to MinIO
        results_df = pd.DataFrame({"Index": range(len(df_clean)), "Predictions": y_pred.tolist()})

        print('Saving New Dataframe')
        # Pickle the results DataFrame and upload it back to MinIO
        with BytesIO() as bytes_buffer:
            pickle.dump(results_df, bytes_buffer)
            bytes_buffer.seek(0)
            client.put_object(bucket_name, output_file_name, bytes_buffer, len(bytes_buffer.getvalue()))

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
    Entry point for handling incoming HTTP requests for the rbf_svm function.
    """
    if 'request' in context.keys():
        if context.request.method == "POST":
            print(rbf_svm(context.request))
            return context.request.method, 200
        elif context.request.method == "GET":
            return context.request.method, 200
    else:
        return "{}", 200
