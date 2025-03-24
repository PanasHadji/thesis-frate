import json
import pandas as pd
from minio import Minio
from minio.error import S3Error
from io import BytesIO
import pickle
from sklearn.ensemble import RandomForestClassifier
from chart_manager.generate_charts import auto_generate_charts_and_report


def random_forest(req):
    """
    Use the Nearest Neighbors algorithm to compute the k-nearest neighbors for a dataset read from MinIO
    and upload the results back to MinIO.
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

        # Extract input and output file names directly without parsing
        input_file_name = req.json['inputs']['Dataframe']['value']
        output_file_name = req.json['outputs']['Dataframe']['destination']

        # Additional parameters for Nearest Neighbors
        n_neighbors = req.json['inputs']['N_Neighbors']['value']
        if n_neighbors == '':
            n_neighbors = 100 # Default to 5 neighbors

        # Read the pickled DataFrame from MinIO
        obj = client.get_object(bucket_name, input_file_name)
        pickled_data = obj.read()

        # Unpickle the DataFrame
        df = pickle.loads(pickled_data)

        # Clean the DataFrame: Ensure all columns are numeric and handle missing values
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.fillna(0)

        # Ensure DataFrame is now all numeric
        if not all(df.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))):
            raise ValueError("Non-numeric columns detected after cleaning.")

        print('Fitting Random Forest Model')
        # Here, we need to extract features (X) and labels (y). Assuming the label is the last column
        X = df.iloc[:, :-1]  # Features (all columns except the last one)
        y = df.iloc[:, -1]  # Labels (last column)

        # Fit the Random Forest model
        rf_model = RandomForestClassifier(n_estimators=100)
        rf_model.fit(X, y)

        # Predictions
        y_pred = rf_model.predict(X)

        # Create a DataFrame for the predictions
        results_df = pd.DataFrame(
            {
                "Index": range(len(df)),
                "Predictions": y_pred.tolist(),
            }
        )

        print('Uploading dataframe')
        # Pickle the neighbors DataFrame and upload it back to MinIO
        with BytesIO() as bytes_buffer:
            pickle.dump(results_df, bytes_buffer)
            bytes_buffer.seek(0)
            client.put_object(bucket_name, output_file_name, bytes_buffer, len(bytes_buffer.getvalue()))

        print('Creating charts')
        auto_generate_charts_and_report(
            df=results_df,
            y_pred=y_pred,
            output_folder_name= output_file_name.split('/')[0]+"/visualization_reports",
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
    Entry point for handling incoming HTTP requests for the nearest_neighbors function.
    """
    if 'request' in context.keys():
        if context.request.method == "POST":
            output_file = random_forest(context.request)
            print(output_file)
            return context.request.method, 200
        elif context.request.method == "GET":
            return context.request.method, 200
    else:
        print("Empty request", flush=True)
        return "{}", 200
