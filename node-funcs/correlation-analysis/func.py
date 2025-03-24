import pandas as pd
from minio import Minio
from minio.error import S3Error
from io import BytesIO
import pickle
import numpy as np
from scipy.stats import spearmanr, kendalltau

def correlation_analysis_df(req):
    """
    Calculate null statistics for a DataFrame read from MinIO and save the statistics to a text file.
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
        correlation_method = req.json['inputs']['Method']['value'].lower()
        output_file_path = req.json['outputs']['FileName']['destination']

        # Read data from the file in MinIO
        obj = client.get_object(bucket_name, input_file_name)
        pickled_data = obj.read()

        # Unpickle the DataFrame
        df = pickle.loads(pickled_data)

        # Ensure numeric data only for correlation
        numeric_data = df.select_dtypes(include=[np.number]).to_numpy()
        columns = df.select_dtypes(include=[np.number]).columns

        # Perform correlation analysis based on the specified method
        if correlation_method == 'pearson':
            corr_matrix = np.corrcoef(numeric_data, rowvar=False)  # Pearson correlation
        elif correlation_method == 'spearman':
            corr_matrix = np.corrcoef(np.argsort(np.argsort(numeric_data, axis=0)), rowvar=False)  # Spearman rank-based
        elif correlation_method == 'kendall':
            corr_matrix = np.zeros((numeric_data.shape[1], numeric_data.shape[1]))
            for i in range(numeric_data.shape[1]):
                for j in range(numeric_data.shape[1]):
                    corr_matrix[i, j] = kendalltau(numeric_data[:, i], numeric_data[:, j]).correlation
        else:
            raise ValueError(
                f"Unsupported correlation method: {correlation_method}. Choose from 'Pearson', 'Spearman', or 'Kendall'.")

        # Convert correlation matrix to CSV format with column labels
        correlation_df = pd.DataFrame(corr_matrix, index=columns, columns=columns)
        csv_buffer = correlation_df.to_csv(index=True)  # Keep index as column names

        # Upload the CSV file to the MinIO bucket
        client.put_object(
            bucket_name,
            f"{output_file_path}.csv",
            BytesIO(csv_buffer.encode()),
            len(csv_buffer)
        )

        return output_file_path
    except KeyError as exc:
        raise RuntimeError(f"Missing key in JSON input: {exc}")
    except S3Error as exc:
        raise RuntimeError(f"An error occurred while reading from or uploading to MinIO: {exc}")
    except Exception as exc:
        raise RuntimeError(f"Error processing file: {exc}")


# Main function to handle incoming requests.
def main(context):
    """
    Main function to handle incoming requests.
    """
    if 'request' in context.keys():
        if context.request.method == "POST":
            output_file = correlation_analysis_df(context.request)
            print(output_file)
            return context.request.method, 200
        elif context.request.method == "GET":
            return context.request.method, 200
    else:
        print("Empty request", flush=True)
        return "{}", 200
