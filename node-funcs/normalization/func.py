import pandas as pd
from minio import Minio
from minio.error import S3Error
from io import BytesIO
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def min_max_scaling(df):
    """
    Apply Min-Max Scaling to numeric columns of the DataFrame.
    """
    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def decimal_scaling(df):
    """
    Apply Decimal Scaling to numeric columns of the DataFrame.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        max_abs_val = df[col].abs().max()
        scaling_factor = 10 ** len(str(int(max_abs_val)))
        df[col] = df[col] / scaling_factor
    return df


def discretization_standard_scaling(df):
    """
    Apply Standard Scaling (Z-score normalization) to numeric columns of the DataFrame.
    """
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def normalize_df(req):
    """
    Normalize a DataFrame using the requested method, read from MinIO, and save the processed data back to MinIO.
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
        output_file_path = req.json['outputs']['Dataframe']['destination']
        normalization_method = req.json['inputs']['Method']['value']

        # Read data from CSV file in MinIO with Pandas
        obj = client.get_object(bucket_name, input_file_name)
        pickled_data = obj.read()

        # Unpickle the DataFrame
        df = pickle.loads(pickled_data)

        # Apply the selected normalization method
        if normalization_method == "MinMax":
            df = min_max_scaling(df)
        elif normalization_method == "DecimalScaling":
            df = decimal_scaling(df)
        elif normalization_method == "Discretization":
            df = discretization_standard_scaling(df)
        else:
            raise ValueError(f"Unknown normalization method: {normalization_method}")

        # Save the processed data
        with BytesIO() as bytes_buffer:
            pickle.dump(df, bytes_buffer)
            bytes_buffer.seek(0)
            client.put_object(bucket_name, output_file_path, bytes_buffer, len(bytes_buffer.getvalue()))

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
            output_file = normalize_df(context.request)
            print(output_file)
            return context.request.method, 200
        elif context.request.method == "GET":
            return context.request.method, 200
    else:
        print("Empty request", flush=True)
        return "{}", 200
