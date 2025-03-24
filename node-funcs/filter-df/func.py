import json
import pandas as pd
from minio import Minio
from minio.error import S3Error
from io import BytesIO
import pickle

def filter_df(req):
    """
    Filter multiple columns in a DataFrame read from MinIO and upload the filtered DataFrame back to MinIO.
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

        # Extract input and output file names
        input_file_name = req.json['inputs']['Dataframe']['value']
        output_file_name = req.json['outputs']['Dataframe']['destination']

        # Read data from pickled file in MinIO
        obj = client.get_object(bucket_name, input_file_name)
        pickled_data = obj.read()

        # Unpickle the DataFrame
        df = pickle.loads(pickled_data)

        # Get the columns to filter (split by comma and remove spaces)
        columns_to_filter = req.json['inputs']['ColumnsToFilter']['value'].replace(" ", "").split(",")
        threshold = float(req.json['inputs']['Threshold']['value'])
        condition = req.json['inputs']['Condition']['value']  # This is assumed to be a string, e.g., "LargerEqual"
        # Apply filtering logic for each column
        for column in columns_to_filter:
            if column in df.columns:
                # Convert the column to numeric if it isn't already (removes commas and converts to float)
                df[column] = df[column].astype(str).str.replace(',', '').astype(float)
                # Apply condition
                if condition == "Equal":
                    mask = df[column] == threshold
                elif condition == "NotEqual":
                    mask = df[column] != threshold
                elif condition == "Larger":
                    mask = df[column] > threshold
                elif condition == "LargerEqual":
                    mask = df[column] >= threshold
                elif condition == "Smaller":
                    mask = df[column] < threshold
                elif condition == "SmallerEqual":
                    mask = df[column] <= threshold
                else:
                    raise ValueError(f"Invalid condition: {condition}")

                # Remove rows where the condition is NOT met (inverse mask)
                df = df[mask]

        # Pickle the modified DataFrame
        with BytesIO() as bytes_buffer:
            pickle.dump(df, bytes_buffer)
            bytes_buffer.seek(0)
            client.put_object(bucket_name, output_file_name, bytes_buffer, len(bytes_buffer.getvalue()))

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
    Main function to handle incoming requests.
    """
    if 'request' in context.keys():
        if context.request.method == "POST":
            output_file = filter_df(context.request)
            print(output_file)
            return context.request.method, 200
        elif context.request.method == "GET":
            return context.request.method, 200
    else:
        print("Empty request", flush=True)
        return "{}", 200
