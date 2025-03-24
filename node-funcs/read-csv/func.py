import json
import pandas as pd
from minio import Minio
from minio.error import S3Error
from io import BytesIO
import pickle


def read_file(req):
    """
    Read CSV file from MinIO, process data, and upload the result back to MinIO.
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
        input_file_name = req.json['inputs']['FileName']['value']

        # Read data from CSV file in MinIO with Pandas
        obj = client.get_object(bucket_name, input_file_name)
        csv_data = obj.read()

        # Convert bytes data to DataFrame with Pandas
        df = pd.read_csv(BytesIO(csv_data))

        # Pickle the DataFrame and upload to MinIO bucket
        output_file_name = req.json['outputs']['Dataframe']['destination']
        if req.json['outputs']['Dataframe']['type'] == "pickleDf":
            with BytesIO() as bytes_buffer:
                pickle.dump(df, bytes_buffer)
                bytes_buffer.seek(0)
                client.put_object(bucket_name, output_file_name, bytes_buffer, len(bytes_buffer.getvalue()))
        else:
            # Convert DataFrame to CSV in memory
            csv_buffer = df.to_csv(index=False)
            client.put_object(bucket_name, output_file_name, BytesIO(csv_buffer.encode()), len(csv_buffer.encode()))

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
            output_file = read_file(context.request)
            print(output_file)
            return context.request.method, 200
        elif context.request.method == "GET":
            return context.request.method, 200
    else:
        print("Empty request", flush=True)
        return "{}", 200
