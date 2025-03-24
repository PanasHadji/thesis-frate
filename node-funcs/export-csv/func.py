import json
import pandas as pd
from minio import Minio
from minio.error import S3Error
from io import BytesIO
import pickle


def export_csv(req):
    """
    Export data to a CSV file and upload it to MinIO.
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

        # Extract input file name from JSON input
        input_file_name = req.json['inputs']['Dataframe']['value']
        output_dataframe_name = req.json['outputs']['Dataframe']['destination']
        output_file_name = req.json['outputs']['FileName']['destination']

        # Read data from pickled file in MinIO
        obj = client.get_object(bucket_name, input_file_name)
        pickled_data = obj.read()

        # Unpickle the DataFrame
        df = pickle.loads(pickled_data)

        # Convert DataFrame to CSV in memory
        csv_buffer = df.to_csv(index=False)

        # Upload the CSV data to MinIO bucket
        client.put_object(bucket_name, output_file_name+'.csv', BytesIO(csv_buffer.encode()), len(csv_buffer))

        with BytesIO() as bytes_buffer:
            pickle.dump(df, bytes_buffer)
            bytes_buffer.seek(0)
            client.put_object(bucket_name, output_dataframe_name, bytes_buffer, len(bytes_buffer.getvalue()))

        return output_dataframe_name
    except KeyError as exc:
        raise RuntimeError(f"Missing key in JSON input: {exc}")
    except S3Error as exc:
        raise RuntimeError(f"An error occurred while uploading to MinIO: {exc}")
    except Exception as exc:
        raise RuntimeError(f"Error processing file: {exc}")


# Main function to handle incoming requests.
def main(context):
    """
    Main function to handle incoming requests.
    """
    if 'request' in context.keys():
        if context.request.method == "POST":
            output_file = export_csv(context.request)
            print(output_file)
            return context.request.method, 200
        elif context.request.method == "GET":
            return context.request.method, 200
    else:
        print("Empty request", flush=True)
        return "{}", 200
