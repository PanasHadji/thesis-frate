import json
import pandas as pd
from minio import Minio
from minio.error import S3Error
from io import BytesIO
import pickle

def append_tables(req):
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
        first_input_file_name = req.json['inputs']['Dataframe1']['value']
        second_input_file_name = req.json['inputs']['Dataframe2']['value']
        output_file_name = req.json['outputs']['Dataframe']['destination']

        # Read data from pickled files in MinIO
        try:
            obj1 = client.get_object(bucket_name, first_input_file_name)
            pickled_data1 = obj1.read()
            obj2 = client.get_object(bucket_name, second_input_file_name)
            pickled_data2 = obj2.read()
        except S3Error as e:
            raise RuntimeError(f"Error reading files from MinIO: {e}")
        print(first_input_file_name)
        print(second_input_file_name)
        # Unpickle the DataFrame
        try:
            df1 = pickle.loads(pickled_data1)
            df2 = pickle.loads(pickled_data2)
        except pickle.UnpicklingError:
            raise RuntimeError("Error unpickling the DataFrame. The data might not be in the expected format.")

        # Append the DataFrames
        newDf = pd.concat([df1, df2], ignore_index=True)

        # Pickle the modified DataFrame
        with BytesIO() as bytes_buffer:
            pickle.dump(newDf, bytes_buffer)
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
            output_file = append_tables(context.request)
            print(output_file)
            return context.request.method, 200
        elif context.request.method == "GET":
            return context.request.method, 200
    else:
        print("Empty request", flush=True)
        return "{}", 200
