import json
import pandas as pd
from minio import Minio
from minio.error import S3Error
from io import BytesIO
import pickle

def sort_columns(req):
    """
    Impute missing values in a DataFrame read from MinIO and upload the imputed DataFrame back to MinIO.
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

        # Read data from pickled file in MinIO
        obj = client.get_object(bucket_name, input_file_name)
        pickled_data = obj.read()

        # Unpickle the DataFrame
        df = pickle.loads(pickled_data)

        # Extract columns to sort
        columns_to_sort = req.json['inputs']['ColumnsToSort']['value']

        # Process the column names and sort directions
        sort_instructions = [
            (col.strip().rsplit(' ', 1)[0],
             col.strip().rsplit(' ', 1)[1].lower() if ' ' in col.strip() else 'desc')
            for col in columns_to_sort.split(',')
        ]

        # Separate column names and sort orders
        columns = [col[0] for col in sort_instructions]
        sort_orders = [True if order == 'asc' else False for _, order in sort_instructions]

        # Check for missing columns and raise an error if any are not found
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columns not found in the DataFrame: {', '.join(missing_columns)}")

        # Sort the DataFrame
        df.sort_values(by=columns, ascending=sort_orders, inplace=True)

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
            output_file = sort_columns(context.request)
            print(output_file)
            return context.request.method, 200
        elif context.request.method == "GET":
            return context.request.method, 200
    else:
        print("Empty request", flush=True)
        return "{}", 200
