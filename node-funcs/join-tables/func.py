import json
import pandas as pd
from minio import Minio
from minio.error import S3Error
from io import BytesIO
import pickle

def join_tables(req):
    """
    Join multiple DataFrames read from MinIO and upload the joined DataFrame back to MinIO.
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

        # Extract input and output file names and join method
        first_input_file_name = req.json['inputs']['Dataframe1']['value']
        second_input_file_name = req.json['inputs']['Dataframe2']['value']
        join_method = req.json['inputs']['Method']['value'].lower()  # 'inner', 'left', etc.
        on_column = req.json['inputs'].get('OnColumn', {}).get('value', None)  # The column to join on (optional)
        output_file_name = req.json['outputs']['Dataframe']['destination']

        # Read data from pickled files in MinIO
        try:
            obj1 = client.get_object(bucket_name, first_input_file_name)
            pickled_data1 = obj1.read()
            obj2 = client.get_object(bucket_name, second_input_file_name)
            pickled_data2 = obj2.read()
        except S3Error as e:
            raise RuntimeError(f"Error reading files from MinIO: {e}")

        # Unpickle the DataFrames
        try:
            df1 = pickle.loads(pickled_data1)
            df2 = pickle.loads(pickled_data2)
        except pickle.UnpicklingError:
            raise RuntimeError("Error unpickling the DataFrame. The data might not be in the expected format.")

        # Set the join column as index if specified
        if on_column:
            df1.set_index(on_column, inplace=True)
            df2.set_index(on_column, inplace=True)

        # Find overlapping columns between df1 and df2
        overlapping_columns = df1.columns.intersection(df2.columns)

        # Perform the join operation based on the specified method
        if join_method == "inner":
            joined_df = df1.join(df2, how="inner")
        elif join_method == "left":
            joined_df = df1.join(df2, how="left")
        elif join_method == "right":
            joined_df = df1.join(df2, how="right")
        elif join_method == "outer":
            joined_df = df1.join(df2, how="outer")
        else:
            raise ValueError(f"Invalid join method: {join_method}")

        # Add suffixes to overlapping columns
        for col in overlapping_columns:
            if col in joined_df.columns:
                joined_df.rename(columns={col: col + '_df1'}, inplace=True)
                df2.rename(columns={col: col + '_df2'}, inplace=True)

        # Perform the join operation again with the updated column names
        joined_df = df1.join(df2, how=join_method)

        # Compare second row to the header and remove it if they are identical
        if joined_df.iloc[0].values.flatten().tolist() == joined_df.columns.tolist():
            joined_df = joined_df.iloc[1:]
        print(joined_df)
        # Pickle the joined DataFrame
        with BytesIO() as bytes_buffer:
            pickle.dump(joined_df, bytes_buffer)
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
            output_file = join_tables(context.request)
            print(output_file)
            return context.request.method, 200
        elif context.request.method == "GET":
            return context.request.method, 200
    else:
        print("Empty request", flush=True)
        return "{}", 200
