from minio import Minio
from minio.error import S3Error
from io import BytesIO
#from chart_builders.dataset_chart_builder import create_files_for_dataframe
import pickle
import json
import datetime
import time
import pandas as pd


def impute_missing_values(req):
    """
    Impute missing values in a DataFrame read from MinIO and upload the imputed DataFrame back to MinIO.
    """
    logs = []  # List to store log entries
    trust_metrics = {
        "success": 0,
        "failure": 0,
        "warnings": 0,
        "processing_time": {"total": 0, "file_read": 0, "processing": 0, "file_write": 0, "chart_generation": 0},  # Track time for each phase
        "data_integrity": {"input_rows": 0, "output_rows": 0},
        "error_frequency": {}  # Track the count of different error types
    }

    def log_event(action, details=""):
        """Helper function to log events with timestamps."""
        logs.append({"timestamp": datetime.datetime.utcnow().isoformat(), "action": action, "details": details})

    try:
        # Start processing time tracking
        start_time = time.time()

        # Extract configuration
        access_key = req.json['config']['access_key']['value']
        secret_key = req.json['config']['secret_key']['value']
        bucket_name = req.json['config']['bucket_name']['value']

        log_event("Extracted configuration", f"Bucket: {bucket_name}")

        # Create a MinIO client
        client = Minio(
            "minio:24001",
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )

        log_event("MinIO client initialized")

        # Extract input and output file names
        input_file_name = req.json['inputs']['Dataframe']['value']
        output_file_name = req.json['outputs']['Dataframe']['destination']

        log_event("Input parameters extracted", f"File: {input_file_name}, Output: {output_file_name}")

        # Start file read time tracking
        file_read_start_time = time.time()

        # Read data from pickled file in MinIO
        obj = client.get_object(bucket_name, input_file_name)
        pickled_data = obj.read()

        # Track file read time
        trust_metrics["processing_time"]["file_read"] = abs(time.time() - file_read_start_time)

        log_event("File read from MinIO", f"Size: {len(pickled_data)} bytes")

        # Unpickle the DataFrame
        df = pickle.loads(pickled_data)

        # Track data integrity
        trust_metrics["data_integrity"]["input_rows"] = len(df)

        # Loop through all columns in the DataFrame for imputation
        for col in df.columns:
            if df[col].isnull().any():
                # Log the columns with missing values
                log_event("Imputation started", f"Column: {col}, Missing Values: {df[col].isnull().sum()}")

                if df[col].dtype == 'object':
                    # For categorical columns, fill missing values with the mode
                    mode_value = df[col].mode().iloc[0]
                    df[col] = df[col].fillna(mode_value)
                    log_event("Imputation applied", f"Column: {col}, Imputed with Mode Value: {mode_value}")
                else:
                    # For numeric columns, choose imputation method based on distribution and skew
                    skewness = df[col].skew()

                    if skewness < -1 or skewness > 1:
                        # For highly skewed columns, fill missing values with median
                        imputation_value = df[col].median()
                        log_event("Imputation applied", f"Column: {col}, Imputed with Median Value: {imputation_value} (Skewness: {skewness})")
                    else:
                        # For normally distributed or moderately skewed columns, fill missing values with mean
                        imputation_value = df[col].mean()
                        log_event("Imputation applied", f"Column: {col}, Imputed with Mean Value: {imputation_value} (Skewness: {skewness})")

                    df[col] = df[col].fillna(imputation_value)

        # Track processing time for data processing phase
        trust_metrics["processing_time"]["processing"] = abs(time.time() - start_time)

        # Pickle the modified DataFrame
        with BytesIO() as bytes_buffer:
            pickle.dump(df, bytes_buffer)
            bytes_buffer.seek(0)
            client.put_object(bucket_name, output_file_name, bytes_buffer, len(bytes_buffer.getvalue()))

        # Track file write time
        trust_metrics["processing_time"]["file_write"] = abs(time.time() - start_time -
                                                             trust_metrics["processing_time"]["file_read"] -
                                                             trust_metrics["processing_time"]["processing"])
        trust_metrics["processing_time"]["total"] = time.time()

        log_event("Output file saved", f"Path: {output_file_name}")

        # Start chart generation time tracking
        chart_generation_start_time = time.time()

        # Create additional files for the DataFrame (charts)
        #create_files_for_dataframe(output_file_name.split('/')[0], df, client, bucket_name)

        # Track chart generation time
        trust_metrics["processing_time"]["chart_generation"] = max(0, time.time() - chart_generation_start_time)

        log_event("Charts generated for DataFrame")

        # Success tracking
        trust_metrics["success"] += 1
        log_event("Processing completed successfully", f"Output file: {output_file_name}")

        # Save logs and trust metrics to MinIO
        log_output_path = req.json['outputs']['Dataframe']['destination'].split('/')[0] + '/logs.csv'
        log_df = pd.DataFrame(logs)

        with BytesIO() as log_buffer:
            log_df.to_csv(log_buffer, index=False)
            log_buffer.seek(0)
            client.put_object(bucket_name, log_output_path, log_buffer, len(log_buffer.getvalue()))

        log_event("Logs saved to MinIO", f"Path: {log_output_path}")

        # Save trust metrics to MinIO
        trust_metrics_output_path = req.json['outputs']['Dataframe']['destination'].split('-')[0] + '/' + \
                                    req.json['outputs']['Dataframe']['destination'].split('-')[1] + '-trust_metrics.json'
        with BytesIO() as trust_buffer:
            trust_buffer.write(json.dumps(trust_metrics).encode())
            trust_buffer.seek(0)
            client.put_object(bucket_name, trust_metrics_output_path, trust_buffer, len(trust_buffer.getvalue()))

        log_event("Trust metrics saved to MinIO", f"Path: {trust_metrics_output_path}")

        return output_file_name

    except KeyError as exc:
        log_event("Error", f"Missing key in JSON input: {exc}")
        trust_metrics["failure"] += 1  # Increment failure count
        trust_metrics["error_frequency"][str(exc)] = trust_metrics["error_frequency"].get(str(exc), 0) + 1
        # Save logs and metrics before raising the exception
        _save_logs_and_metrics(logs, trust_metrics, client, bucket_name, req.json['outputs']['Dataframe']['destination'].split('/')[0])
        raise RuntimeError(f"Missing key in JSON input: {exc}")

    except S3Error as exc:
        log_event("Error", f"MinIO error: {exc}")
        trust_metrics["failure"] += 1  # Increment failure count
        trust_metrics["error_frequency"][str(exc)] = trust_metrics["error_frequency"].get(str(exc), 0) + 1
        # Save logs and metrics before raising the exception
        _save_logs_and_metrics(logs, trust_metrics, client, bucket_name, req.json['outputs']['Dataframe']['destination'].split('/')[0])
        raise RuntimeError(f"An error occurred while reading from or uploading to MinIO: {exc}")

    except Exception as exc:
        log_event("Error", f"General processing error: {exc}")
        trust_metrics["failure"] += 1  # Increment failure count
        trust_metrics["error_frequency"][str(exc)] = trust_metrics["error_frequency"].get(str(exc), 0) + 1
        # Save logs and metrics before raising the exception
        _save_logs_and_metrics(logs, trust_metrics, client, bucket_name, req.json['outputs']['Dataframe']['destination'].split('/')[0])
        raise RuntimeError(f"Error processing file: {exc}")


def _save_logs_and_metrics(logs, trust_metrics, client, bucket_name, path):
    """
    Helper function to save logs and trust metrics to MinIO.
    """
    log_output_path = path + "/logs.csv"
    trust_metrics_output_path = "trust_metrics.json"

    log_df = pd.DataFrame(logs)
    with BytesIO() as log_buffer:
        log_df.to_csv(log_buffer, index=False)
        log_buffer.seek(0)
        client.put_object(bucket_name, log_output_path, log_buffer, len(log_buffer.getvalue()))

    with BytesIO() as trust_buffer:
        trust_buffer.write(json.dumps(trust_metrics).encode())
        trust_buffer.seek(0)
        client.put_object(bucket_name, trust_metrics_output_path, trust_buffer, len(trust_buffer.getvalue()))


# Main function to handle incoming requests.
def main(context):
    """
    Main function to handle incoming requests.
    """
    if 'request' in context.keys():
        if context.request.method == "POST":
            output_file = impute_missing_values(context.request)
            print(output_file)
            return context.request.method, 200
        elif context.request.method == "GET":
            return context.request.method, 200
    else:
        print("Empty request", flush=True)
        return "{}", 200
