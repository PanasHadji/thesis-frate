import json
import pandas as pd
import time
import datetime
from minio import Minio
from minio.error import S3Error
from io import BytesIO
import pickle


def export_to_file(req):
    """
    Export data to a file (CSV or Fixed-width) and upload it to MinIO.
    """
    logs = []  # List to store log entries
    trust_metrics = {
        "success": 0,
        "failure": 0,
        "warnings": 0,
        "processing_time": {"total": 0, "file_read": 0, "processing": 0, "file_write": 0},  # Track time for each phase
        "data_integrity": {"input_rows": 0, "output_rows": 0},
        "error_frequency": {}  # Track the count of different error types
    }

    def log_event(action, details=""):
        """Helper function to log events with timestamps."""
        logs.append({"timestamp": datetime.datetime.utcnow().isoformat(), "action": action, "details": details})

    try:
        # Extract configuration from parsed JSON input
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

        # Extract input file name and output file names from JSON input
        input_file_name = req.json['inputs']['Dataframe']['value']
        output_dataframe_name = req.json['outputs']['Dataframe']['destination']
        output_file_name = req.json['outputs']['FileName']['destination']
        mode = req.json['inputs']['Mode']['value']

        log_event("Input parameters extracted", f"File: {input_file_name}, Mode: {mode}")

        # Start file read time tracking
        file_read_start_time = time.time()

        # Read data from pickled file in MinIO
        obj = client.get_object(bucket_name, input_file_name)
        pickled_data = obj.read()

        # Unpickle the DataFrame
        df = pickle.loads(pickled_data)

        # Track file read time
        trust_metrics["processing_time"]["file_read"] = abs(time.time() - file_read_start_time)

        log_event("File read from MinIO", f"Size: {len(df)} bytes")

        # Map delimiter input to actual delimiter characters
        delimiter_map = {
            "Comma": ",",
            "Tab": "\t",
            "Pipe": "|",
            "Semicolon": ";"
        }

        # Extract delimiter and get corresponding character
        delimiter = req.json['inputs']['Delimiter']['value']
        delimiter_char = delimiter_map.get(delimiter, ",")
        log_event("Delimiter extracted", f"Chosen delimiter: {delimiter_char}")

        # Track the start of the export processing time
        export_start_time = time.time()

        # Check the mode and export accordingly
        if mode == "Delimited":
            # Export to CSV with the chosen delimiter
            csv_buffer = df.to_csv(index=False, sep=delimiter_char)
            file_extension = '.csv'
            log_event("Exporting data", f"Exporting to CSV with delimiter: {delimiter_char}")
        elif mode == "Text":
            # Export to fixed-width format (for simplicity, treating it as a simple text export here)
            text_buffer = df.to_string(index=False, header=False, line_width=100)
            csv_buffer = text_buffer
            file_extension = '.txt'
            log_event("Exporting data", "Exporting to fixed-width text format")

        # Track the export processing time
        trust_metrics["processing_time"]["processing"] = time.time() - export_start_time
        log_event("Export process completed", f"Exported file with extension: {file_extension}")

        # Upload the file (CSV or Text) to MinIO
        client.put_object(bucket_name, output_file_name + file_extension, BytesIO(csv_buffer.encode()), len(csv_buffer))

        # Optionally, upload the DataFrame back in pickled format
        with BytesIO() as bytes_buffer:
            pickle.dump(df, bytes_buffer)
            bytes_buffer.seek(0)
            client.put_object(bucket_name, output_dataframe_name, bytes_buffer, len(bytes_buffer.getvalue()))

        log_event("Output file saved", f"Path: {output_file_name}")

        # Track file write time
        trust_metrics["processing_time"]["file_write"] = abs(
            time.time() - export_start_time - trust_metrics["processing_time"]["file_read"] -
            trust_metrics["processing_time"]["processing"])
        trust_metrics["processing_time"]["total"] = time.time()

        # Success tracking
        trust_metrics["success"] += 1
        log_event("Processing completed successfully", f"Output file: {output_file_name}")

        # Save logs and trust metrics at the very end
        log_output_path = req.json['outputs']['Dataframe']['destination'].split('/')[0] + '/logs.csv'
        log_df = pd.DataFrame(logs)

        with BytesIO() as log_buffer:
            log_df.to_csv(log_buffer, index=False)
            log_buffer.seek(0)
            client.put_object(bucket_name, log_output_path, log_buffer, len(log_buffer.getvalue()))

        log_event("Logs saved to MinIO", f"Path: {log_output_path}")

        # Save trust metrics to MinIO
        trust_metrics_output_path = req.json['outputs']['Dataframe']['destination'].split('-')[0] + '/' + \
                                    req.json['outputs']['Dataframe']['destination'].split('-')[
                                        1] + '-trust_metrics.json'
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
        _save_logs_and_metrics(logs, trust_metrics, client, bucket_name)
        raise RuntimeError(f"Missing key in JSON input: {exc}")

    except S3Error as exc:
        log_event("Error", f"MinIO error: {exc}")
        trust_metrics["failure"] += 1  # Increment failure count
        trust_metrics["error_frequency"][str(exc)] = trust_metrics["error_frequency"].get(str(exc), 0) + 1
        # Save logs and metrics before raising the exception
        _save_logs_and_metrics(logs, trust_metrics, client, bucket_name)
        raise RuntimeError(f"An error occurred while reading from or uploading to MinIO: {exc}")

    except Exception as exc:
        log_event("Error", f"General processing error: {exc}")
        trust_metrics["failure"] += 1  # Increment failure count
        trust_metrics["error_frequency"][str(exc)] = trust_metrics["error_frequency"].get(str(exc), 0) + 1
        # Save logs and metrics before raising the exception
        _save_logs_and_metrics(logs, trust_metrics, client, bucket_name)
        raise RuntimeError(f"Error processing file: {exc}")


def _save_logs_and_metrics(logs, trust_metrics, client, bucket_name):
    """Helper function to save logs and metrics to MinIO"""
    try:
        log_output_path = 'logs.csv'
        log_df = pd.DataFrame(logs)

        with BytesIO() as log_buffer:
            log_df.to_csv(log_buffer, index=False)
            log_buffer.seek(0)
            client.put_object(bucket_name, log_output_path, log_buffer, len(log_buffer.getvalue()))

        trust_metrics_output_path = 'trust_metrics.json'
        with BytesIO() as trust_buffer:
            trust_buffer.write(json.dumps(trust_metrics).encode())
            trust_buffer.seek(0)
            client.put_object(bucket_name, trust_metrics_output_path, trust_buffer, len(trust_buffer.getvalue()))

    except Exception as e:
        print(f"Error saving logs or metrics: {e}")


# Main function to handle incoming requests.
def main(context):
    """
    Main function to handle incoming requests.
    """
    if 'request' in context.keys():
        if context.request.method == "POST":
            output_file = export_to_file(context.request)
            print(output_file)
            return context.request.method, 200
        elif context.request.method == "GET":
            return context.request.method, 200
    else:
        print("Empty request", flush=True)
        return "{}", 200
