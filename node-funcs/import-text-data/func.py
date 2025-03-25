import json
import pandas as pd
from minio import Minio
from minio.error import S3Error
from io import BytesIO
import pickle
import datetime
import time


def import_text_data(req):
    """
    Read file from MinIO, process data based on mode, and upload the result back to MinIO.
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

        # Extract input and output file names and parameters
        input_file_name = req.json['inputs']['FileName']['value']
        mode = req.json['inputs']['Mode']['value']
        delimiter = req.json['inputs']['Delimiter']['value']
        start_line = req.json['inputs'].get('StartLine', {}).get('value', "") or 0
        end_line = req.json['inputs'].get('EndLine', {}).get('value', "") or 0

        log_event("Input parameters extracted", f"File: {input_file_name}, Mode: {mode}, Delimiter: {delimiter}")

        # Start file read time tracking
        file_read_start_time = time.time()

        # Map delimiter input to actual delimiter characters
        delimiter_map = {"Comma": ",", "Tab": "\t", "Pipe": "|", "Semicolon": ";"}
        delimiter_char = delimiter_map.get(delimiter, ",")

        # Read the file from MinIO
        obj = client.get_object(bucket_name, input_file_name)
        file_data = obj.read().decode('utf-8')

        # Track file read time
        trust_metrics["processing_time"]["file_read"] = abs(time.time() - file_read_start_time)

        log_event("File read from MinIO", f"Size: {len(file_data)} bytes")

        # Process the file based on mode
        if mode == "Delimited":
            options = {"delimiter": delimiter_char}
            header = file_data.splitlines()[0]

            # Handling start and end lines
            if start_line != 0:
                lines = file_data.splitlines()[start_line - 1:]
                lines.insert(0, header)
            else:
                lines = file_data.splitlines()

            if end_line != 0:
                lines = lines[:int(end_line) - (int(start_line) - 1 if start_line else 0)]

            df = pd.read_csv(BytesIO("\n".join(lines).encode()), delimiter=delimiter_char)

            log_event("Processed as Delimited", f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            trust_metrics["data_integrity"]["input_rows"] = len(file_data.splitlines())
            trust_metrics["data_integrity"]["output_rows"] = df.shape[0]

        elif mode == "Fixed":
            lines = file_data.splitlines()

            # Handle start line and end line for fixed mode
            if start_line != 0:
                lines = lines[int(start_line) - 1:]
            if end_line != 0:
                lines = lines[:int(end_line) - (int(start_line) - 1 if start_line else 0)]

            df = pd.DataFrame({"Content": lines})

            log_event("Processed as Fixed", f"Rows: {len(lines)}")
            trust_metrics["data_integrity"]["input_rows"] = len(file_data.splitlines())
            trust_metrics["data_integrity"]["output_rows"] = len(lines)

        else:
            raise ValueError(f"Invalid mode: {mode}. Expected 'Delimited' or 'Fixed'.")

        # Track processing time for data processing phase
        trust_metrics["processing_time"]["processing"] = abs(time.time() - start_time)

        # Pickle the DataFrame and upload to MinIO bucket
        output_file_name = req.json['outputs']['Dataframe']['destination']
        if req.json['outputs']['Dataframe']['type'] == "pickleDf":
            with BytesIO() as bytes_buffer:
                pickle.dump(df, bytes_buffer)
                bytes_buffer.seek(0)
                client.put_object(bucket_name, output_file_name, bytes_buffer, len(bytes_buffer.getvalue()))
        else:
            csv_buffer = df.to_csv(index=False)
            client.put_object(bucket_name, output_file_name, BytesIO(csv_buffer.encode()), len(csv_buffer.encode()))

        log_event("Output file saved", f"Path: {output_file_name}")

        # Track file write time
        trust_metrics["processing_time"]["file_write"] = abs(time.time() - start_time -
                                                             trust_metrics["processing_time"]["file_read"] -
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
            output_file = import_text_data(context.request)
            print(output_file)
            return context.request.method, 200
        elif context.request.method == "GET":
            return context.request.method, 200
    else:
        print("Empty request", flush=True)
        return "{}", 200
