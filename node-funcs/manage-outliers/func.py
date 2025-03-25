import json
import pandas as pd
from minio import Minio
from minio.error import S3Error
from io import BytesIO
import pickle
import scipy.stats as stats
import numpy as np
import datetime
import time


def find_outliers_IQR(series):
    """
    Get the outliers using IQR
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    IQR = q3 - q1
    outliers = series[((series < (q1 - 1.5 * IQR)) | (series > (q3 + 1.5 * IQR)))]
    return outliers


def find_outliers_CI(series):
    """
    Get the outliers using Confidence Interval
    """
    mean = series.mean()
    std_dev = series.std()
    confidence_level = 0.95  # 95% Confidence Interval
    z_value = stats.norm.ppf((1 + confidence_level) / 2)
    ci_lower = mean - z_value * std_dev
    ci_upper = mean + z_value * std_dev
    outliers = series[(series < ci_lower) | (series > ci_upper)]
    return outliers


def find_outliers_ZS(series):
    """
    Get the outliers using Z Score
    """
    z_scores = stats.zscore(series)
    outliers = series[(z_scores < -3) | (z_scores > 3)]
    return outliers


def calculate_statistics(df):
    """
    Calculate statistics for each numeric column in the DataFrame.

    Parameters:
    - df: pd.DataFrame, the input DataFrame

    Returns:
    - stats: dict, dictionary containing statistics for each column
    - total_values: int, total number of non-null values
    - outliers_counts: dict, total number of outliers for each method
    """
    numeric_df = df.select_dtypes(include='number')
    stats = {}
    total_values = 0
    outliers_counts = {'IQR': 0, 'CI': 0, 'ZS': 0}

    for column in numeric_df.columns:
        col_data = numeric_df[column].dropna()

        # Determine outliers using boolean masks
        outliers_IQR = find_outliers_IQR(col_data).index
        outliers_CI = find_outliers_CI(col_data).index
        outliers_ZS = find_outliers_ZS(col_data).index

        is_outlier = np.zeros(len(col_data), dtype=bool)
        is_outlier[outliers_IQR] = True
        is_outlier[outliers_CI] = True
        is_outlier[outliers_ZS] = True

        stats[column] = {
            'Average': col_data.mean(),
            'Outliers IQR': outliers_IQR.tolist(),
            'Outlier Percentage IQR': (len(outliers_IQR) / len(col_data)) * 100 if len(col_data) > 0 else 0,
            'Outliers CI': outliers_CI.tolist(),
            'Outlier Percentage CI': (len(outliers_CI) / len(col_data)) * 100 if len(col_data) > 0 else 0,
            'Outliers ZS': outliers_ZS.tolist(),
            'Outlier Percentage ZS': (len(outliers_ZS) / len(col_data)) * 100 if len(col_data) > 0 else 0,
            'Outliers Fusion': is_outlier.tolist()
        }

        total_values += len(col_data)
        outliers_counts['IQR'] += len(outliers_IQR)
        outliers_counts['CI'] += len(outliers_CI)
        outliers_counts['ZS'] += len(outliers_ZS)

    return stats, total_values, outliers_counts


def manage_outliers(req):
    """
    Manage outliers in the DataFrame based on the specified method from the request.
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
        start_time = time.time()

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

        input_stats_file_name = req.json['inputs']['Dataframe']['value']
        output_file_name = req.json['outputs']['Dataframe']['destination']

        log_event("Input parameters extracted",
                  f"File: {input_stats_file_name}, Output file: {output_file_name}, Output outlier file: {output_file_name}_outlier_stats")

        # Start file read time tracking
        file_read_start_time = time.time()

        obj_stats = client.get_object(bucket_name, input_stats_file_name)
        pickled_stats_data = obj_stats.read()

        # Track file read time
        trust_metrics["processing_time"]["file_read"] = abs(time.time() - file_read_start_time)
        log_event("File read from MinIO", f"Size: {len(pickled_stats_data)} bytes")

        df = pickle.loads(pickled_stats_data)

        stats, total_values, outliers_counts = calculate_statistics(df)

        method = req.json['inputs']['Method']['value']
        if method == "Trim":
            log_event("Applying method", f"Method: {method}")
            df = apply_outlier_trim(df, stats)
        elif method == "Cap":
            df = apply_outlier_cap(df, stats)
            log_event("Applying method", f"Method: {method}")
        elif method == "Winsorize":
            df = apply_outlier_winsorize(df, stats)
            log_event("Applying method", f"Method: {method}")
        else:
            raise ValueError(f"Unknown outlier method: {method}")

        # Create an outlier summary DataFrame
        outlier_data = []
        for column, column_stats in stats.items():
            log_event("Creating an outlier summary for column", f"Column: {column}")
            outlier_data.append({
                'Column': column,
                'Outliers IQR': len(column_stats['Outliers IQR']),
                'Outliers CI': len(column_stats['Outliers CI']),
                'Outliers ZS': len(column_stats['Outliers ZS']),
                'Outliers Fusion': column_stats['Outliers Fusion'].count(True),
                'Confidence Interval Lower': column_stats.get('CI_Lower', None),
                'Confidence Interval Upper': column_stats.get('CI_Upper', None)
            })

        # Track processing time for data processing phase
        trust_metrics["processing_time"]["processing"] = abs(time.time() - start_time)

        # Save the processed data
        with BytesIO() as bytes_buffer:
            pickle.dump(df, bytes_buffer)
            bytes_buffer.seek(0)
            client.put_object(bucket_name, output_file_name, bytes_buffer, len(bytes_buffer.getvalue()))

        outlier_df = pd.DataFrame(outlier_data)

        # Convert outlier DataFrame to CSV
        csv_buffer = outlier_df.to_csv(index=False)
        client.put_object(
            bucket_name,
            f"{output_file_name}_outlier_stats.csv",
            BytesIO(csv_buffer.encode()),
            len(csv_buffer)
        )

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
        _save_logs_and_metrics(logs, trust_metrics, client, bucket_name, req.json['outputs']['Dataframe']['destination'].split('/')[0])
        raise RuntimeError(f"Missing key in JSON input: {exc}")

    except S3Error as exc:
        log_event("Error", f"MinIO error: {exc}")
        trust_metrics["failure"] += 1  # Increment failure count
        trust_metrics["error_frequency"][str(exc)] = trust_metrics["error_frequency"].get(str(exc), 0) + 1
        _save_logs_and_metrics(logs, trust_metrics, client, bucket_name, req.json['outputs']['Dataframe']['destination'].split('/')[0])
        raise RuntimeError(f"An error occurred while reading from or uploading to MinIO: {exc}")

    except Exception as exc:
        log_event("Error", f"General processing error: {exc}")
        trust_metrics["failure"] += 1  # Increment failure count
        trust_metrics["error_frequency"][str(exc)] = trust_metrics["error_frequency"].get(str(exc), 0) + 1
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


def apply_outlier_trim(df, stats):
    """
    Trim outliers in the DataFrame by replacing with NaNs.
    """
    numeric_df = df.select_dtypes(include='number')
    for column in numeric_df.columns:
        is_outlier = pd.Series(stats[column]['Outliers Fusion'], index=numeric_df.index)
        df.loc[is_outlier, column] = np.nan
    return df


def apply_outlier_cap(df, stats):
    """
    Cap outliers in the DataFrame.
    """
    numeric_df = df.select_dtypes(include='number')
    for column in numeric_df.columns:
        is_outlier = pd.Series(stats[column]['Outliers Fusion'], index=numeric_df.index)
        q1 = numeric_df[column].quantile(0.25)
        q3 = numeric_df[column].quantile(0.75)
        df.loc[is_outlier & (numeric_df[column] < q1), column] = q1
        df.loc[is_outlier & (numeric_df[column] > q3), column] = q3
    return df


def apply_outlier_winsorize(df, stats):
    """
    Winsorize outliers in the DataFrame.
    """
    numeric_df = df.select_dtypes(include='number')
    for column in numeric_df.columns:
        is_outlier = pd.Series(stats[column]['Outliers Fusion'], index=numeric_df.index)
        q1 = numeric_df[column].quantile(0.25)
        q3 = numeric_df[column].quantile(0.75)
        df[column] = np.where(is_outlier, df[column].clip(lower=q1, upper=q3), df[column])
    return df


# Main function to handle incoming requests.
def main(context):
    """
    Main function to handle incoming requests.
    """
    if 'request' in context.keys():
        if context.request.method == "POST":
            print(manage_outliers(context.request))
            return context.request.method, 200
        elif context.request.method == "GET":
            return context.request.method, 200
    else:
        print("Empty request", flush=True)
        return "{}", 200
