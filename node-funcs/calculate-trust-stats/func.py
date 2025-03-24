from fpdf import FPDF
import json
import pandas as pd
from minio import Minio
from minio.error import S3Error
from io import BytesIO
import pickle
import scipy.stats as stats
import requests
import numpy as np

# Define the outlier detection methods
def find_outliers_IQR(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    IQR = q3 - q1
    outliers = series[((series < (q1 - 1.5 * IQR)) | (series > (q3 + 1.5 * IQR)))]
    return outliers


def find_outliers_CI(series):
    mean = series.mean()
    std_dev = series.std()
    confidence_level = 0.95
    z_value = stats.norm.ppf((1 + confidence_level) / 2)
    ci_lower = mean - z_value * std_dev
    ci_upper = mean + z_value * std_dev
    outliers = series[(series < ci_lower) | (series > ci_upper)]
    return outliers


def find_outliers_ZS(series):
    z_scores = stats.zscore(series)
    outliers = series[(z_scores < -3) | (z_scores > 3)]
    return outliers


def calculate_statistics(df):
    numeric_df = df.select_dtypes(include='number')
    stats = {}
    total_values = 0
    outliers_counts = {'IQR': 0, 'CI': 0, 'ZS': 0}

    for column in numeric_df.columns:
        col_data = numeric_df[column].dropna()
        outliers_IQR = find_outliers_IQR(col_data)
        outliers_CI = find_outliers_CI(col_data)
        outliers_ZS = find_outliers_ZS(col_data)

        stats[column] = {
            'Average': col_data.mean(),
            'Null Count': numeric_df[column].isnull().sum(),
            'Null Percentage': (numeric_df[column].isnull().sum() / len(numeric_df[column])) * 100,
            'Outliers IQR': outliers_IQR.tolist(),
            'Outlier Percentage IQR': (len(outliers_IQR) / len(col_data)) * 100 if len(col_data) > 0 else 0,
            'Outliers CI': outliers_CI.tolist(),
            'Outlier Percentage CI': (len(outliers_CI) / len(col_data)) * 100 if len(col_data) > 0 else 0,
            'Outliers ZS': outliers_ZS.tolist(),
            'Outlier Percentage ZS': (len(outliers_ZS) / len(col_data)) * 100 if len(col_data) > 0 else 0,
        }

        total_values += len(col_data)
        outliers_counts['IQR'] += len(outliers_IQR)
        outliers_counts['CI'] += len(outliers_CI)
        outliers_counts['ZS'] += len(outliers_ZS)

    return stats, total_values, outliers_counts


def calculate_average_metrics(output_evaluation_json_list):
    """
    Calculate average memory usage and execution time from the JSON data.

    Args:
        output_evaluation_json_list (list): List of JSON objects containing execution metrics.

    Returns:
        float: Average memory usage in MB.
        float: Average execution time in seconds.
    """
    total_memory_usage = 0.0
    total_execution_time = 0.0
    count = len(output_evaluation_json_list)

    for json_data in output_evaluation_json_list:
        total_memory_usage += json_data["ResourceUsage"]["MemoryUsageBytes"] / (1024 * 1024)  # Convert to MB
        total_execution_time += json_data["ExecutionTimeMs"] / 1000.0  # Convert to seconds

    average_memory_usage = total_memory_usage / count if count > 0 else 0.0
    average_execution_time = total_execution_time / count if count > 0 else 0.0

    return average_memory_usage, average_execution_time


def calculate_trustworthiness_index(null_percentage, outlier_percentages, output_metrics_json_list,
                                    output_evaluation_json_list, prometheus_metrics):
    """
    Calculate the Trustworthiness Index using a more intelligent penalty approach.

    Args:
        null_percentage (float): Percentage of null values in the dataset.
        outlier_percentages (dict): Outlier percentages for IQR, CI, and Z-Score methods.
        output_metrics_json_list (list): List of JSON objects containing output metrics.
        output_evaluation_json_list (list): List of JSON objects containing execution metrics.

    Returns:
        float: Trustworthiness Index.
        dict: Breakdown of penalties.
    """
    # Start with 100%
    trustworthiness_index = 100.0

    # Subtract null percentage
    null_penalty = null_percentage
    trustworthiness_index -= null_penalty

    # Subtract average outlier percentage
    average_outlier_percentage = sum(outlier_percentages.values()) / len(outlier_percentages)
    outlier_penalty = average_outlier_percentage
    trustworthiness_index -= outlier_penalty

    # Subtract penalties for failures
    failure_penalty = 0.0
    for json_data in output_metrics_json_list:
        failure_penalty += json_data.get('failure', 0)
    trustworthiness_index -= failure_penalty

    # Calculate average memory usage and execution time
    average_memory_usage, average_execution_time = calculate_average_metrics(output_evaluation_json_list)

    # Subtract penalties for memory spikes
    memory_spike_penalty = 0.0
    memory_threshold = average_memory_usage * 1.10  # 10% above average
    for json_data in output_evaluation_json_list:
        memory_usage = json_data["ResourceUsage"]["MemoryUsageBytes"] / (1024 * 1024)  # Convert to MB
        if memory_usage > memory_threshold:
            memory_increase = ((memory_usage - memory_threshold) / memory_threshold) * 100
            memory_spike_penalty += memory_increase * 0.01
    trustworthiness_index -= memory_spike_penalty

    # Subtract penalties for execution time spikes
    time_spike_penalty = 0.0
    time_threshold = average_execution_time + 1.0  # Average + 1 second
    for json_data in output_evaluation_json_list:
        execution_time = json_data["ExecutionTimeMs"] / 1000.0  # Convert to seconds
        if execution_time > time_threshold:
            time_increase = ((execution_time - time_threshold) / time_threshold) * 100
            time_spike_penalty += time_increase * 0.01
    trustworthiness_index -= time_spike_penalty

    # Incorporate Prometheus metrics
    prometheus_penalty = 0.0
    prometheus_penalty_breakdown = {}
    
    # Helper function to handle None values
    def get_metric(metrics, key, default=0):
        value = metrics.get(key, default)
        return value if value is not None else default
    
    # Example: Penalty for high HTTP request duration
    http_request_duration_sum = get_metric(prometheus_metrics, 'http_request_duration_seconds_sum')
    if http_request_duration_sum > 10:  # Threshold of 10 seconds
        penalty = min((http_request_duration_sum - 10) * 0.5, 10)  # 0.5% penalty per second above threshold, max 10%
        prometheus_penalty += penalty
        prometheus_penalty_breakdown['HTTP Request Duration'] = penalty
    
    # Example: Penalty for high memory usage
    process_memory_bytes = get_metric(prometheus_metrics, 'process_resident_memory_bytes')
    if process_memory_bytes > 1e9:  # Threshold of 1 GB
        penalty = min((process_memory_bytes - 1e9) / 1e8 * 0.1, 5)  # 0.1% penalty per 100 MB above threshold, max 5%
        prometheus_penalty += penalty
        prometheus_penalty_breakdown['Memory Usage'] = penalty
    
    # Example: Penalty for high CPU usage
    process_cpu_seconds = get_metric(prometheus_metrics, 'process_cpu_seconds_total')
    if process_cpu_seconds > 100:  # Threshold of 100 seconds
        penalty = min((process_cpu_seconds - 100) * 0.1, 5)  # 0.1% penalty per second above threshold, max 5%
        prometheus_penalty += penalty
        prometheus_penalty_breakdown['CPU Usage'] = penalty
    
    # Example: Penalty for MinIO bucket usage
    minio_bucket_usage = get_metric(prometheus_metrics, 'minio_bucket_usage_total_bytes')
    if minio_bucket_usage > 5e9:  # Threshold of 5 GB
        penalty = min((minio_bucket_usage - 5e9) / 1e9 * 0.2, 5)  # 0.2% penalty per GB above threshold, max 5%
        prometheus_penalty += penalty
        prometheus_penalty_breakdown['MinIO Bucket Usage'] = penalty
    
    # Example: Penalty for PostgreSQL errors
    postgres_errors = get_metric(prometheus_metrics, 'postgres_exporter_errors_total')
    if postgres_errors > 0:
        penalty = min(postgres_errors * 0.5, 5)  # 0.5% penalty per error, max 5%
        prometheus_penalty += penalty
        prometheus_penalty_breakdown['PostgreSQL Errors'] = penalty
    
    trustworthiness_index -= prometheus_penalty

    # Ensure trustworthiness index does not go below 0
    trustworthiness_index = max(trustworthiness_index, 0.0)

    # Combine all penalties
    penalties = {
        'Null Percentage': null_penalty,
        'Outlier Percentage': outlier_penalty,
        'Failures': failure_penalty,
        'Memory Spikes': memory_spike_penalty,
        'Time Spikes': time_spike_penalty,
        'Prometheus Metrics': prometheus_penalty,
        'Prometheus Breakdown': prometheus_penalty_breakdown,
    }

    return trustworthiness_index, penalties


def create_pdf_report(stats, null_percentage, outlier_percentages, output_metrics_json_list,
                      output_evaluation_json_list, prometheus_metrics):
    # Initialize PDF with landscape orientation (A4)
    pdf = FPDF('L', 'mm', 'A4')  # 'L' for landscape, 'mm' for millimeters, 'A4' is the paper size
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Add Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(270, 10, 'Trustworthiness Report', ln=True, align='C')

    # Calculate Trustworthiness Index
    trustworthiness_index, penalties = calculate_trustworthiness_index(
        null_percentage, outlier_percentages, output_metrics_json_list, output_evaluation_json_list, prometheus_metrics
    )

    # Add Trustworthiness Index
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(270, 10, f'Trustworthiness Index: {trustworthiness_index:.2f}%', ln=True, align='C')

    # Add Penalties Breakdown
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(270, 10, 'Penalties Breakdown:', ln=True)
    pdf.set_font('Arial', '', 10)
    for penalty, value in penalties.items():
        if isinstance(value, (int, float)):
            pdf.cell(270, 10, f'{penalty}: {value:.2f}%', ln=True)

    # Add Explanations for Penalties
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(270, 10, 'Explanation of Penalties:', ln=True)
    pdf.set_font('Arial', '', 10)

    # Explanation for Null Percentage
    pdf.multi_cell(270, 6,
                   '1. **Null Percentage**: This penalty is applied based on the percentage of missing or null values in the dataset. '
                   'A higher percentage of null values reduces the reliability of the data, as it may lead to incomplete or biased analysis.',
                   align='L')

    # Explanation for Outlier Percentage
    pdf.ln(5)
    pdf.multi_cell(270, 6,
                   '2. **Outlier Percentage**: This penalty is calculated as the average percentage of outliers detected using three methods: '
                   'Interquartile Range (IQR), Confidence Interval (CI), and Z-Score (ZS). Outliers can indicate data errors or anomalies, '
                   'which may affect the accuracy of the analysis.',
                   align='L')

    # Explanation for Failures
    pdf.ln(5)
    pdf.multi_cell(270, 6,
                   '3. **Failures**: This penalty is based on the number of failures reported during the execution of the process. '
                   'Failures indicate issues in the workflow, such as errors in data processing or system crashes, which reduce trustworthiness.',
                   align='L')

    # Explanation for Memory Spikes
    pdf.ln(5)
    pdf.multi_cell(270, 6,
                   '4. **Memory Spikes**: This penalty is applied when memory usage exceeds the average memory consumption. '
                   'Memory spikes can indicate inefficient resource usage or potential memory leaks, which may lead to system instability.',
                   align='L')

    # Explanation for Time Spikes
    pdf.ln(5)
    pdf.multi_cell(270, 6,
                   '5. **Time Spikes**: This penalty is applied when execution time exceeds the average execution time. '
                   'Time spikes can indicate performance bottlenecks or inefficiencies in the process, which may affect overall reliability.',
                   align='L')

    # Add Outlier Percentages
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(270, 10, 'Outlier Percentages:', ln=True)
    pdf.set_font('Arial', '', 10)
    pdf.cell(270, 10, f'IQR: {outlier_percentages["IQR"]:.2f}%', ln=True)
    pdf.cell(270, 10, f'CI: {outlier_percentages["CI"]:.2f}%', ln=True)
    pdf.cell(270, 10, f'ZS: {outlier_percentages["ZS"]:.2f}%', ln=True)

    # Add Detailed Statistics Table
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(270, 10, 'Detailed Statistics:', ln=True)

    # Define column widths for the table
    col_widths = [40, 30, 30, 30, 40, 40, 40]

    # Add Table Header
    pdf.set_font('Arial', 'B', 10)
    headers = ['Column', 'Average', 'Null Count', 'Null %', 'Outliers IQR', 'Outliers CI', 'Outliers ZS']
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 10, header, 1, 0, 'C')
    pdf.ln()

    # Add Table Rows
    pdf.set_font('Arial', '', 8)
    for column, data in stats.items():
        pdf.cell(col_widths[0], 10, column, 1, 0, 'C')
        pdf.cell(col_widths[1], 10, f'{data["Average"]:.2f}', 1, 0, 'C')
        pdf.cell(col_widths[2], 10, str(data["Null Count"]), 1, 0, 'C')
        pdf.cell(col_widths[3], 10, f'{data["Null Percentage"]:.2f}%', 1, 0, 'C')
        pdf.cell(col_widths[4], 10, str(len(data["Outliers IQR"])), 1, 0, 'C')
        pdf.cell(col_widths[5], 10, str(len(data["Outliers CI"])), 1, 0, 'C')
        pdf.cell(col_widths[6], 10, str(len(data["Outliers ZS"])), 1, 1, 'C')

    pdf.ln(10)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(270, 10, 'Output Metric Summary:', ln=True)

    output_col_widths = [25, 20, 45, 40, 25, 30, 25]

    pdf.set_font('Arial', 'B', 10)
    headers = ['Success', 'Failure', 'Warnings', 'Total Time (s)', 'File Read (s)', 'Processing (s)', 'File Write (s)']
    for i, header in enumerate(headers):
        pdf.cell(output_col_widths[i], 10, header, 1, 0, 'C')
    pdf.ln()

    pdf.set_font('Arial', '', 8)
    for json_data in output_metrics_json_list:
        total_time = json_data["processing_time"]["total"] / 1000.0
        file_read = json_data["processing_time"]["file_read"] / 1000.0
        processing = json_data["processing_time"]["processing"] / 1000.0
        file_write = json_data["processing_time"]["file_write"] / 1000.0

        pdf.cell(output_col_widths[0], 10, str(json_data['success']), 1, 0, 'C')
        pdf.cell(output_col_widths[1], 10, str(json_data['failure']), 1, 0, 'C')
        pdf.cell(output_col_widths[2], 10, str(json_data['warnings']), 1, 0, 'C')
        pdf.cell(output_col_widths[3], 10, f'{total_time:.5f}', 1, 0, 'C')
        pdf.cell(output_col_widths[4], 10, f'{file_read:.5f}', 1, 0, 'C')
        pdf.cell(output_col_widths[5], 10, f'{processing:.5f}', 1, 0, 'C')
        pdf.cell(output_col_widths[6], 10, f'{file_write:.5f}', 1, 1, 'C')
        
    # Add Prometheus Metrics Section
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(270, 10, 'Prometheus Metrics:', ln=True)

    # Define column widths for the table
    col_widths = [60, 45, 60]  # Adjust widths as needed

    # Add Table Header
    pdf.set_font('Arial', 'B', 10)
    headers = ['Metric', 'Value', 'Description']
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 10, header, 1, 0, 'C')
    pdf.ln()

    # Add Table Rows
    pdf.set_font('Arial', '', 8)
    metric_descriptions = {
        'http_request_duration_seconds_count': 'Total number of HTTP requests.',
        'http_request_duration_seconds_sum': 'Total time spent on HTTP requests.',
        'process_cpu_seconds_total': 'Total CPU usage in seconds.',
        'process_resident_memory_bytes': 'Memory usage in bytes.',
        'minio_bucket_usage_total_bytes': 'Total bucket usage in MinIO.',
        'postgres_exporter_errors_total': 'Total errors reported by PostgreSQL exporter.',
    }

    for metric, value in prometheus_metrics.items():
        description = metric_descriptions.get(metric, 'No description available.')
        pdf.cell(col_widths[0], 10, metric, 1, 0, 'L')
        pdf.cell(col_widths[1], 10, str(value), 1, 0, 'C')
        pdf.cell(col_widths[2], 10, description, 1, 1, 'L')

    pdf.ln(10)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(270, 10, 'Execution Metric Summary:', ln=True)

    exec_col_widths = [45, 25, 25, 30, 30, 30, 30, 30]

    pdf.set_font('Arial', 'B', 10)
    headers = ['StartTime', 'Success', 'Errors', 'Exec Time (s)', 'CPU Usage', 'Memory Usage', 'Peak Memory', 'Threads']
    for i, header in enumerate(headers):
        pdf.cell(exec_col_widths[i], 10, header, 1, 0, 'C')
    pdf.ln()

    pdf.set_font('Arial', '', 8)
    for json_data in output_evaluation_json_list:
        start_time = json_data["StartTime"]
        success = 1 if json_data["Success"] else 0
        errors_count = len(json_data["Errors"])
        total_time = json_data["ExecutionTimeMs"] / 1000.0

        cpu_usage = json_data["ResourceUsage"]["CpuUsageMs"] / 1000.0
        memory_usage = json_data["ResourceUsage"]["MemoryUsageBytes"] / (1024 * 1024)
        peak_memory_usage = json_data["ResourceUsage"]["PeakMemoryUsageBytes"] / (1024 * 1024)
        threads_count = json_data["ResourceUsage"]["ThreadsCount"]

        pdf.cell(exec_col_widths[0], 10, start_time, 1, 0, 'C')
        pdf.cell(exec_col_widths[1], 10, str(success), 1, 0, 'C')
        pdf.cell(exec_col_widths[2], 10, str(errors_count), 1, 0, 'C')
        pdf.cell(exec_col_widths[3], 10, f'{total_time:.5f}', 1, 0, 'C')
        pdf.cell(exec_col_widths[4], 10, f'{cpu_usage:.5f}', 1, 0, 'C')
        pdf.cell(exec_col_widths[5], 10, f'{memory_usage:.4f} MB', 1, 0, 'C')
        pdf.cell(exec_col_widths[6], 10, f'{peak_memory_usage:.4f} MB', 1, 0, 'C')
        pdf.cell(exec_col_widths[7], 10, str(threads_count), 1, 1, 'C')

    pdf_output = pdf.output(dest='S').encode('latin1')
    pdf_output_stream = BytesIO(pdf_output)

    return pdf_output_stream


def calculate_trust_stats(req):
    """
    Calculate trust bounds and confidence percentage from the input DataFrames.
    """
    try:
        exec_info = json.loads(req.json['ExecInfo'])
        # Extract configuration from parsed JSON input
        access_key = exec_info['config']['access_key']['value']
        secret_key = exec_info['config']['secret_key']['value']
        bucket_name = exec_info['config']['bucket_name']['value']

        # Create a MinIO client
        client = Minio(
            "minio:24001",
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )

        # Save trust metrics to MinIO (JSON)
        trust_metrics_output_path = exec_info['outputs']['Dataframe']['destination'].split('-')[0] + '/' + \
                                    exec_info['outputs']['Dataframe']['destination'].split('-')[
                                        1] + '-trust_evaluation.json'
        with BytesIO() as trust_buffer:
            trust_buffer.write(json.dumps(req.json).encode())
            trust_buffer.seek(0)
            client.put_object(bucket_name, trust_metrics_output_path, trust_buffer, len(trust_buffer.getvalue()))

        if 'Dataframe' in exec_info['outputs']:
            input_stats_file_name = exec_info['outputs']['Dataframe']['destination']
            obj_stats = client.get_object(bucket_name, input_stats_file_name)
            pickled_stats_data = obj_stats.read()
            df = pickle.loads(pickled_stats_data)
        else:
            input_stats_file_name = exec_info['outputs']['FileName']['destination']
            obj_stats = client.get_object(bucket_name, input_stats_file_name)
            csv_data = obj_stats.read()
            df = pd.read_csv(BytesIO(csv_data))

        # Initialize empty DataFrame to collect all data
        output_metrics_json_list = []
        output_evaluation_json_list = []

        # Check if there are multiple files in the directory to process
        folder_name = exec_info['outputs']['Dataframe']['destination'].split('-')[0]
        files = client.list_objects(bucket_name, prefix=folder_name, recursive=True)

        for file in files:
            if file.object_name.endswith('trust_metrics.json'):
                obj = client.get_object(bucket_name, file.object_name)
                file_data = obj.read()
                json_data = json.loads(file_data.decode('utf-8'))
                output_metrics_json_list.append(json_data)
            if file.object_name.endswith('trust_evaluation.json'):
                obj = client.get_object(bucket_name, file.object_name)
                file_data = obj.read()
                json_data = json.loads(file_data.decode('utf-8'))
                output_evaluation_json_list.append(json_data)

        # Calculate statistics (outliers, averages, null counts)
        stats, total_values, outliers_counts = calculate_statistics(df)

        total_cells = df.size
        null_cells = df.isnull().sum().sum()
        null_percentage = (null_cells / total_cells) * 100
        outlier_percentages = {
            method: (count / total_values) * 100 if total_values > 0 else 0
            for method, count in outliers_counts.items()
        }
        
        # Fetch Prometheus metrics
        metric_names = [
            'http_request_duration_seconds_count',
            'http_request_duration_seconds_sum',
            'process_cpu_seconds_total',
            'process_resident_memory_bytes',
            'minio_bucket_usage_total_bytes',
            'postgres_exporter_errors_total',
        ]
        prometheus_metrics = fetch_prometheus_metrics(metric_names)

        # Generate the PDF report and get the PDF content in memory
        pdf_output_stream = create_pdf_report(stats, null_percentage, outlier_percentages, output_metrics_json_list,
                                              output_evaluation_json_list, prometheus_metrics)

        # Upload the PDF to MinIO
        pdf_file_path = folder_name + '/Trustworthiness_Report.pdf'
        upload_pdf_to_minio(client, bucket_name, pdf_output_stream, pdf_file_path)

        return f"Report Uploaded"

    except KeyError as exc:
        raise RuntimeError(f"Missing key in JSON input: {exc}")
    except S3Error as exc:
        raise RuntimeError(f"An error occurred while uploading to MinIO: {exc}")
    except Exception as exc:
        raise RuntimeError(f"Error processing file: {exc}")


def fetch_prometheus_metrics(metric_names):
    """ Fetch multiple metrics from Prometheus """
    metrics = {}
    for metric_name in metric_names:
        response = requests.get("http://prometheus:9090/api/v1/query", params={"query": metric_name})
        result = response.json()
        if 'data' in result and 'result' in result['data'] and len(result['data']['result']) > 0:
            metrics[metric_name] = float(result['data']['result'][0]['value'][1])  # Extract metric value
        else:
            metrics[metric_name] = None
    return metrics


def upload_pdf_to_minio(client, bucket_name, pdf_output_stream, output_path):
    client.put_object(bucket_name, output_path, pdf_output_stream, len(pdf_output_stream.getvalue()))

# Main function to handle incoming requests.
def main(context):
    """
    Main function to handle incoming requests.
    """
    if 'request' in context.keys():
        if context.request.method == "POST":
            output_file = calculate_trust_stats(context.request)
            print(output_file)
            return context.request.method, 200
        elif context.request.method == "GET":
            return context.request.method, 200
    else:
        print("Empty request", flush=True)
        return "{}", 200
