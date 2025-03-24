import pandas as pd
from minio import Minio
from minio.error import S3Error
from io import BytesIO
import pickle
from scipy.stats import skew, kurtosis


def stats_df(req):
    """
    Calculate null statistics for a DataFrame read from MinIO and save the statistics to a text file.
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
        output_file_path = req.json['outputs']['FileName']['destination']

        # Read data from CSV file in MinIO with Pandas
        obj = client.get_object(bucket_name, input_file_name)
        pickled_data = obj.read()

        # Unpickle the DataFrame
        df = pickle.loads(pickled_data)

        # Calculate null statistics
        null_counts = df.isnull().sum()
        null_percentages = (null_counts / len(df)) * 100
        null_stats = pd.DataFrame({
            'Column': null_counts.index,
            'Null Counts': null_counts.values,
            'Null Percentage': null_percentages.values
        })

        # Calculate descriptive statistics
        describe_stats = df.describe().T  # Transpose for easier readability

        # Add skewness and kurtosis
        skewness = df.apply(lambda x: skew(x.dropna()) if pd.api.types.is_numeric_dtype(x) else None)
        kurtosis_vals = df.apply(lambda x: kurtosis(x.dropna()) if pd.api.types.is_numeric_dtype(x) else None)
        describe_stats['Skewness'] = skewness
        describe_stats['Kurtosis'] = kurtosis_vals

        # Detect outliers using the IQR method
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        outliers_lower = (df < (Q1 - 1.5 * IQR)).sum()
        outliers_upper = (df > (Q3 + 1.5 * IQR)).sum()
        describe_stats['Outliers (Lower)'] = outliers_lower
        describe_stats['Outliers (Upper)'] = outliers_upper

        # Combine null statistics with describe statistics for a comprehensive report
        full_stats = describe_stats.reset_index().rename(columns={'index': 'Column'})
        full_stats = pd.merge(full_stats, null_stats, on='Column', how='outer').fillna('N/A')

        # Convert the statistics to CSV format
        csv_buffer = full_stats.to_csv(index=False)

        # Upload the CSV file to the MinIO bucket
        client.put_object(bucket_name, output_file_path + ".csv", BytesIO(csv_buffer.encode()), len(csv_buffer))

        return output_file_path
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
            output_file = stats_df(context.request)
            print(output_file)
            return context.request.method, 200
        elif context.request.method == "GET":
            return context.request.method, 200
    else:
        print("Empty request", flush=True)
        return "{}", 200
