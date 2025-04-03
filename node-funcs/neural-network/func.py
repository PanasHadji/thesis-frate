import json
import pandas as pd
from minio import Minio
from minio.error import S3Error
from io import BytesIO
import pickle
import datetime
import time
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
#from chart_manager.generate_charts import auto_generate_charts_and_report
#from lime.lime_tabular import LimeTabularExplainer


def neural_network(req):
    """
    Use a Neural Network (MLPClassifier) to classify a dataset read from MinIO and upload the results back to MinIO.
    """
    logs = []  # List to store log entries
    trust_metrics = {
        "success": 0,
        "failure": 0,
        "warnings": 0,
        "processing_time": {"total": 0, "file_read": 0, "processing": 0, "file_write": 0, "chart_generation": 0},
        "data_integrity": {"input_rows": 0, "output_rows": 0},
        "error_frequency": {}
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

        # Extract input and output file names directly
        input_file_name = req.json['inputs']['Dataframe']['value']
        output_file_name = req.json['outputs']['Dataframe']['destination']
        target_column = req.json['inputs']['TargetColumn']['value']

        log_event("Input parameters extracted",
                  f"Input File: {input_file_name}, Output File: {output_file_name}, Target Column: {target_column}")

        file_read_start_time = time.time()

        # Read the pickled DataFrame from MinIO
        obj = client.get_object(bucket_name, input_file_name)
        pickled_data = obj.read()

        trust_metrics["processing_time"]["file_read"] = abs(time.time() - file_read_start_time)
        log_event("File read from MinIO", f"Size: {len(pickled_data)} bytes")

        # Unpickle the DataFrame
        df = pickle.loads(pickled_data)

        log_event("Cleaning Dataset", "Cleaning and preprocessing dataset.")
        # Remove the first row (headers if mistakenly included as data)
        df = df.drop(index=0).reset_index(drop=True)

        # Clean the target column
        df[target_column] = df[target_column].apply(lambda x: ''.join(e for e in str(x) if e.isprintable())).str.strip()
        df[target_column] = df[target_column].fillna('Unknown')  # Replace NaNs with 'Unknown'

        # Drop rows with any NaN values
        df_clean = df.dropna(subset=[target_column]).dropna(axis=0, how='any')
        log_event("Dataset cleaned", f"Rows after cleaning: {len(df_clean)}")

        log_event("Preparing Features", "Preparing feature variables for training.")
        # Prepare features and target
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        log_event("Fitting Neural Network Model (MLPClassifier)", "Training Neural Network Model (MLPClassifier).")
        # Fit the Neural Network model (MLPClassifier)
        mlp_model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, solver='lbfgs', early_stopping=True, n_iter_no_change=10)
        mlp_model.fit(X_scaled, y)

        # Predictions
        log_event("Model trained", "Decision Tree model training completed.")

        # Predictions
        log_event("Making Predictions", "Making predictions on the test set.")
        y_pred = mlp_model.predict(X_scaled)

        log_event("Predictions made", f"Predictions completed. Number of predictions: {len(y_pred)}")

        # LIME explanation
#         lime_explainer = LimeTabularExplainer(
#             training_data=X.values,
#             feature_names=X.columns,
#             class_names=y.unique().tolist(),
#             discretize_continuous=True
#         )
# 
#         # Select an instance to explain
#         lime_explanation = lime_explainer.explain_instance(X.iloc[0].values, mlp_model.predict_proba)
#         lime_explanation_str = lime_explanation.as_list()
# 
#         log_event("LIME Explanation", f"Explanation: {lime_explanation_str}")
# 
#         # Generate the LIME plot
#         fig = lime_explanation.as_pyplot_figure()
# 
#         # Save the plot to a BytesIO object
#         plot_buffer = BytesIO()
#         fig.savefig(plot_buffer, format='png')
#         plot_buffer.seek(0)
# 
#         # Upload the plot to MinIO
#         plot_filename = f"{output_file_name.split('/')[0]}/lime_plot.png"
#         client.put_object(bucket_name, plot_filename, plot_buffer, len(plot_buffer.getvalue()))
# 
#         log_event("LIME Plot", f"Uploaded Explanation Plot to MinIO as {plot_filename}")

        # Create a DataFrame for the predictions
        results_df = pd.DataFrame(
            {
                "Index": range(len(df)),
                "Predictions": y_pred.tolist(),
            }
        )

        # Optionally use indices or distances as `y_pred` (modify as needed based on your use case)
        y_pred = y_pred.flatten()

        print('Uploading dataframe')
        # Pickle the results DataFrame and upload it back to MinIO
        with BytesIO() as bytes_buffer:
            pickle.dump(results_df, bytes_buffer)
            bytes_buffer.seek(0)
            client.put_object(bucket_name, output_file_name, bytes_buffer, len(bytes_buffer.getvalue()))

        log_event("Output file saved", f"Path: {output_file_name}")

        trust_metrics["processing_time"]["file_write"] = abs(
            time.time() - start_time - trust_metrics["processing_time"]["file_read"] - trust_metrics["processing_time"][
                "processing"])

        print('Creating charts')
        # Generate charts and reports
#         auto_generate_charts_and_report(
#             df=results_df,
#             y_pred=y_pred,
#             output_folder_name=output_file_name.split('/')[0] + "/visualization_reports",
#             client=client,
#             bucket_name=bucket_name
#         )
#         chart_generation_start_time = time.time()
# 
#         trust_metrics["processing_time"]["chart_generation"] = max(0, time.time() - chart_generation_start_time)
# 
#         log_event("Charts generated for DataFrame")

        trust_metrics["processing_time"]["total"] = time.time()

        trust_metrics["success"] += 1
        log_event("Processing completed successfully", f"Output file: {output_file_name}")

        # Save logs and trust metrics
        log_output_path = req.json['outputs']['Dataframe']['destination'].split('/')[0] + '/logs.csv'
        log_df = pd.DataFrame(logs)

        with BytesIO() as log_buffer:
            log_df.to_csv(log_buffer, index=False)
            log_buffer.seek(0)
            client.put_object(bucket_name, log_output_path, log_buffer, len(log_buffer.getvalue()))

        log_event("Logs saved to MinIO", f"Path: {log_output_path}")

        # Save trust metrics
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
        trust_metrics["failure"] += 1
        trust_metrics["error_frequency"][str(exc)] = trust_metrics["error_frequency"].get(str(exc), 0) + 1
        _save_logs_and_metrics(logs, trust_metrics, client, bucket_name,
                               req.json['outputs']['Dataframe']['destination'].split('/')[0])
        raise RuntimeError(f"Missing key in JSON input: {exc}")

    except S3Error as exc:
        log_event("Error", f"MinIO error: {exc}")
        trust_metrics["failure"] += 1
        trust_metrics["error_frequency"][str(exc)] = trust_metrics["error_frequency"].get(str(exc), 0) + 1
        _save_logs_and_metrics(logs, trust_metrics, client, bucket_name,
                               req.json['outputs']['Dataframe']['destination'].split('/')[0])
        raise RuntimeError(f"An error occurred while reading from or uploading to MinIO: {exc}")

    except Exception as exc:
        log_event("Error", f"General processing error: {exc}")
        trust_metrics["failure"] += 1
        trust_metrics["error_frequency"][str(exc)] = trust_metrics["error_frequency"].get(str(exc), 0) + 1
        _save_logs_and_metrics(logs, trust_metrics, client, bucket_name,
                               req.json['outputs']['Dataframe']['destination'].split('/')[0])
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
    Entry point for handling incoming HTTP requests for the neural_network function.
    """
    if 'request' in context.keys():
        if context.request.method == "POST":
            output_file = neural_network(context.request)
            print(output_file)
            return context.request.method, 200
        elif context.request.method == "GET":
            return context.request.method, 200
    else:
        print("Empty request", flush=True)
        return "{}", 200
