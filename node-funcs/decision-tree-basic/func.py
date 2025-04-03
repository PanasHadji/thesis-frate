import json
import pandas as pd
from minio import Minio
from minio.error import S3Error
from io import BytesIO
import pickle
import datetime
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
#from chart_manager.generate_charts import auto_generate_charts_and_report
#from lime.lime_tabular import LimeTabularExplainer


def decision_tree(req):
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
        access_key = req.json['config']['access_key']['value']
        secret_key = req.json['config']['secret_key']['value']
        bucket_name = req.json['config']['bucket_name']['value']

        log_event("Extracted configuration", f"Bucket: {bucket_name}")

        client = Minio(
            "minio:24001",
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )

        log_event("MinIO client initialized")

        input_file_name = req.json['inputs']['Dataframe']['value']
        output_file_name = req.json['outputs']['Dataframe']['destination']
        target_column = req.json['inputs']['TargetVariable']['value']
        test_size = float(req.json['inputs']['TestSize']['value']) if req.json['inputs']['TestSize']['value'] else 0.3

        log_event("Input parameters extracted",
                  f"Input File: {input_file_name}, Output File: {output_file_name}, Target Column: {target_column}, Test Size: {test_size}")

        # Decision Tree hyperparameters
        criterionEnum = req.json['inputs']['Criterion']['value']
        splitterEnum = req.json['inputs']['Splitter']['value']

        criterion = 'gini'
        if criterionEnum == 1:
            criterion = 'gini'
        elif criterionEnum == 2:
            criterion = 'entropy'

        splitter = 'best'
        if splitterEnum == 1:
            splitter = 'best'
        elif splitterEnum == 2:
            splitter = 'random'

        max_depth = int(req.json['inputs']['MaxDepth']['value']) if req.json['inputs']['MaxDepth']['value'] else None
        min_samples_split = int(req.json['inputs']['MinSamplesSplit']['value']) if \
        req.json['inputs']['MinSamplesSplit']['value'] else 2
        min_samples_leaf = int(req.json['inputs']['MinSamplesLeaf']['value']) if req.json['inputs']['MinSamplesLeaf'][
            'value'] else 1
        min_weight_fraction_leaf = float(req.json['inputs']['MinWeightFraction']['value']) if \
        req.json['inputs']['MinWeightFraction']['value'] else 0.0
        max_features = req.json['inputs']['MaxFeatures']['value'] if req.json['inputs']['MaxFeatures'][
            'value'] else None
        max_leaf_nodes = int(req.json['inputs']['MaxLeafNodes']['value']) if req.json['inputs']['MaxLeafNodes'][
            'value'] else None

        log_event("Hyperparameters extracted",
                  f"Criterion: {criterion}, Splitter: {splitter}, Max Depth: {max_depth}, Min Samples Split: {min_samples_split}, Min Samples Leaf: {min_samples_leaf}, Min Weight Fraction Leaf: {min_weight_fraction_leaf}, Max Features: {max_features}, Max Leaf Nodes: {max_leaf_nodes}")

        file_read_start_time = time.time()

        # Read the pickled DataFrame from MinIO
        obj = client.get_object(bucket_name, input_file_name)
        pickled_data = obj.read()

        trust_metrics["processing_time"]["file_read"] = abs(time.time() - file_read_start_time)
        log_event("File read from MinIO", f"Size: {len(pickled_data)} bytes")

        df = pickle.loads(pickled_data)

        log_event("Cleaning Dataset", "Cleaning and preprocessing dataset.")
        df = df.drop(index=0).reset_index(drop=True)
        df[target_column] = df[target_column].apply(lambda x: ''.join(e for e in str(x) if e.isprintable())).str.strip()
        df[target_column] = df[target_column].fillna('Unknown')
        df_clean = df.dropna(subset=[target_column]).dropna(axis=0, how='any')
        log_event("Dataset cleaned", f"Rows after cleaning: {len(df_clean)}")

        log_event("Preparing Features", "Preparing feature variables for training.")
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]

        log_event("Splitting Dataset", "Splitting data into training and test sets.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        log_event("Dataset split", f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

        log_event("Fitting Decision Tree Model", "Training Decision Tree model.")
        dt_model = DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes
        )
        dt_model.fit(X_train, y_train)

        log_event("Model trained", "Decision Tree model training completed.")

        # Predictions
        log_event("Making Predictions", "Making predictions on the test set.")
        y_pred = dt_model.predict(X_test)

        log_event("Predictions made", f"Predictions completed. Number of predictions: {len(y_pred)}")

        # LIME explanation
#         lime_explainer = LimeTabularExplainer(
#             training_data=X_train.values,
#             feature_names=X_train.columns,
#             class_names=y_train.unique().tolist(),
#             discretize_continuous=True
#         )
#         lime_explanation = lime_explainer.explain_instance(X_test.iloc[0].values, dt_model.predict_proba)
#         lime_explanation_str = lime_explanation.as_list()
# 
#         # Log LIME  explanations
#         log_event("LIME Explanation", f"Explanation: {lime_explanation_str}")
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
#         # Log LIME explanation and plot upload
#         log_event("LIME Plot", f"Uploaded Explanation Plot")

        # Results DataFrame
        results_df = pd.DataFrame({
            "Index": range(len(y_test)),
            "Actual": y_test.tolist(),
            "Predictions": y_pred.tolist(),
        })

        print('Uploading dataframe')
        with BytesIO() as bytes_buffer:
            pickle.dump(results_df, bytes_buffer)
            bytes_buffer.seek(0)
            client.put_object(bucket_name, output_file_name, bytes_buffer, len(bytes_buffer.getvalue()))

        log_event("Output file saved", f"Path: {output_file_name}")

        trust_metrics["processing_time"]["file_write"] = abs(
            time.time() - start_time - trust_metrics["processing_time"]["file_read"] - trust_metrics["processing_time"][
                "processing"])

#         print('Creating charts')
#         auto_generate_charts_and_report(
#             df=results_df,
#             y_pred=y_pred,
#             output_folder_name=output_file_name.split('/')[0] + "/visualization_reports",
#             client=client,
#             bucket_name=bucket_name
#         )

#         chart_generation_start_time = time.time()

#         trust_metrics["processing_time"]["chart_generation"] = max(0, time.time() - chart_generation_start_time)

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
        _save_logs_and_metrics(logs, trust_metrics, client, bucket_name, req.json['outputs']['Dataframe']['destination'].split('/')[0])
        raise RuntimeError(f"Missing key in JSON input: {exc}")

    except S3Error as exc:
        log_event("Error", f"MinIO error: {exc}")
        trust_metrics["failure"] += 1
        trust_metrics["error_frequency"][str(exc)] = trust_metrics["error_frequency"].get(str(exc), 0) + 1
        _save_logs_and_metrics(logs, trust_metrics, client, bucket_name, req.json['outputs']['Dataframe']['destination'].split('/')[0])
        raise RuntimeError(f"An error occurred while reading from or uploading to MinIO: {exc}")

    except Exception as exc:
        log_event("Error", f"General processing error: {exc}")
        trust_metrics["failure"] += 1
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


# Main function to handle incoming requests.
def main(context):
    """
    Entry point for handling incoming HTTP requests for the decision_tree function.
    """
    if 'request' in context.keys():
        if context.request.method == "POST":
            output_file = decision_tree(context.request)
            print(output_file)
            return context.request.method, 200
        elif context.request.method == "GET":
            return context.request.method, 200
    else:
        print("Empty request", flush=True)
        return "{}", 200
