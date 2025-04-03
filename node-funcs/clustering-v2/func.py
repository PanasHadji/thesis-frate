import pandas as pd
from minio import Minio
from minio.error import S3Error
from io import BytesIO
import pickle
import datetime
import time
import json
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, Birch, SpectralClustering,
    AffinityPropagation, OPTICS
)
import hdbscan
#from chart_manager.generate_charts import create_files
#from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LogisticRegression


def run_clustering(req):
    """Run a clustering algorithm on a DataFrame read from MinIO and upload the result back to MinIO."""
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

        client = Minio("minio:24001", access_key=access_key, secret_key=secret_key, secure=False)
        log_event("MinIO client initialized")

        input_file_name = req.json['inputs']['Dataframe']['value']
        output_file_name = req.json['outputs']['Dataframe']['destination']
        algorithm = req.json['inputs']['Algorithm']['value']

        log_event("Input parameters extracted",
                  f"Input File: {input_file_name}, Output File: {output_file_name}, Algorithm: {algorithm}")

        file_read_start_time = time.time()

        # Load the dataset
        obj = client.get_object(bucket_name, input_file_name)
        df = pickle.loads(obj.read())

        trust_metrics["processing_time"]["file_read"] = abs(time.time() - file_read_start_time)
        log_event("File read from MinIO", f"Size: {len(df)} bytes")

        df_numeric = df.select_dtypes(include=[float, int]).dropna()
#         scaler = StandardScaler()
#         df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric))

        # Initialize clustering model
        if algorithm == "KMeans":
            log_event("Using Algorithm", f"Algorithm: {algorithm}")
            n_clusters = req.json['inputs'].get('N_Clusters', {}).get('value', '') or 3
            random_state = req.json['inputs'].get('Random_State', {}).get('value', '') or 42
            model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')

        elif algorithm == "DBSCAN":
            log_event("Using Algorithm", f"Algorithm: {algorithm}")
            eps = req.json['inputs'].get('Eps', {}).get('value', '') or 0.5
            min_samples = req.json['inputs'].get('Min_Samples', {}).get('value', '') or 5
            model = DBSCAN(eps=eps, min_samples=min_samples)

        elif algorithm == "GaussianMixture":
            log_event("Using Algorithm", f"Algorithm: {algorithm}")
            n_clusters = req.json['inputs'].get('N_Clusters', {}).get('value', '') or 3
            random_state = req.json['inputs'].get('Random_State', {}).get('value', '') or 42
            model = GaussianMixture(n_components=n_clusters, random_state=random_state)

        elif algorithm == "AffinityPropagation":
            log_event("Using Algorithm", f"Algorithm: {algorithm}")
            damping = req.json['inputs'].get('Damping', {}).get('value', '') or 0.9
            model = AffinityPropagation(damping=damping)

        elif algorithm == "Ward":
            log_event("Using Algorithm", f"Algorithm: {algorithm}")
            n_clusters = req.json['inputs'].get('N_Clusters', {}).get('value', '') or 3
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")

        elif algorithm == "AGNES":
            log_event("Using Algorithm", f"Algorithm: {algorithm}")
            n_clusters = req.json['inputs'].get('N_Clusters', {}).get('value', '') or 3
            linkage = req.json['inputs'].get('Linkage', {}).get('value', '') or "average"
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)

        elif algorithm == "HDBSCAN":
            log_event("Using Algorithm", f"Algorithm: {algorithm}")
            min_cluster_size = req.json['inputs'].get('Min_Cluster_Size', {}).get('value', '') or 5
            model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)

        elif algorithm == "OPTICS":
            log_event("Using Algorithm", f"Algorithm: {algorithm}")
            min_samples = req.json['inputs'].get('Min_Samples', {}).get('value', '') or 5
            max_eps = req.json['inputs'].get('Max_Eps', {}).get('value', '') or float('inf')
            model = OPTICS(min_samples=min_samples, max_eps=max_eps)

        elif algorithm == "BIRCH":
            log_event("Using Algorithm", f"Algorithm: {algorithm}")
            n_clusters = req.json['inputs'].get('N_Clusters', {}).get('value', '') or 3
            threshold = req.json['inputs'].get('Threshold', {}).get('value', '') or 0.5
            model = Birch(n_clusters=n_clusters, threshold=threshold)

        elif algorithm == "SpectralClustering":
            log_event("Using Algorithm", f"Algorithm: {algorithm}")
            n_clusters = req.json['inputs'].get('N_Clusters', {}).get('value', '') or 3
            affinity = req.json['inputs'].get('Affinity', {}).get('value', '') or "rbf"
            model = SpectralClustering(n_clusters=n_clusters, affinity=affinity)

        else:
            raise ValueError(f"Algorithm '{algorithm}' is not supported.")

        # Fit the model
        if algorithm in ["KMeans", "GaussianMixture", "SpectralClustering"]:
            y_pred = model.fit_predict(df_numeric)
        elif algorithm == "HDBSCAN":
            y_pred = model.fit(df_numeric).labels_
        else:
            y_pred = model.fit(df_numeric).labels_

        # Evaluate clustering using silhouette score
        silhouette_avg = silhouette_score(df_numeric, y_pred)

        # Prepare metrics
        model_metrics = {"silhouette_score": silhouette_avg}
        if algorithm == "KMeans":
            model_metrics.update({
                "inertia": model.inertia_,
                "cluster_centers": model.cluster_centers_.tolist(),
                "inertia_values": [KMeans(n_clusters=i, n_init='auto').fit(df_numeric).inertia_ for i in range(1, 11)]
            })

        # Save results
        output_folder_name = output_file_name.split('/')[0]
        with BytesIO() as bytes_buffer:
            pickle.dump(df, bytes_buffer)
            bytes_buffer.seek(0)
            client.put_object(bucket_name, output_file_name, bytes_buffer, len(bytes_buffer.getvalue()))

        log_event("Output file saved", f"Path: {output_file_name}")

        trust_metrics["processing_time"]["file_write"] = abs(
            time.time() - start_time - trust_metrics["processing_time"]["file_read"] -
            trust_metrics["processing_time"][
                "processing"])

        # LIME Explanation
#         log_event("Generating LIME Explanation", "Creating LIME explanation for clustering results.")
#         surrogate_model = LogisticRegression(max_iter=1000)
#         surrogate_model.fit(df_scaled, y_pred)
# 
#         lime_explainer = LimeTabularExplainer(
#             training_data=df_scaled.values,
#             feature_names=df_numeric.columns,
#             class_names=[f"Cluster {i}" for i in range(n_clusters)],
#             discretize_continuous=True
#         )
#         log_event("LIME Explainer Initialized",
#                   f"Features: {df_numeric.columns.tolist()}, Classes: {[f'Cluster {i}' for i in range(n_clusters)]}")
# 
#         # Explain an instance
#         instance_index = 0  # Explain the first instance
#         lime_explanation = lime_explainer.explain_instance(
#             df_scaled.iloc[instance_index].values,
#             surrogate_model.predict_proba,
#             num_features=len(df_numeric.columns)
#         )
# 
#         # Save LIME explanation plot
#         lime_plot = lime_explanation.as_pyplot_figure()
#         lime_plot.suptitle(f"LIME Explanation for Instance {instance_index}")
#         lime_plot_buffer = BytesIO()
#         lime_plot.savefig(lime_plot_buffer, format='png')
#         lime_plot_buffer.seek(0)
# 
#         # Upload LIME plot to MinIO
#         lime_plot_filename = f"{output_folder_name}/lime_explanation.png"
#         client.put_object(bucket_name, lime_plot_filename, lime_plot_buffer, len(lime_plot_buffer.getvalue()))
#         log_event("LIME Explanation Plot Uploaded", f"Path: {lime_plot_filename}")

        # Additional output for labels and metrics
        #create_files(output_folder_name, algorithm, y_pred, df_scaled, model_metrics, client, bucket_name)

        chart_generation_start_time = time.time()
        trust_metrics["processing_time"]["chart_generation"] = max(0, time.time() - chart_generation_start_time)
        log_event("Charts generated for DataFrame")
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


def main(context):
    """Main function to handle incoming requests."""
    if 'request' in context.keys():
        if context.request.method == "POST":
            run_clustering(context.request)
            return context.request.method, 200
        elif context.request.method == "GET":
            return context.request.method, 200
    else:
        print("Empty request", flush=True)
        return "{}", 200
