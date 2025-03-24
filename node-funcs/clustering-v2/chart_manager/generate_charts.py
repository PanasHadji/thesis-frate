import os
import tempfile
import matplotlib
from io import BytesIO
import pickle
import matplotlib.pyplot as plt
from fpdf import FPDF
import seaborn as sns

matplotlib.use('Agg')

def create_text_file_in_memory(content_lines):
    """Create a text file in memory and return its bytes."""
    file_content = "\n".join(content_lines).encode('utf-8')
    return BytesIO(file_content)


def upload_file_to_minio(client, bucket_name, file_name, file_content):
    """Upload a file content to MinIO."""
    client.put_object(bucket_name, file_name, file_content, file_content.getbuffer().nbytes)


def create_and_upload_text_file(client, bucket_name, output_folder_name, file_name, content):
    """Create and upload a text file to MinIO."""
    upload_file_to_minio(client, bucket_name, f"{output_folder_name}/{file_name}",
                         create_text_file_in_memory(content))


def create_data_imbalance_report(df):
    """Create a report on data imbalance."""
    print("Create imbalance report")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    imbalance_lines = ["Data Imbalances\n"]
    for col in categorical_cols:
        class_proportions = df[col].value_counts(normalize=True)
        imbalance_lines.append(f"Class percentages for {col}:\n{class_proportions}\n")
        imbalance_exists = abs(class_proportions.max() - class_proportions.min()) > 0.05
        imbalance_lines.append(f"Difference: {abs(class_proportions.max() - class_proportions.min()) * 100:.2f}%\n")
        imbalance_lines.append(f"Class imbalance exists: {imbalance_exists}\n")
    return imbalance_lines


def create_performance_metrics_report(model_metrics, algorithm, y_pred):
    """Create a report of performance metrics."""
    print("Create performance metrics report")
    performance_lines = ["Model Performance Metrics\n"]
    for metric, value in model_metrics.items():
        performance_lines.append(f"{metric}: {value}\n")

    if algorithm == "KMeans":
        performance_lines.extend([
            f"Inertia: {model_metrics.get('inertia', None)}\n",
            f"Cluster Centers:\n{model_metrics.get('cluster_centers', None)}\n"
        ])
    elif algorithm in ["DBSCAN", "GaussianMixture"]:
        performance_lines.append(f"Number of clusters: {len(set(y_pred))}\n")
    return performance_lines


def create_clusters_visualization(df, y_pred, algorithm, tmpdirname):
    """Create a clusters visualization plot."""
    print("Create clusters visualization")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=y_pred, palette='viridis', s=5)
    plt.title(f'Clusters Visualization for {algorithm}')
    plot_file_path = os.path.join(tmpdirname, 'clusters_visualization.png')
    plt.savefig(plot_file_path)
    plt.close()
    return plot_file_path


def create_elbow_method_graph(model_metrics, tmpdirname):
    """Create the KMeans Elbow Method graph."""
    print("Create elbow method graph")
    inertia_values = model_metrics.get('inertia_values', [])
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(inertia_values) + 1), inertia_values, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plot_file_path = os.path.join(tmpdirname, 'elbow_method.png')
    plt.savefig(plot_file_path)
    plt.close()
    return plot_file_path


def create_scatter_plot(df, y_pred, tmpdirname):
    """Create a scatter plot and save it to a temporary directory."""
    print("Create scatter plot")
    plt.figure(figsize=(10, 6))
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=y_pred, cmap='viridis', s=5)
    plt.title('Scatter Plot')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plot_file_path = os.path.join(tmpdirname, 'scatter_plot.png')
    plt.savefig(plot_file_path)
    plt.close()
    return plot_file_path


def create_histogram(df, tmpdirname):
    """Create a histogram and save it to a temporary directory."""
    print("Create histogram")
    plt.figure(figsize=(10, 6))
    df.hist(bins=30)
    plt.title('Histogram')
    plot_file_path = os.path.join(tmpdirname, 'histogram.png')
    plt.savefig(plot_file_path)
    plt.close()
    return plot_file_path


def create_box_plot(df, tmpdirname):
    """Create a box plot and save it to a temporary directory."""
    print("Create box plot")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df)
    plt.title('Box Plot')
    plot_file_path = os.path.join(tmpdirname, 'box_plot.png')
    plt.savefig(plot_file_path)
    plt.close()
    return plot_file_path


def create_violin_plot(df, tmpdirname):
    """Create a violin plot and save it to a temporary directory."""
    print("Create violin plot")
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df)
    plt.title('Violin Plot')
    plot_file_path = os.path.join(tmpdirname, 'violin_plot.png')
    plt.savefig(plot_file_path)
    plt.close()
    return plot_file_path


def create_files(output_folder_name, algorithm, y_pred, df, model_metrics, client, bucket_name):
    """Create text files and plots based on clustering results and upload them to MinIO."""

    file_names = {
        "label_encoding_file": 'label_encodings.txt',
        "data_imbalance_file": 'data_imbalance.txt',
        "performance_metrics_file": 'performance_metrics.txt',
        "clusters_visualization": 'clusters_visualization.png',
        "pdf_file_name": 'Trustworthiness_Report.pdf',
        "bar_chart": 'bar_chart.png',
        "line_plot": 'line_plot.png',
        "scatter_plot": 'scatter_plot.png',
        "histogram": 'histogram.png',
        "box_plot": 'box_plot.png',
        "pie_chart": 'pie_chart.png',
        "heatmap": 'heatmap.png',
        "violin_plot": 'violin_plot.png',
        "strip_plot": 'strip_plot.png',
        "swarm_plot": 'swarm_plot.png',
    }

    if algorithm == "KMeans":
        file_names["elbow_method"] = 'elbow_method.png'

    print('Creating Files in Memory')

    # Create and save text files in the temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create label encodings file
        label_encoding_content = [
            "Label Encodings to handle categorical variables\n",
            "Categorical variables encoded as needed.\n"
        ]
        label_encoding_path = os.path.join(tmpdirname, file_names["label_encoding_file"])
        with open(label_encoding_path, 'w') as f:
            f.writelines(label_encoding_content)
        print(f"Created: {label_encoding_path}")

        # Data imbalance report
        imbalance_lines = create_data_imbalance_report(df)
        imbalance_path = os.path.join(tmpdirname, file_names["data_imbalance_file"])
        with open(imbalance_path, 'w') as f:
            f.writelines(imbalance_lines)
        print(f"Created: {imbalance_path}")

        # Performance metrics
        performance_lines = create_performance_metrics_report(model_metrics, algorithm, y_pred)
        performance_path = os.path.join(tmpdirname, file_names["performance_metrics_file"])
        with open(performance_path, 'w') as f:
            f.writelines(performance_lines)
        print(f"Created: {performance_path}")

        # Visualizations
        print('Start Plots')
        plot_files = []

        # Create all plots and save to temporary files
        plot_files.append(create_clusters_visualization(df, y_pred, algorithm, tmpdirname))
        plot_files.append(create_scatter_plot(df, y_pred, tmpdirname))
        plot_files.append(create_histogram(df, tmpdirname))
        plot_files.append(create_box_plot(df, tmpdirname))
        plot_files.append(create_violin_plot(df, tmpdirname))

        # KMeans Elbow Method graph
        if algorithm == "KMeans":
            plot_files.append(create_elbow_method_graph(model_metrics, tmpdirname))

        # Upload all plots and text files to MinIO
        for plot_file in plot_files:
            upload_file_to_minio(client, bucket_name, f"{output_folder_name}/{os.path.basename(plot_file)}",
                                 BytesIO(open(plot_file, 'rb').read()))
            print(f"Uploaded plot: {plot_file}")

        # Upload text files
        for file_key in ["label_encoding_file", "data_imbalance_file", "performance_metrics_file"]:
            file_path = os.path.join(tmpdirname, file_names[file_key])
            with open(file_path, 'rb') as f:
                upload_file_to_minio(client, bucket_name, f"{output_folder_name}/{file_names[file_key]}",
                                     BytesIO(f.read()))
            print(f"Uploaded text file: {file_names[file_key]}")

        # Creating PDF report
        print('Creating PDF report...')
        pdf_file_path = create_pdf_report(file_names, tmpdirname)

        # Upload PDF to MinIO
        with open(pdf_file_path, 'rb') as f:
            upload_file_to_minio(client, bucket_name, f"{output_folder_name}/{file_names['pdf_file_name']}",
                                 BytesIO(f.read()))
        print(f"Uploaded PDF report: {file_names['pdf_file_name']}")


def create_pdf_report(file_names, tmpdirname):
    """Create a PDF report from text files and saved images."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Trustworthiness Report", ln=True, align='C')

    # Add label encodings
    pdf.cell(200, 10, txt="Label Encodings", ln=True)
    label_encodings_path = os.path.join(tmpdirname, file_names['label_encoding_file'])
    print(f"Reading label encodings from: {label_encodings_path}")
    with open(label_encodings_path, 'r') as f:
        pdf.multi_cell(0, 10, txt=f.read())

    # Add data imbalance report
    pdf.cell(200, 10, txt="Data Imbalance Report", ln=True)
    data_imbalance_path = os.path.join(tmpdirname, file_names['data_imbalance_file'])
    print(f"Reading data imbalance from: {data_imbalance_path}")
    with open(data_imbalance_path, 'r') as f:
        pdf.multi_cell(0, 10, txt=f.read())

    # Add performance metrics report
    pdf.cell(200, 10, txt="Performance Metrics", ln=True)
    performance_metrics_path = os.path.join(tmpdirname, file_names['performance_metrics_file'])
    print(f"Reading performance metrics from: {performance_metrics_path}")
    with open(performance_metrics_path, 'r') as f:
        pdf.multi_cell(0, 10, txt=f.read())

    # Add plots to PDF
    for plot_file in os.listdir(tmpdirname):
        if plot_file.endswith('.png'):  # Ensure we only add image files
            pdf.add_page()
            pdf.image(os.path.join(tmpdirname, plot_file), x=10, w=190)
            print(f"Added plot to PDF: {plot_file}")

    # Save the PDF to a file
    pdf_file_path = os.path.join(tmpdirname, 'Trustworthiness_Report.pdf')
    pdf.output(pdf_file_path)
    return pdf_file_path