import os
import tempfile
import itertools
import matplotlib
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

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


def create_data_summary_report(df):
    """Create a summary report of the DataFrame."""
    print("Create data summary report")
    summary_lines = ["Data Summary\n"]
    summary_lines.append(f"Data Info:\n{df.info()}\n")
    summary_lines.append(f"Data Description:\n{df.describe()}\n")
    return summary_lines


def create_column_chart(df, tmpdirname, column):
    """Create a column chart for a single numeric column."""
    print(f"Create column chart for {column}")
    plt.figure(figsize=(10, 6))
    df[column].plot(kind='bar')
    plt.title(f'Column Chart - {column}')
    plot_file_path = os.path.join(tmpdirname, f'{column}_column_chart.png')
    plt.savefig(plot_file_path)
    plt.close()
    return plot_file_path


def create_bar_chart(df, tmpdirname, column):
    """Create a bar chart for a categorical column vs numeric."""
    print(f"Create bar chart for {column}")
    plt.figure(figsize=(10, 6))
    df.groupby(column).size().plot(kind='bar')
    plt.title(f'Bar Chart - {column}')
    plot_file_path = os.path.join(tmpdirname, f'{column}_bar_chart.png')
    plt.savefig(plot_file_path)
    plt.close()
    return plot_file_path


def create_line_chart(df, tmpdirname, column):
    """Create a line chart for a numeric column."""
    print(f"Create line chart for {column}")
    plt.figure(figsize=(10, 6))
    df[column].plot(kind='line')
    plt.title(f'Line Chart - {column}')
    plot_file_path = os.path.join(tmpdirname, f'{column}_line_chart.png')
    plt.savefig(plot_file_path)
    plt.close()
    return plot_file_path


def create_area_chart(df, tmpdirname, column):
    """Create an area chart for a numeric column."""
    print(f"Create area chart for {column}")
    plt.figure(figsize=(10, 6))
    df[column].plot(kind='area')
    plt.title(f'Area Chart - {column}')
    plot_file_path = os.path.join(tmpdirname, f'{column}_area_chart.png')
    plt.savefig(plot_file_path)
    plt.close()
    return plot_file_path


def create_pie_chart(df, tmpdirname, column):
    """Create a pie chart for a categorical column."""
    print(f"Create pie chart for {column}")
    plt.figure(figsize=(8, 8))
    df[column].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title(f'Pie Chart - {column}')
    plot_file_path = os.path.join(tmpdirname, f'{column}_pie_chart.png')
    plt.savefig(plot_file_path)
    plt.close()
    return plot_file_path


def create_scatter_plot(df, tmpdirname, x_column, y_column):
    """Create a scatter plot for two numeric columns."""
    print(f"Create scatter plot for {x_column} vs {y_column}")
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_column], df[y_column])
    plt.title(f'Scatter Plot - {x_column} vs {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plot_file_path = os.path.join(tmpdirname, f'{x_column}_{y_column}_scatter_plot.png')
    plt.savefig(plot_file_path)
    plt.close()
    return plot_file_path


def create_bubble_chart(df, tmpdirname, x_column, y_column, size_column):
    """Create a bubble chart for three numeric columns with color differentiation."""
    print(f"Create bubble chart for {x_column}, {y_column}, {size_column}")

    # Define a color map based on the range of values in one of the columns (e.g., y_column)
    norm = plt.Normalize(df[y_column].min(), df[y_column].max())
    colors = plt.cm.viridis(norm(df[y_column]))  # Use a colormap like 'viridis' or any other

    plt.figure(figsize=(10, 6))

    # Create the bubble chart, setting colors for each point
    scatter = plt.scatter(df[x_column], df[y_column], s=df[size_column] * 100, alpha=0.5, c=colors)

    plt.title(f'Bubble Chart - {x_column}, {y_column}, {size_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)

    # Add a color bar to represent the scale
    plt.colorbar(scatter, label=y_column)  # Labeling the color bar

    plot_file_path = os.path.join(tmpdirname, f'{x_column}_{y_column}_{size_column}_bubble_chart.png')
    plt.savefig(plot_file_path)
    plt.close()

    return plot_file_path


def create_files_for_dataframe(output_folder_name, df, client, bucket_name):
    """Create text files and plots based on the DataFrame and upload them to MinIO."""

    file_names = {
        "data_summary_file": 'data_summary.txt',
    }

    print('Creating Files in Memory')

    # Create and save text files in the temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Data summary report
        summary_lines = create_data_summary_report(df)
        summary_path = os.path.join(tmpdirname, file_names["data_summary_file"])
        with open(summary_path, 'w') as f:
            f.writelines(summary_lines)
        print(f"Created: {summary_path}")

        # Visualizations
        print('Start Plots')
        plot_files = []

        # Iterate through each column for pie charts and bar charts (for labels)
        for column in df.columns:
            if df[column].dtype in ['object', 'category']:
                # Create pie chart for categorical data (labels)
                plot_files.append(create_pie_chart(df, tmpdirname, column))
                # Create bar chart for categorical data (labels)
                plot_files.append(create_bar_chart(df, tmpdirname, column))
            elif df[column].dtype in ['int64', 'float64']:
                # Create column chart, line chart, area chart for numeric data
                #plot_files.append(create_column_chart(df, tmpdirname, column))
                plot_files.append(create_line_chart(df, tmpdirname, column))
                plot_files.append(create_area_chart(df, tmpdirname, column))

        # Get all numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

        # Create scatter plots for only 3 combinations of 2 numeric columns
        scatter_combinations = list(itertools.combinations(numeric_columns, 2))[:2]
        for x_column, y_column in scatter_combinations:
            plot_files.append(create_scatter_plot(df, tmpdirname, x_column, y_column))

        # Create bubble charts for only 3 combinations of 3 numeric columns
        bubble_combinations = list(itertools.combinations(numeric_columns, 3))[:2]
        for x_column, y_column, size_column in bubble_combinations:
            plot_files.append(create_bubble_chart(df, tmpdirname, x_column, y_column, size_column))

        # Upload all plots and text files to MinIO
        for plot_file in plot_files:
            upload_file_to_minio(client, bucket_name, f"{output_folder_name}/{os.path.basename(plot_file)}",
                                 BytesIO(open(plot_file, 'rb').read()))
            print(f"Uploaded plot: {plot_file}")

        # Upload text files
        for file_key in ["data_summary_file"]:
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
            upload_file_to_minio(client, bucket_name, f"{output_folder_name}/{file_names['data_summary_file'].replace('.txt', '.pdf')}",
                                 BytesIO(f.read()))
        print(f"Uploaded PDF report: {file_names['data_summary_file'].replace('.txt', '.pdf')}")


def create_pdf_report(file_names, tmpdirname):
    """Create a PDF report from text files and saved images."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Data Summary Report", ln=True, align='C')

    # Add data summary report
    pdf.cell(200, 10, txt="Data Summary", ln=True)
    data_summary_path = os.path.join(tmpdirname, file_names['data_summary_file'])
    print(f"Reading data summary from: {data_summary_path}")
    with open(data_summary_path, 'r') as f:
        pdf.multi_cell(0, 10, txt=f.read())

    # Add plots to PDF
    for plot_file in os.listdir(tmpdirname):
        if plot_file.endswith('.png'):  # Ensure we only add image files
            pdf.add_page()
            pdf.image(os.path.join(tmpdirname, plot_file), x=10, w=190)
            print(f"Added plot to PDF: {plot_file}")

    # Save the PDF to a file
    pdf_file_path = os.path.join(tmpdirname, 'Data_Summary_Report.pdf')
    pdf.output(pdf_file_path)
    return pdf_file_path
