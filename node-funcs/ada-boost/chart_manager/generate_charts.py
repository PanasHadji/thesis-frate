import os
import tempfile
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
import seaborn as sns
import pandas as pd

def auto_generate_charts_and_report(df, y_pred, output_folder_name, client, bucket_name):
    """
    Automatically generate visualizations and a PDF report from a DataFrame and predictions.
    """
    # Identify categorical and numeric columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    numeric_cols = df.select_dtypes(include=['number']).columns

    # File paths and structure
    file_names = {
        "pdf_file_name": "Visualization_Report.pdf",
        "charts": []
    }

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Generate charts based on requirements
        for col in numeric_cols:
            # Column chart
            print('Column chart')
            create_bar_or_column_chart(df, col, tmpdirname, f"{col}_column_chart.png", file_names, chart_type="column")
            print('Bar chart')
            # Bar chart
            create_bar_or_column_chart(df, col, tmpdirname, f"{col}_bar_chart.png", file_names, chart_type="bar")
            print('Line chart')
            # Line chart
            create_line_chart(df, col, tmpdirname, f"{col}_line_chart.png", file_names)
            print('Area chart')
            # Area chart
            create_area_chart(df, col, tmpdirname, f"{col}_area_chart.png", file_names)

        for col in categorical_cols:
            # Pie chart
            print('Pie chart')
            create_pie_chart(df, col, tmpdirname, f"{col}_pie_chart.png", file_names, threshold=0.05)

        # Scatter chart (2 numeric dimensions)
        if len(numeric_cols) >= 2:
            for i in range(len(numeric_cols) - 1):
                print('Scatter chart')
                create_scatter_chart(df, numeric_cols[i], numeric_cols[i + 1], tmpdirname,
                                     f"{numeric_cols[i]}_vs_{numeric_cols[i + 1]}_scatter.png", file_names)

        # Bubble chart (3+ numeric dimensions)
        if len(numeric_cols) >= 3:
            print('Bubble chart')
            create_bubble_chart(df, numeric_cols[:3], tmpdirname, "bubble_chart.png", file_names)

        # Create PDF report
        print('Creating PDF')
        pdf_file_path = create_pdf_report(file_names, tmpdirname)

        # Upload all generated files to MinIO
        print('Uploading all files')
        for chart_file in file_names["charts"]:
            with open(chart_file, 'rb') as f:
                client.put_object(bucket_name, f"{output_folder_name}/{os.path.basename(chart_file)}", f, os.path.getsize(chart_file))

        print('Uploading PDF')
        with open(pdf_file_path, 'rb') as f:
            client.put_object(bucket_name, f"{output_folder_name}/{file_names['pdf_file_name']}", f, os.path.getsize(pdf_file_path))


def create_bar_or_column_chart(df, col, tmpdirname, file_name, file_names, chart_type):
    """Create bar or column chart."""
    plt.figure(figsize=(10, 6))
    if chart_type == "bar":
        sns.barplot(x=df.index, y=df[col])
    else:
        plt.bar(df.index, df[col])
    plt.title(f"{chart_type.capitalize()} Chart - {col}")
    chart_path = os.path.join(tmpdirname, file_name)
    plt.savefig(chart_path)
    plt.close()
    file_names["charts"].append(chart_path)


def create_line_chart(df, col, tmpdirname, file_name, file_names):
    """Create line chart."""
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[col])
    plt.title(f"Line Chart - {col}")
    chart_path = os.path.join(tmpdirname, file_name)
    plt.savefig(chart_path)
    plt.close()
    file_names["charts"].append(chart_path)


def create_area_chart(df, col, tmpdirname, file_name, file_names):
    """Create area chart."""
    plt.figure(figsize=(10, 6))
    plt.fill_between(df.index, df[col])
    plt.title(f"Area Chart - {col}")
    chart_path = os.path.join(tmpdirname, file_name)
    plt.savefig(chart_path)
    plt.close()
    file_names["charts"].append(chart_path)


def create_pie_chart(df, col, tmpdirname, file_name, file_names, threshold=0.05):
    """
    Create a pie chart with grouped categories for smaller groups.
    Parameters:
    - df: The DataFrame containing the data.
    - col: The column name to create the pie chart for.
    - tmpdirname: Temporary directory to save the chart.
    - file_name: File name for the chart.
    - file_names: Dictionary to store chart paths.
    - threshold: Percentage threshold to group smaller categories as 'Other'.
    """
    # Count the occurrences of each category
    category_counts = df[col].value_counts()

    # Calculate the threshold count (based on the total number of records)
    total_count = category_counts.sum()
    threshold_count = total_count * threshold

    # Group categories below the threshold into "Other"
    other_categories = category_counts[category_counts < threshold_count].index
    df[col] = df[col].apply(lambda x: 'Other' if x in other_categories else x)

    # Create the pie chart
    plt.figure(figsize=(8, 8))
    df[col].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title(f"Pie Chart - {col}")
    chart_path = os.path.join(tmpdirname, file_name)
    plt.savefig(chart_path)
    plt.close()
    file_names["charts"].append(chart_path)


def create_scatter_chart(df, col_x, col_y, tmpdirname, file_name, file_names):
    """Create scatter chart."""
    plt.figure(figsize=(10, 6))
    plt.scatter(df[col_x], df[col_y], c='blue', alpha=0.5)
    plt.title(f"Scatter Chart - {col_x} vs {col_y}")
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    chart_path = os.path.join(tmpdirname, file_name)
    plt.savefig(chart_path)
    plt.close()
    file_names["charts"].append(chart_path)


def create_bubble_chart(df, cols, tmpdirname, file_name, file_names):
    """Create bubble chart with 3+ numeric dimensions."""
    if len(cols) < 3:
        return
    plt.figure(figsize=(10, 6))
    plt.scatter(df[cols[0]], df[cols[1]], s=df[cols[2]], alpha=0.5, c=df[cols[2]], cmap='viridis')
    plt.title(f"Bubble Chart - {cols[0]}, {cols[1]}, {cols[2]}")
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    chart_path = os.path.join(tmpdirname, file_name)
    plt.savefig(chart_path)
    plt.close()
    file_names["charts"].append(chart_path)


def create_pdf_report(file_names, tmpdirname):
    """Create a PDF report including all charts."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Visualization Report", ln=True, align='C')

    for chart_file in file_names["charts"]:
        pdf.add_page()
        pdf.image(chart_file, x=10, w=190)

    pdf_file_path = os.path.join(tmpdirname, file_names["pdf_file_name"])
    pdf.output(pdf_file_path)
    return pdf_file_path
