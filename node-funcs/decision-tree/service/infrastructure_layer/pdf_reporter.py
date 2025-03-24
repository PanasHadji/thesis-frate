import os

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from service.infrastructure_layer.external.minio import MinIoClient
from service.infrastructure_layer.file_manager import TempFileManager
from service.infrastructure_layer.options.minio_options import MinIoConfig


class PdfBuilder:
    def __init__(self, minIoClient: MinIoClient):
        self.minIoClient = minIoClient
        print('PdfReporter')
        self.OUTPUT_PATH = "ml_model_report.pdf"
        self.pdf = SimpleDocTemplate(self.OUTPUT_PATH, pagesize=letter)
        self.elements = []
        self.styles = getSampleStyleSheet()

    def create_report_heading(self):
        title = Paragraph("ML Model Trustworthiness Report", self.styles['Title'])
        self.elements.append(title)
        self.elements.append(Spacer(1, 12))

    def append_image_to_report(self, title, local_file):
        TempFileManager.add_temp_file(local_file)

        svg_title = Paragraph(title, self.styles['Heading2'])
        self.elements.append(svg_title)
        self.elements.append(Spacer(1, 12))

        # Convert SVG to PNG if needed
        if local_file.endswith(".svg"):
            from svglib.svglib import svg2rlg
            drawing = svg2rlg(local_file)
            self.scale_drawing(drawing, max_width=450, max_height=600)
            self.elements.append(drawing)
        else:
            img = Image(local_file, width=450, height=600)
            self.elements.append(img)
        self.elements.append(Spacer(1, 12))

    @staticmethod
    def scale_drawing(drawing, max_width, max_height):
        # Get the original width and height of the drawing
        original_width = drawing.width
        original_height = drawing.height

        # Calculate the scaling factors for width and height
        scale_x = max_width / original_width
        scale_y = max_height / original_height

        # Use the smaller scaling factor to maintain aspect ratio
        scale = min(scale_x, scale_y)

        # Apply the scaling
        drawing.width = original_width * scale
        drawing.height = original_height * scale
        drawing.scale(scale, scale)

    def append_text_to_report(self, title, text_path):
        TempFileManager.add_temp_file(text_path)

        text_title = Paragraph(title, self.styles['Heading2'])
        self.elements.append(text_title)
        self.elements.append(Spacer(1, 12))

        with open(text_path, 'r', encoding='utf-8') as text_file:
            text_content = text_file.read()

        text_paragraph = Paragraph(text_content.replace('\n', '<br/>'), self.styles['BodyText'])
        self.elements.append(text_paragraph)
        self.elements.append(Spacer(1, 12))

    def append_performance_metrics(self, metrics):
        metrics_title = Paragraph("Model Performance Metrics", self.styles['Heading2'])
        self.elements.append(metrics_title)
        self.elements.append(Spacer(1, 12))

        # Convert metrics to a list of lists for Table
        data = [[key, str(value)] for key, value in metrics.items()]
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        self.elements.append(table)
        self.elements.append(Spacer(1, 12))

    def build_report(self):
        self.pdf.build(self.elements)
        print(f"Report generated: {self.OUTPUT_PATH}")

        # Upload PDF report to MinIO bucket
        self.minIoClient.upload_file_to_bucket(MinIoConfig.get_folder_name() + '/' + self.OUTPUT_PATH, self.OUTPUT_PATH)
        print(f"PDF report uploaded to MinIO as '{self.OUTPUT_PATH}'")

        # Clean up local resources
        os.remove(self.OUTPUT_PATH)
