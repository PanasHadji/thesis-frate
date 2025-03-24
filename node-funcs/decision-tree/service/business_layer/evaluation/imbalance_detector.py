from service.business_layer.dtos.model_data_dto import Dataset
from service.infrastructure_layer.constants import ImbalanceThreshold
from service.infrastructure_layer.file_manager import TempFileManager
from service.infrastructure_layer.pdf_reporter import PdfBuilder


class ImbalanceDetector:

    def __init__(self):
        print('ImbalanceDetector')

    def analyse_imbalances_in_target_variable(self, dataset: Dataset, target):
        file_name = "imbalance_analysis.txt"
        TempFileManager.add_temp_file(file_name)

        # Assuming df is your DataFrame and 'target' is your target variable column
        class_counts = dataset.data[target].value_counts(normalize=True) * 100

        # Print the percentage of each class
        print(class_counts)

        max_class_proportion = class_counts.max()
        min_class_proportion = class_counts.min()
        difference = max_class_proportion - min_class_proportion

        is_imbalanced = difference > ImbalanceThreshold.DEFAULT.value

        results = {
            "Class Percentages": class_counts.to_dict(),
            "Difference between max and min class proportions (%)": difference,
            "Class imbalance exists": is_imbalanced
        }

        # Print the results
        percentages_text = f"Class percentages:\n{class_counts}"
        difference_text = f"Difference between max and min class proportions: {difference:.2f}%"
        is_imbalanced_text = f"Class imbalance exists: {is_imbalanced}"
        print(percentages_text)
        print(difference_text)
        print(is_imbalanced_text)

        with open(file_name, "w") as file:
            file.write(percentages_text)
            file.write(difference_text)
            file.write(is_imbalanced_text)

        return file_name