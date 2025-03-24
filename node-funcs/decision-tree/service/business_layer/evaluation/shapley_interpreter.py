
import shap
from sklearn.tree import DecisionTreeClassifier

from service.business_layer.dtos.model_data_dto import Dataset
from service.infrastructure_layer.constants import ShapleyTopFeatures
from service.infrastructure_layer.external.minio import MinIoClient
from service.infrastructure_layer.file_manager import TempFileManager
import matplotlib.pyplot as plt

from service.infrastructure_layer.options.minio_options import MinIoConfig
from service.infrastructure_layer.pdf_reporter import PdfBuilder

"""
See: https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability
"""


class ShapleyInterpreter:

    def __init__(self, clf: DecisionTreeClassifier, initial_info: Dataset, minIoClient: MinIoClient, pdf_builder: PdfBuilder):
        self.clf = clf
        self.dataset = initial_info
        self.minIoClient = minIoClient
        self.pdf_builder = pdf_builder

    def get_shap_values(self):

        # Extract feature names
        feature_names = self.dataset.x_test.columns

        explainer = shap.TreeExplainer(self.clf)

        shap_values = explainer.shap_values(self.dataset.x_test)
        # self.save_to_minio_bucket('shapley_variable_impact.png')

        # Summary plot for feature importance
        shap_global_variable_impact1_file = 'shapley_global_variable_impact_1.png'
        shap.summary_plot(shap_values, self.dataset.x_test, show=False)
        self.save_to_minio_bucket(shap_global_variable_impact1_file, "Global Variable Impact")


        # shap_values[0] -> represents class at position 0
        num_classes = len(shap_values)
        for class_index in range(num_classes):
            shap.summary_plot(shap_values[class_index],  self.dataset.x_test, show=False)
            local_title = f'SHAP Summary Plot for Class {class_index}'
            local_file_name = f'shapley_variable_impact_2_CLASS_{class_index}.png'
            self.save_to_minio_bucket(local_file_name, local_title)


        # Calculate interaction values
        shapley_interactions_file = 'shapley_interactions.png'
        interaction_values = shap.TreeExplainer(self.clf).shap_interaction_values(self.dataset.x_test)
        shap.summary_plot(interaction_values[0], self.dataset.x_test, show=False)
        self.save_to_minio_bucket(shapley_interactions_file, "SHAP Interaction Values")


        if len(feature_names) >= ShapleyTopFeatures.DEFAULT.value:
            # Extract the top features (e.g., top 2 for dependence plot)
            number_of_features = ShapleyTopFeatures.DEFAULT.value
            top_features = feature_names[: number_of_features]
            shap.dependence_plot(top_features[0], shap_values[0], self.dataset.x_test,
                                 interaction_index=top_features[1], show=False)
            self.save_to_minio_bucket('shapley_dependence_plot.png', "SHAP Dependence Plot")

        shap.plots.force(explainer.expected_value[0], shap_values[0][0, :], self.dataset.x_test.iloc[0, :],
                         matplotlib=True, show=False)

        # Note: The target label “1” decision plot is tilted towards “1”.
        shap.decision_plot(explainer.expected_value[1], shap_values[1], )
        for target in self.dataset.y_unique_values:
            shap.decision_plot(explainer.expected_value[target], shap_values[target], self.dataset.x_test.columns,
                               title=f"Decision Plot for Target {target}", show=False)
            self.save_to_minio_bucket(f'shapley_decision_plot_Class_{target}.png', "SHAP Decision Plot")
            # plt.show()

    def save_to_minio_bucket(self, image_name, report_section_title):
        plt.savefig(image_name)
        plt.close()  # Close the figure to free memory
        print(f"Plot saved locally as '{image_name}'")
        TempFileManager.add_temp_file(image_name)

        self.pdf_builder.append_image_to_report(report_section_title, image_name)
        self.minIoClient.upload_file_to_bucket(MinIoConfig.get_folder_name() + '/' + image_name, image_name)
