from aif360.metrics import BinaryLabelDatasetMetric
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from service.business_layer.dtos.model_data_dto import Dataset
from aif360.datasets import BinaryLabelDataset
import numpy as np

# [OBSOLETE]
class BiasDetector:
    def __init__(self):
        print('Bias Detector')

    def find_bias_on_trained_classifier(self, clf:DecisionTreeClassifier, dataset: Dataset):

        df = pd.DataFrame(dataset.data, columns=dataset.x_columns)
        df['diagnosis'] = dataset.Y

        # Adding a synthetic protected attribute 'gender' (1: Male, 0: Female)
        np.random.seed(0)  # Ensure reproducibility
        df['gender'] = np.random.choice([0, 1], size=len(df))

        # Convert DataFrame to BinaryLabelDataset
        dataset = BinaryLabelDataset(df=df, label_names=['diagnosis'], favorable_label=1,
                                     protected_attribute_names=['gender'])

        # Check for bias before reweighing
        metric = BinaryLabelDatasetMetric(dataset, unprivileged_groups=[{'gender': 0}],
                                          privileged_groups=[{'gender': 1}])

        print("Disparate impact before reweighing: ", metric.disparate_impact())

    # Function to load the breast cancer dataset and prepare the protected attribute
    def load_and_prepare_data(protected_attribute, dataset: Dataset, threshold=None):
        df = pd.DataFrame(dataset.data, columns=dataset.x_columns)
        df['diagnosis'] = dataset.Y

        df = pd.DataFrame(dataset.data, columns=dataset.x_columns)
        df['diagnosis'] = dataset.Y

        if pd.api.types.is_numeric_dtype(df[protected_attribute]):
            if threshold is None:
                threshold = df[protected_attribute].median()
            df[protected_attribute + '_binary'] = np.where(df[protected_attribute] > threshold, 1, 0)
            protected_attribute += '_binary'
        else:
            unique_values = df[protected_attribute].unique()
            if len(unique_values) != 2:
                raise ValueError("Protected attribute must be binary or need to provide a way to binarize it.")
            df[protected_attribute] = df[protected_attribute].apply(lambda x: 1 if x == unique_values[1] else 0)

        return df, protected_attribute

    # Function to calculate disparate impact
    def calculate_disparate_impact(df, protected_attribute):
        dataset = BinaryLabelDataset(
            favorable_label=1,
            unfavorable_label=0,
            df=df,
            label_names=['diagnosis'],
            protected_attribute_names=[protected_attribute],
            privileged_protected_attributes=[[1]],
            unprivileged_protected_attributes=[[0]]
        )

        # Check for bias before reweighing
        metric = BinaryLabelDatasetMetric(dataset, unprivileged_groups=[{protected_attribute: 0}],
                                          privileged_groups=[{protected_attribute: 1}])


        print("Disparate impact after reweighing: ", metric.disparate_impact())