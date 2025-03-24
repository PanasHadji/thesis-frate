import lime
import lime.lime_tabular
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from service.business_layer.dtos.model_data_dto import Dataset
from service.infrastructure_layer.constants import LimeInstance
from service.infrastructure_layer.graph_helper import GraphHelper
import webbrowser

class LimeInterpreter:
    def __init__(self, clf: DecisionTreeClassifier, initial_info: Dataset):
        self.graph_helper = GraphHelper()
        self.clf = clf
        self.dataset = initial_info


    def explain_lime(self):
        # TODO: Upload to Minio.
        # Create a LIME explainer object
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=np.array(self.dataset.x_train),
                feature_names=self.dataset.x_columns,
                class_names=[str(i) for i in np.unique(self.dataset.y_unique_values)],
                mode='classification', discretize_continuous=True)

        # Select an instance to explain
        for i in range(LimeInstance.DEFAULT.value):
            exp = explainer.explain_instance(
                data_row=self.dataset.x_test.iloc[i],
                predict_fn=self.clf.predict_proba
            )
            filename = f'lime_explanation_{i}.html'
            exp.save_to_file(filename)
            webbrowser.open(filename)


