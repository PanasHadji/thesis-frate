import eli5
from eli5.sklearn import PermutationImportance
from sklearn.tree import DecisionTreeClassifier

from service.business_layer.dtos.model_data_dto import Dataset
from service.infrastructure_layer.graph_helper import GraphHelper


class Eli5Interpreter:
    def __init__(self, clf: DecisionTreeClassifier, initial_info: Dataset):
        self.graph_helper = GraphHelper()
        self.clf = clf
        self.dataset = initial_info


    def explain_eli5(self):
        perm = PermutationImportance(self.clf, random_state=1).fit(self.dataset.x_test, self.dataset.y_test)
        print(perm.feature_importances_)

        feature_importances = eli5.show_weights(self.clf, feature_names=self.dataset.x_test.columns)
        print(feature_importances)

        # eli5.show_weights(perm, feature_names=self.dataset.x_test.columns.tolist())
        #
        # eli5.show_weights(self.clf, feature_names=self.dataset.x_train.columns.tolist())

        # Explain a single prediction
        explanation = eli5.explain_prediction(self.clf, self.dataset.x_test.iloc[0])
        eli5.show_prediction(self.clf, self.dataset.x_test.iloc[0])