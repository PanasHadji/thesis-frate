import numpy as np
import shap
from sklearn import tree
from sklearn.metrics import roc_auc_score
from sklearn.tree import _tree, DecisionTreeClassifier
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
from alibi.explainers.ale import ALE
from treeinterpreter import treeinterpreter as ti

from service.business_layer.dtos.model_data_dto import Dataset
from service.business_layer.evaluation.eli5_interpreter import Eli5Interpreter
from service.business_layer.evaluation.lime_interpreter import LimeInterpreter
from service.business_layer.evaluation.shapley_interpreter import ShapleyInterpreter
from service.infrastructure_layer.constants import TopFeatures
from service.infrastructure_layer.external.minio import MinIoClient
from service.infrastructure_layer.file_manager import TempFileManager
from service.infrastructure_layer.graph_helper import GraphHelper
from service.infrastructure_layer.logger import Logger
from service.infrastructure_layer.options.minio_options import MinIoConfig
from service.infrastructure_layer.pdf_reporter import PdfBuilder


class TreeInterpreter:
    def __init__(self, clf: DecisionTreeClassifier, initial_info: Dataset, minIoClient: MinIoClient, report_builder: PdfBuilder):
        self.graph_helper = GraphHelper(report_builder)
        self.clf = clf
        self.dataset = initial_info
        self.minIoClient = minIoClient
        self.pdf_report_builder = report_builder


    def interprete_decision_tree(self, target):

        # See: https://github.com/jphall663/interpretable_machine_learning_with_python?tab=readme-ov-file

        # DT Diagram
        self.graph_helper.generate_tree_diagram(self.clf, target, self.dataset.x_columns, self.dataset, self.minIoClient)

        # Feature Importance
        self.extract_feature_importance(self.dataset.x_columns)

        # Decision Rules
        self.extract_decision_rules(self.dataset.x_columns)

        # Gini Impurity
        if self.clf.criterion == 'gini':
            self.extract_gini_impurity_before_training(self.dataset.y_train)

        # Entropy
        if self.clf.criterion == 'entropy':
            self.extract_entropy_before_training(self.dataset.y_train)

        # Node Impurities
        self.node_impurities()

        # Leaf Statistics
        self.extract_leaf_node_statistics(self.dataset.y_unique_values)

        # Assessment Score Distribution
        self.assessment_score_distribution()

        # Partial Dependence Plot
        self.extract_partial_dependence_for_top_n_features(TopFeatures.DEFAULT.value)

        # Assessment Score Rankings
        self.assessment_score_rankings()

        """
        Advanced Interpretation contributing to Trustworthiness.
        """
        # Shapley
        self.extract_shapley_values()

        self.pdf_report_builder.build_report()

        if True is False:

            """
            To be checked !!
            """

            # Tree Code
            #self.extract_tree_code(self.dataset.x_columns) # Remove, handle it above.

            self.extract_lime_values()

            self.extract_eli5_values()



    def extract_leaf_node_statistics(self, unique_y_values):
        # https://chat.openai.com/share/41ad797e-c4dc-4100-bc76-fb72dd31e0be
        print()
        print()
        print('=====> interpretation.extract_leaf_node_statistics <=====')
        try:
            statistics = []
            statistics_text = []
            tree = self.clf.tree_
            leaf_indices = np.where((tree.children_left == -1) & (tree.children_right == -1))[0]
            leaf_class_counts = [tree.value[leaf_index].flatten() for leaf_index in leaf_indices]
            count = 0
            try:

                for index, leaf in zip(leaf_indices, leaf_class_counts):
                    count += 1
                    stats = {unique_y_values[i]: int(count) for i, count in enumerate(leaf)}
                    statistics.append((index, stats))
                    leaf_info = f"{count}.Leaf node {index} has class counts: {stats}"
                    statistics_text.append(leaf_info)
                    print(leaf_info)

                file_name = "leaf_node_statistics.txt"
                with open(file_name, "w", encoding="utf-8") as file:
                    file.write("Leaf Statistics")
                    for leaf_info in statistics_text:
                        file.write(leaf_info + "\n")

                self.pdf_report_builder.append_text_to_report('Leaf Statistics', file_name)
                self.minIoClient.upload_file_to_bucket(MinIoConfig.get_folder_name() + '/' + file_name, file_name)
            except Exception as exception:
                print(f'An error occurred while executing `extract_leaf_node_statistics`. {exception}')

            # Generate Visualization
            self.graph_helper.show_leaf_statistics(statistics, unique_y_values, self.minIoClient)

            return statistics_text
        except Exception as exc:
            print(f'An error occurred while executing `extract_leaf_node_statistics`. {exc}')

    def extract_decision_rules(self, columns):
        print()
        print()
        print('=====> interpretation.extract_decision_rules <=====')
        # See: decision_path
        decision_path = tree.export_text(self.clf, feature_names=columns)
        Logger.log(decision_path)
        file_name = "decision_rules.txt"
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(decision_path)
        self.pdf_report_builder.append_text_to_report('Decision Path', file_name)
        self.minIoClient.upload_file_to_bucket(MinIoConfig.get_folder_name() + '/' + file_name, file_name)
        return decision_path

    def extract_feature_importance(self, columns):
        print()
        print()
        print('=====> interpretation.extract_feature_importance <=====')
        importances = self.clf.feature_importances_
        feature_importance_list = [{"Feature": name, "Importance": imp} for name, imp in zip(columns, importances)]
        sorted_feature_importance_list = sorted(feature_importance_list, key=lambda x: x['Importance'], reverse=True)
        table_str = tabulate(sorted_feature_importance_list, headers="keys", tablefmt='fancy_grid')
        print(table_str)
        # Save the table to a text file
        file_name = "feature_importance.txt"
        TempFileManager.add_temp_file(file_name)
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(table_str)

        file_name_list = "feature_importance_list.txt"

        with open(file_name_list, "w", encoding="utf-8") as file:
            for feature in sorted_feature_importance_list:
                file.write(f"{feature['Feature']}: {feature['Importance']}\n")

        self.pdf_report_builder.append_text_to_report('Feature Importance', file_name_list)
        self.minIoClient.upload_file_to_bucket(MinIoConfig.get_folder_name() + '/' + file_name, file_name)

        return sorted_feature_importance_list

    def extract_gini_impurity_before_training(self, y_train):
        print()
        print()
        print('=====> interpretation.extract_gini_impurity <=====')
        _, counts = np.unique(y_train, return_counts=True)
        proportions = counts / counts.sum()
        gini_impurity = 1 - np.sum(proportions ** 2)
        Logger.log(f'Gini Impurity: {gini_impurity}')
        # Save the table to a text file
        file_name = "gini_impurity_before_training.txt"
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(f'Gini Impurity: {gini_impurity}')
        self.pdf_report_builder.append_text_to_report('Gini Impurity', file_name)
        self.minIoClient.upload_file_to_bucket(MinIoConfig.get_folder_name() + '/' + file_name, file_name)
        return gini_impurity

    def extract_entropy_before_training(self, y_train):

        print()
        print()
        print('=====> interpretation.extract_entropy <=====')
        _, counts = np.unique(y_train, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities))
        Logger.log(f'Entropy: {entropy}')
        # Save the table to a text file
        file_name = "entropy_impurity_before_training.txt"
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(f'Entropy: {entropy}')
        self.pdf_report_builder.append_text_to_report('Entropy', file_name)
        self.minIoClient.upload_file_to_bucket(MinIoConfig.get_folder_name() + '/' + file_name, file_name)
        return entropy

    def extract_tree_code(self, feature_names):
        """

        """
        print()
        print()
        print('=====> interpretation.extract_tree_code <=====')
        code_text = []
        tree_ = self.clf.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        print("def tree({}):".format(", ".join(feature_names)))
        code_text.append("def tree({}):".format(", ".join(feature_names)))

        def recurse(node, depth):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                print("{}if {} <= {}:".format(indent, name, threshold))
                recurse(tree_.children_left[node], depth + 1)
                print("{}else:  # if {} > {}".format(indent, name, threshold))
                recurse(tree_.children_right[node], depth + 1)
            else:
                print("{}return {}".format(indent, np.argmax(tree_.value[node])))

        recurse(0, 1)
        print()
        print()
        return code_text

    def assessment_score_distribution(self):
        """
        Provides insights into how well your model's predicted probabilities
        align with actual outcomes.
        This type of analysis is crucial for understanding model calibration and performance.
        """
        print()
        print()
        print('=====> interpretation.assessment_score_distribution <=====')

        # Predict probabilities
        y_proba = self.clf.predict_proba(self.dataset.x_test)[:, 1]

        # Define probability bins
        bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
        labels = [f"{bins[i]:.2f}-{bins[i + 1]:.2f}" for i in range(len(bins) - 1)]
        df = pd.DataFrame({'y_true': self.dataset.y_test, 'y_proba': y_proba})

        # Bin probabilities and calculate summary statistics
        df['prob_bin'] = pd.cut(df['y_proba'], bins=bins, labels=labels, include_lowest=True)
        summary = df.groupby('prob_bin').agg(
            events=('y_true', 'sum'),
            nonevents=('y_true', lambda x: (x == 0).sum()),
            mean_proba=('y_proba', 'mean')
        ).reset_index()

        summary['percentage'] = summary['events'] / summary['events'].sum() * 100

        summary['mean_proba'] = summary['mean_proba'].fillna(0)

        print(summary)

        self.graph_helper.plot_assessment_score_distribution(summary, self.minIoClient)

        return summary.to_dict(orient='records')

    def assessment_score_rankings(self):
        """
        Provides insight into how well your model performs at
        different depths of the decision tree, as well as
        how it differentiates between positive and negative outcomes.
        """
        print()
        print()
        print('=====> interpretation.assessment_score_rankings <=====')

        if self.dataset.is_binary_classification is False:
            return self.binary_assessment_score_rankings()

        return self.multiclass__assessment_score_rankings()

    def events_distribution(self):
        """
        Shows the distribution of events and nonevents across different model scores.
        The x-axis represents the model score (predicted probability), and the y-axis represents
        the number of events (positive outcomes) and nonevents (negative outcomes) within each score range.
        """
        y_proba = self.clf.predict_proba(self.dataset.x_test)[:, 1]

        df = pd.DataFrame({'y_true': self.dataset.y_test, 'y_proba': y_proba})

        bins = np.linspace(0, 1, 11)
        df['score_bin'] = pd.cut(df['y_proba'], bins=bins, include_lowest=True)

        events = df[df['y_true'] == 1].groupby('score_bin').size()
        nonevents = df[df['y_true'] == 0].groupby('score_bin').size()

        self.graph_helper.plot_event_distribution(events, nonevents)

    def binary_assessment_score_rankings(self):
        print()
        print()
        print('=====> interpretation.binary_assessment_score_rankings <=====')
        metrics = []
        for depth in range(1, self.clf.max_depth + 1, 5):  # Step through depths
            clf = DecisionTreeClassifier(max_depth=depth)
            clf.fit(self.dataset.x_train, self.dataset.y_train)
            y_proba = clf.predict_proba(self.dataset.x_test)[:, 1]
            auc = roc_auc_score(self.dataset.y_test, y_proba)

            df = pd.DataFrame({'y_true': self.dataset.y_test, 'y_proba': y_proba})
            df['decile'] = pd.qcut(df['y_proba'], 10, labels=False, duplicates='drop')

            cum_gain = 0
            cum_lift = 0
            for decile in range(10):
                sub_df = df[df['decile'] == decile]
                events = sub_df['y_true'].sum()
                mean_proba = sub_df['y_proba'].mean()
                lift_value = events / sub_df.shape[0] if sub_df.shape[0] > 0 else 0
                cum_gain += events
                cum_lift += lift_value

                metrics.append({
                    'Depth': depth,
                    'Gain': cum_gain,
                    'Lift': lift_value,
                    'Cumulative Lift': cum_lift / (decile + 1) if decile + 1 > 0 else 0,
                    'Response': (events / sub_df.shape[0]) * 100 if sub_df.shape[0] > 0 else 0,
                    'Cumulative % Response': (cum_gain / df.shape[0]) * 100,
                    'Number of Observations': sub_df.shape[0],
                    'Mean Posterior Probability': mean_proba
                })

        metrics_df = pd.DataFrame(metrics)
        self.accumulate_nan_values(metrics_df)
        print(metrics_df)
        self.graph_helper.binary_assessment_score_rankings(metrics_df, self.minIoClient)
        return metrics

    def multiclass__assessment_score_rankings(self):
        print()
        print()
        print('=====> interpretation.multiclass__assessment_score_rankings <=====')
        y_proba = self.clf.predict_proba(self.dataset.x_test)
        metrics = []
        for class_idx in range(y_proba.shape[1]):
            class_metrics = []
            y_test_binary = (self.dataset.y_test == class_idx).astype(int)
            for depth in range(1, self.clf.max_depth + 1, 5):  # Step through depths
                clf = DecisionTreeClassifier(max_depth=depth)
                clf.fit(self.dataset.x_train, self.dataset.y_train)
                y_proba = clf.predict_proba(self.dataset.x_test)[:, class_idx]
                auc = roc_auc_score(y_test_binary, y_proba)

                df = pd.DataFrame({'y_true': y_test_binary, 'y_proba': y_proba})
                df['decile'] = pd.qcut(df['y_proba'], 10, labels=False, duplicates='drop')

                gain = []
                lift = []
                cum_gain = 0
                cum_lift = 0
                for decile in range(10):
                    sub_df = df[df['decile'] == decile]
                    events = sub_df['y_true'].sum()
                    nonevents = sub_df.shape[0] - events
                    mean_proba = sub_df['y_proba'].mean()
                    gain.append(events)
                    lift.append(events / sub_df.shape[0])
                    cum_gain += events
                    cum_lift += (events / sub_df.shape[0])

                    class_metrics.append({
                        'Class': class_idx,
                        'Depth': depth,
                        'Gain': cum_gain,
                        'Lift': lift[-1],
                        'Cumulative Lift': cum_lift / (decile + 1),
                        'Response': (events / sub_df.shape[0]) * 100,
                        'Cumulative % Response': (cum_gain / df.shape[0]) * 100,
                        'Number of Observations': sub_df.shape[0],
                        'Mean Posterior Probability': mean_proba
                    })
            metrics.extend(class_metrics)
        metrics_df = pd.DataFrame(metrics)
        self.accumulate_nan_values(metrics_df)
        print(metrics_df)

        self.graph_helper.binary_assessment_score_rankings(metrics_df, self.minIoClient)
        return metrics_df

    def accumulate_nan_values(self, metrics_df):
        metrics_df['Lift'] = metrics_df['Lift'].fillna(0)
        metrics_df['Cumulative Lift'] = metrics_df['Cumulative Lift'].fillna(0)
        metrics_df['Response'] = metrics_df['Response'].fillna(0)
        metrics_df['Mean Posterior Probability'] = metrics_df['Mean Posterior Probability'].fillna(0)

    def extract_ale_explanation(self):
        ale = ALE(self.clf.predict, feature_names=self.dataset.column_names)

        # Compute ALE
        ale_exp = ale.explain(self.dataset.x_train)

        # Plot ALE for multiple features
        num_features = 3  # Number of features to plot
        plt.figure(figsize=(15, 10))
        for i in range(num_features):
            plt.subplot(num_features, 1, i + 1)
            plt.plot(ale_exp.feature_values[i], ale_exp.ale_values[i])
            plt.xlabel(self.dataset.column_names[i])
            plt.ylabel('ALE')
            plt.title(f'ALE Plot for {self.dataset.column_names[i]}')
        plt.tight_layout()
        plt.show()

    def extract_variable_contribution(self):
        #Read: https://coderzcolumn.com/tutorials/machine-learning/treeinterpreter-interpreting-tree-based-models-prediction-of-individual-sample
        raise('Not Implemented yet!')

    def extract_tree_interpreter_results(self):
        # Choose a sample to interpret
        sample = self.dataset.x_test[0].reshape(1, -1)

        # Get prediction and contributions
        prediction, bias, contributions = ti.predict(self.clf, sample)

        # Print results
        print("Prediction: ", prediction)
        print("Bias (trainset mean): ", bias)
        print("Feature contributions: ", contributions)

        # Optionally, match contributions with feature names
        feature_names = self.dataset.x_columns
        for name, contribution in zip(feature_names, contributions[0]):
            print(f"{name}: {contribution}")

    def extract_partial_dependence_for_top_n_features(self, number_of_features: int):
        """
         Partial Dependence Plots (PDPs) are used to visualize and interpret
         the relationship between a feature (or features) and the predicted outcome
         of a machine learning model.
         Used for top N features with the highest importance on the Decision Tree model.
        """
        importances = self.clf.feature_importances_
        feature_names = self.dataset.x_columns

        importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importances_df = importances_df.sort_values(by='Importance', ascending=False)

        # Define the top features based on importance
        top_features = importances_df['Feature'].head(number_of_features).tolist()

        self.graph_helper.plot_partial_dependance(model=self.clf, X_train=self.dataset.x_train,
                                                  top_features=top_features, target_names=self.dataset.y_unique_values,
                                                  num_classes=len(self.dataset.y_unique_values), minIoClient=self.minIoClient)

    def extract_decision_boundary(self):
        """
        See: https://scikit-learn.org/stable/modules/generated/sklearn.inspection.DecisionBoundaryDisplay.html
        """
        raise('Not Implemented yet!')
        # Get feature importances and select top N features
        importances = self.clf.feature_importances_
        top_N = TopFeatures.DEFAULT.value
        indices = np.argsort(importances)[-top_N:][::-1]  # Get indices of top N features
        top_features = [self.dataset.x_columns[i] for i in indices]  # Get the names of the top N features

        # Select the top N features from the dataset
        X_train_top = self.dataset.x_train.iloc[:, indices]
        X_test_top = self.dataset.x_test.iloc[:, indices]

        self.graph_helper.plot_decision_boundaries(clf=self.clf, X_train_top=X_train_top, top_features=top_features, y_train=self.dataset.y_train)

    def extract_shapley_values(self):
        """
        See: https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability
        """
        # shap.initjs()
        shapley_interpreter = ShapleyInterpreter(clf=self.clf, initial_info=self.dataset, minIoClient=self.minIoClient, pdf_builder=self.pdf_report_builder)
        shapley_interpreter.get_shap_values()

    def extract_eli5_values(self):
        """
        See: https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability
        """
        # shap.initjs()
        eli5_interpreter = Eli5Interpreter(clf=self.clf, initial_info=self.dataset)
        eli5_interpreter.explain_eli5()

    def extract_lime_values(self):
        """
        See: https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability
        """
        # shap.initjs()
        lime_interpreter = LimeInterpreter(clf=self.clf, initial_info=self.dataset)
        lime_interpreter.explain_lime()

    def node_impurities(self):
        file_name = 'node_impurities.txt'
        split_criterion = self.clf.criterion
        impurity = self.clf.tree_.impurity

        if split_criterion == 'entropy':
            impurity = impurity * np.log2(impurity + 1e-10)  # +1e-10 to avoid log(0)

        # Save impurities to a text file
        with open(file_name, "w", encoding="utf-8") as file:
            for i, imp in enumerate(impurity):
                file.write(f"Node {i}: Impurity = {imp}\n")

        self.pdf_report_builder.append_text_to_report('Node Impurities', file_name)
        self.minIoClient.upload_file_to_bucket(MinIoConfig.get_folder_name() + '/' + file_name, file_name)
        return impurity


