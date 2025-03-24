import io

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dtreeviz import *
from sklearn.inspection import *
from sklearn.tree import plot_tree

from service.business_layer.dtos.model_data_dto import Dataset
from service.infrastructure_layer.external.minio import MinIoClient
from service.infrastructure_layer.image_encoder import ImageEncoder
import pandas as pd

from service.infrastructure_layer.options.minio_options import MinIoConfig
from service.infrastructure_layer.pdf_reporter import PdfBuilder


class GraphHelper:
    def __init__(self):
        print('GraphHelper')
        self.image_encoder = ImageEncoder()

    def __init__(self, report_builder: PdfBuilder):
        print('GraphHelper')
        self.image_encoder = ImageEncoder()
        self.pdf_builder = report_builder

    def show_leaf_statistics(self, statistics, y, minIoClient: MinIoClient):
        try:
            # Number of leaves and classes
            num_leaves = len(statistics)
            num_classes = len(y)

            class_counts = np.zeros((num_leaves, num_classes))

            # Fill the array with class counts
            for i, (_, counts) in enumerate(statistics):
                class_counts[i] = [counts[name] for name in y]

            fig, ax = plt.subplots(figsize=(10, 6))
            width = 0.35  # width of the bars

            for idx, class_name in enumerate(y):
                ax.bar(np.arange(num_leaves) + width * idx, class_counts[:, idx], width, label=class_name)

            # Extract the actual leaf indices for labeling
            leaf_labels = [f'Leaf {index}' for index, _ in statistics]

            ax.set_xticks(np.arange(num_leaves) + width / 2 * (num_classes - 1))
            ax.set_xticklabels(leaf_labels)

            ax.set_xlabel('Leaf Node Index')
            ax.set_ylabel('Number of Instances')
            ax.set_title('Class Distribution per Leaf Node')
            ax.legend(title="Class Names")
        except Exception as exc:
            print(exc)

        local_image_file = "show_leaf_statistics_plot.png"
        plt.savefig(local_image_file)
        plt.close()  # Close the figure to free memory
        print(f"Plot saved locally as '{local_image_file}'")
        self.pdf_builder.append_image_to_report('Leaf Statistics', local_image_file)
        minIoClient.upload_file_to_bucket(MinIoConfig.get_folder_name() + '/' + local_image_file, local_image_file)

    @staticmethod
    def visualize_pruning_process(ccp_alphas, train_scores, test_scores, best_alpha, classifier=''):
        # see: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
        # Visualize Pruning Process
        fig, ax = plt.subplots()
        ax.set_xlabel("alpha")
        ax.set_ylabel("accuracy")
        ax.set_title(f"Accuracy vs alpha for training and testing sets ({classifier})")
        ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
        ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
        ax.legend()

        ax.axvline(x=best_alpha, color='r', linestyle='--', lw=2, label='best alpha')
        ax.legend()
        plt.show()

    @staticmethod
    def visualize_pruning_details(ccp_alphas, clfs, best_alpha, classifier=''):
        node_counts = [clf.tree_.node_count for clf in clfs]
        depth = [clf.tree_.max_depth for clf in clfs]
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
        ax[0].set_xlabel("alpha")
        ax[0].set_ylabel("number of nodes")
        ax[0].set_title("Number of nodes vs alpha")
        ax[0].axvline(x=best_alpha, color='r', linestyle='--', lw=2, label='best alpha')
        ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
        ax[1].set_xlabel("alpha")
        ax[1].set_ylabel("depth of tree")
        ax[1].set_title(f"Depth vs alpha - {classifier}")
        ax[1].axvline(x=best_alpha, color='r', linestyle='--', lw=2, label='best alpha')
        fig.tight_layout()

    def generate_tree_diagram(self, clf, target, feature_names, initial_info: Dataset, minIoClient: MinIoClient):

        # Ensure y_unique_values is a list of class names
        if isinstance(initial_info.y_unique_values, np.ndarray):
            class_names = [str(c) for c in initial_info.y_unique_values]
        else:
            class_names = initial_info.y_unique_values

        try:
            # Create the visualization model
            viz_model = model(
                clf,
                initial_info.x_train,
                initial_info.y_train,
                target_name=target,
                feature_names=feature_names,
                class_names=class_names
            )

            v = viz_model.view(instance_orientation="TD", orientation="TD",
                               show_node_labels=True)  # render as SVG into internal object
            # Save the diagram to a local SVG file
            local_svg_file = "tree_diagram.svg"
            v.save(local_svg_file)
            print(f"Diagram saved locally as '{local_svg_file}'")
            self.pdf_builder.append_image_to_report('Tree Diagram DTreeViz', local_svg_file)
            minIoClient.upload_file_to_bucket(MinIoConfig.get_folder_name() + '/' + local_svg_file, local_svg_file)
            # v.show()  # pop up window
        except:
            print("An exception occurred while creating dtreeviz diagram")


        plt.figure(figsize=(20, 10))
        plot_tree(clf, filled=True, feature_names=feature_names, class_names=target)
        plt.title(f"Decision Tree")

        local_image_file = "tree_diagram_2.png"
        plt.savefig(local_image_file)
        plt.close()  # Close the figure to free memory
        print(f"Plot saved locally as '{local_image_file}'")
        self.pdf_builder.append_image_to_report('Tree Diagram Extended', local_image_file)
        minIoClient.upload_file_to_bucket(MinIoConfig.get_folder_name() + '/' + local_image_file, local_image_file)

    @staticmethod
    def convert_to_byte_format(plt, file_format='png'):
        print('convert_to_byte_format!')
        # Save the plot to a BytesIO object
        bytes_io = io.BytesIO()
        plt.savefig(bytes_io, format=file_format)
        plt.close()  # Close the plot to free up memory

        # Move the cursor to the beginning of the BytesIO object
        bytes_io.seek(0)

        return bytes_io

    def plot_assessment_score_distribution(self, summary, minIoClient: MinIoClient):
        plt.figure(figsize=(12, 8))
        sns.barplot(data=summary, x='prob_bin', y='percentage', palette='viridis')
        plt.xlabel('Posterior Probability Range')
        plt.ylabel('Percentage')
        plt.title('Assessment Score Distribution')
        local_image_file = "assessment_score_distribution.png"
        plt.savefig(local_image_file)
        plt.close()  # Close the figure to free memory
        print(f"Plot saved locally as '{local_image_file}'")
        self.pdf_builder.append_image_to_report('Assessment Score Distribution', local_image_file)
        minIoClient.upload_file_to_bucket(MinIoConfig.get_folder_name() + '/' + local_image_file, local_image_file)

    def binary_assessment_score_rankings(self, metrics_df, minIoClient: MinIoClient):
        # Normalize the metrics
        metrics_df['Gain_norm'] = metrics_df['Gain'] / metrics_df['Gain'].max()
        metrics_df['Lift_norm'] = metrics_df['Lift'] / metrics_df['Lift'].max()
        metrics_df['Cumulative Lift_norm'] = metrics_df['Cumulative Lift'] / metrics_df['Cumulative Lift'].max()

        plt.figure(figsize=(12, 8))
        sns.lineplot(data=metrics_df, x='Depth', y='Gain', palette='viridis', label='Gain')
        sns.lineplot(data=metrics_df, x='Depth', y='Lift', palette='viridis', label='Lift')
        sns.lineplot(data=metrics_df, x='Depth', y='Cumulative Lift', palette='viridis',
                     label='Cumulative Lift')
        plt.xlabel('Tree Depth')
        plt.ylabel('Metrics')
        plt.title('Assessment Score Rankings')
        plt.legend()
        local_image_file = "assessment_score_rankings.png"
        plt.savefig(local_image_file)
        plt.close()  # Close the figure to free memory
        print(f"Plot saved locally as '{local_image_file}'")
        self.pdf_builder.append_image_to_report('Assessment Score Ranking', local_image_file)
        minIoClient.upload_file_to_bucket(MinIoConfig.get_folder_name() + '/' + local_image_file, local_image_file)

    @staticmethod
    def multiclass_assessment_score_rankings(metrics_df):
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=metrics_df, x='Depth', y='Gain', hue='Class', palette='viridis')
        sns.lineplot(data=metrics_df, x='Depth', y='Lift', hue='Class', palette='viridis')
        sns.lineplot(data=metrics_df, x='Depth', y='Cumulative Lift', hue='Class', palette='viridis')
        plt.xlabel('Tree Depth')
        plt.ylabel('Metrics')
        plt.title('Assessment Score Rankings')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_event_distribution(events, nonevents):
        plt.figure(figsize=(12, 8))
        plt.plot(events.index.astype(str), events.values, label='Number of Events', color='blue')
        plt.plot(nonevents.index.astype(str), nonevents.values, label='Number of Nonevents', color='red')
        plt.xlabel('Model Score')
        plt.ylabel('Count')
        plt.title('Number of Events and Nonevents by Model Score')
        plt.legend()
        plt.show()

    def plot_partial_dependance(self, model, X_train, top_features, num_classes, target_names, minIoClient: MinIoClient):
        # Create PDPs for each class
        fig, axes = plt.subplots(num_classes, len(top_features), figsize=(12, 4 * num_classes), sharey=True)
        if num_classes == 1:
            axes = [axes]  # Ensure axes is always a list for consistent indexing

        for class_idx in range(num_classes):
            for feature_idx, feature in enumerate(top_features):
                ax = axes[class_idx][feature_idx]
                display = PartialDependenceDisplay.from_estimator(model, X_train, features=[feature], ax=ax,
                                                                  grid_resolution=50, target=class_idx)
                ax.set_title(f'Class: {target_names[class_idx]}, Feature: {feature}')

        plt.tight_layout()
        local_image_file = "partial_dependence_plot.png"
        plt.savefig(local_image_file)
        plt.close(fig)  # Close the figure to free memory
        print(f"Plot saved locally as '{local_image_file}'")
        self.pdf_builder.append_image_to_report('Partial Dependence', local_image_file)
        minIoClient.upload_file_to_bucket(MinIoConfig.get_folder_name() + '/' + local_image_file, local_image_file)

    def plot_decision_boundaries(self, clf, X_train_top, top_features, y_train, X_full):
        # Create a mesh grid for plotting based on the top features
        xx, yy = np.meshgrid(
            np.linspace(X_train_top.iloc[:, 0].min() - 1, X_train_top.iloc[:, 0].max() + 1, 100),
            np.linspace(X_train_top.iloc[:, 1].min() - 1, X_train_top.iloc[:, 1].max() + 1, 100)
        )

        # Plot the decision boundaries
        # Create a dataframe for prediction with the full set of features
        # Use the mean values of the other features to ensure compatibility

        # Plot the decision boundaries
        fig, ax = plt.subplots(figsize=(10, 6))
        display = DecisionBoundaryDisplay.from_estimator(
            model, X_full, response_method="predict", grid_resolution=100, ax=ax
        )

        # Plot the training points
        scatter = ax.scatter(X_train_top.iloc[:, 0], X_train_top.iloc[:, 1], c=y_train, edgecolor='k', s=50)
        legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend1)

        # Label axes with feature names dynamically
        ax.set_xlabel(top_features[0])
        ax.set_ylabel(top_features[1])
        ax.set_title('Decision Boundaries of Decision Tree Classifier')
        plt.show()
