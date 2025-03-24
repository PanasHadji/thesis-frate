from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate

from service.business_layer.dtos.model_data_dto import Dataset
from service.business_layer.evaluation.evaluation import TreeEvaluator
from service.infrastructure_layer.pdf_reporter import PdfBuilder


def decision_tree_classifier(parameters, data_inputs: Dataset, pdf_builder: PdfBuilder):
    print('START decision_tree_classifier')

    print("Parameters being passed to DecisionTreeClassifier:")
    for key, value in parameters.items():
        print(f"{key}: {value}")

    clf = DecisionTreeClassifier(**parameters)
    clf.fit(data_inputs.x_train, data_inputs.y_train)

    tree_evaluator = TreeEvaluator()
    model_performance = tree_evaluator.generate_performance(clf, data_inputs)
    pdf_builder.append_performance_metrics(model_performance)
    evaluations = [model_performance]
    print(tabulate(evaluations, headers='keys', tablefmt='fancy_grid'))
    return clf



