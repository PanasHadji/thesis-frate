from dataclasses import dataclass


@dataclass(kw_only=True)
class Dataset:
    """
    Data class for parameters supporting DecisionTreeClassifier
    """
    x_train: any = None
    x_test: any = None
    y_train: any = None
    y_test: any = None
    x_columns: any = None
    y_unique_values: any = None
    column_names: any = None
    X: any = None
    Y: any = None
    is_binary_classification: bool = False
    data: any = None
