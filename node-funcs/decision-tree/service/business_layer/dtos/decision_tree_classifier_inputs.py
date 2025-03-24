from dataclasses import dataclass
from typing import Optional

from service.infrastructure_layer.constants import State


@dataclass(kw_only=True)
class DecisionTreeClassifierInputs:
    """
    Data class for parameters supporting DecisionTreeClassifier
    """
    random_state: int = State.DEFAULT.value
    criterion: Optional[str] = 'gini'
    splitter: Optional[str] = 'best'
    max_depth: Optional[int] = None
    min_samples_split: Optional[float] = 2
    min_samples_leaf: Optional[int] = 1
    min_weight_fraction_leaf: Optional[float] = 0.0
    max_features: Optional[str] = None
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: Optional[float] = 0.0
    class_weight: Optional[any] = None
    ccp_alpha: Optional[float] = 0.0
