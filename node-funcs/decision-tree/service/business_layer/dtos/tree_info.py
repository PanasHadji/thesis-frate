from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass(kw_only=True)
class EvaluationInfo:
    roc_auc: Optional[float] = None
    recall: Optional[float] = None
    precision: Optional[float] = None
    accuracy: Optional[float] = None
    misclassification: Optional[float] = None
    training_accuracy: Optional[float] = None
    testing_accuracy: Optional[float] = None
    discrepancy_accuracy: Optional[float] = None

@dataclass(kw_only=True)
class TreeInfoResponse:
    """
    Response model
    """
    feature_importance: Optional[Dict[str, float]] = field(default=None)
    evaluation_info: Optional[EvaluationInfo] = None
    path_analysis: Optional[str] = None
    gini_impurity: Optional[float] = None
    entropy: Optional[float] = None
    alphas: Optional[float] = None


