

from enum import Enum


class State(Enum):
    DEFAULT = 42

class SplitSize(Enum):
    DEFAULT = 0.3

class Criterion(Enum):
    Gini = 1,
    Entropy = 2
class Splitter(Enum):
    Best = 1,
    Random = 2

class TopFeatures(Enum):
    DEFAULT = 2

class ShapleyTopFeatures(Enum):
    DEFAULT = 2
class LimeInstance(Enum):
    DEFAULT = 2
class ImbalanceThreshold(Enum):
    DEFAULT = 40