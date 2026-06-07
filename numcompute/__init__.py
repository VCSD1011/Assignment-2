"""NumCompute: streaming data processing and machine learning utilities."""

from numcompute.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, Imputer, SimpleImputer
from numcompute.pipeline import Pipeline
from numcompute.tree import DecisionTreeClassifier
from numcompute.ensemble import EnsembleClassifier
from numcompute.stream import StreamTrainer
from numcompute.metrics import StreamingClassificationMetrics, RollingClassificationMetrics
from numcompute.stats import StreamingStats

__all__ = [
    "StandardScaler",
    "MinMaxScaler",
    "OneHotEncoder",
    "Imputer",
    "SimpleImputer",
    "Pipeline",
    "DecisionTreeClassifier",
    "EnsembleClassifier",
    "StreamTrainer",
    "StreamingClassificationMetrics",
    "RollingClassificationMetrics",
    "StreamingStats",
]
