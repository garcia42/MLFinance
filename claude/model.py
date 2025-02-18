import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import BaggingClassifier

@dataclass
class Model:
    triple_barrier_events: pd.DataFrame
    X_clean: pd.DataFrame
    y_side: pd.Series
    y_size: pd.Series
    combined_weights: pd.Series
    avg_uniq: pd.DataFrame
    fit_size_model: BaggingClassifier
    size_model: BaggingClassifier
    fit_side_model: BaggingClassifier
    side_model: BaggingClassifier