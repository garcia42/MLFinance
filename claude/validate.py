# Standard library modules
from typing import Tuple, Union

# Third-party modules
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

# FinancialMachineLearning modules
from FinancialMachineLearning.backtest.backtest_statistics import probabilistic_sharpe_ratio
from FinancialMachineLearning.cross_validation.combinatorial import CombinatorialPurgedKFold

# Claude modules
from claude.model import Model
from claude.train_model import build_model


def _split_data(X: Union[pd.DataFrame, pd.Series], split_date: str = '2023-01-01') -> tuple[Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series]]:
    """
    Split data into training and testing sets based on date
    
    Args:
        X: DataFrame or Series with datetime index
        split_date: String date in format 'YYYY-MM-DD'
    
    Returns:
        tuple of (train_data, test_data)
    """
    # Convert index to datetime if it isn't already
    if not isinstance(X.index, pd.DatetimeIndex):
        X.index = pd.to_datetime(X.index, unit='ms')
        
    # Convert split_date to timestamp
    split_timestamp = pd.Timestamp(split_date)
    
    # Debug prints
    print("First date in data:", X.index.min())
    print("Last date in data:", X.index.max())
    print("Split date:", split_timestamp)
    
    train_mask = X.index < split_date

    print(f"Training samples: {len(X[train_mask])}, Testing samples: {len(X[~train_mask])}")
    
    return X[train_mask], X[~train_mask]

def label_and_cross_validate(features: pd.DataFrame):
    model = build_model(features=features)
    _cross_validate(model)

def _cross_validate(model: Model) -> Tuple[BaggingClassifier, dict]:
    # 1. Handle NaN and infinite values
    # First, let's see where the problematic values are
    print("\nChecking for NaN values in features:")
    print(model.X_clean.isna().sum()[model.X_clean.isna().sum() > 0])
    
    print("\nChecking for infinite values in features:")
    print(np.isinf(model.X_clean.select_dtypes(include=np.number)).sum()[np.isinf(model.X_clean.select_dtypes(include=np.number)).sum() > 0])
    
    X_train, X_test = _split_data(model.X_clean, "2023-01-01")
    y_train, y_test = _split_data(model.y_size, "2023-01-01")
    
    samples_info_sets = model.triple_barrier_events.loc[X_train.index].loc[:'2023-01-01', 't1']

    comb_purge_fold = CombinatorialPurgedKFold(
        n_splits = 5,
        n_test_splits = 2,
        samples_info_sets = samples_info_sets,
        pct_embargo = 0.01
    )

    # Check sizes at each step
    print("Original X_train shape:", X_train.shape)
    print("Triple barrier event shape:", model.triple_barrier_events.shape)
    print("Samples after first filter:", model.triple_barrier_events.loc[X_train.index].shape)
    print("Samples after date filter:", model.triple_barrier_events.loc[X_train.index].loc[:'2023-01-01'].shape)
    print("Final samples_info_sets shape:", samples_info_sets.shape)

    # Check date ranges
    print("X_train index range:", X_train.index.min(), "to", X_train.index.max())
    print("Triple barrier dates:", model.triple_barrier_events.index.min(), "to", model.triple_barrier_events.index.max())

    # Check for NaN values
    print("NaN values in t1:", model.triple_barrier_events['t1'].isna().sum())

    accuracies = []
    for train_indices, test_indices in comb_purge_fold.split(X_train, y_train):
        X_train_valid, X_test_valid = X_train.iloc[train_indices], X_train.iloc[test_indices]
        y_train_valid, y_test_valid = y_train.iloc[train_indices], y_train.iloc[test_indices]

        # Get the corresponding weights for this training subset
        if hasattr(model, 'combined_weights'):
            train_weights = model.combined_weights.loc[X_train_valid.index]
        else:
            train_weights = None

        random_forest_weak = RandomForestClassifier(
            n_estimators = 1,
            criterion = 'entropy',
            bootstrap = False,
            class_weight = 'balanced_subsample',
            min_weight_fraction_leaf = 0.05,
            max_features = 3,
            random_state = 42
        )
        
        # Get the average unique samples for this training subset
        if hasattr(model, 'avg_uniq'):
            avg_unique = model.avg_uniq.loc[X_train_valid.index].mean()[0]
        else:
            avg_unique = int(len(X_train_valid) * 0.5)  # default to 50% if avg_uniq not available
        
        print(f"Max samples = {avg_unique}")
        forest_bagging = BaggingClassifier(
            base_estimator = random_forest_weak,
            n_estimators = 1000,
            max_samples = avg_unique,
            max_features = .5,
            random_state = 42
        )
        
        # Fit with weights if available
        if train_weights is not None:
            forest_bagging_fit = forest_bagging.fit(
                X = X_train_valid,
                y = y_train_valid,
                sample_weight = train_weights
            )
        else:
            forest_bagging_fit = forest_bagging.fit(
                X = X_train_valid,
                y = y_train_valid
            )

        # Get predictions and probabilities
        y_pred = forest_bagging_fit.predict(X_test_valid)
        accuracy = accuracy_score(y_test_valid, y_pred)
        accuracies.append(accuracy)
        print(f'Out of bag Accuracy: {accuracy:.4f}')
        
        y_pred = forest_bagging_fit.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Out of sample Accuracy: {accuracy:.4f}')
    
    print(f'\nMean cross-validation accuracy: {np.mean(accuracies):.4f}')
    print(f'Standard deviation of accuracies: {np.std(accuracies):.4f}')

def calculate_returns(model: Model):
    # Convert predictions to pandas Series with indices
    side_preds = pd.Series(model.side_model.predict(model.X_clean), index=model.X_clean.index)
    size_preds = pd.Series(model.size_model.predict_proba(model.X_clean)[:, 1], index=model.X_clean.index)
    
    # Get actual returns
    actual_returns = model.triple_barrier_events['ret']
    
    # Get common index between all series
    common_index = side_preds.index.intersection(size_preds.index).intersection(actual_returns.index)
    
    # Align all series to the common index
    side_preds = side_preds.loc[common_index]
    size_preds = size_preds.loc[common_index]
    actual_returns = actual_returns.loc[common_index]
    
    print(f"Aligned shapes - side: {len(side_preds)}, size: {len(size_preds)}, returns: {len(actual_returns)}")
    
    # Calculate strategy returns
    strategy_returns = side_preds * size_preds * actual_returns
    
    return strategy_returns

def calculate_psr(model: Model, target_sharpe=1, n_obs=252):
    """
    Calculate probabilistic Sharpe ratio
    """
    # Get strategy returns not raw returns
    returns = calculate_returns(model)
    
    print("Returns stats:")
    print("Mean:", returns.mean())
    print("Std:", returns.std())
    print("Number of returns:", len(returns))
    
    # Calculate annualized Sharpe ratio
    sr_hat = returns.mean() / returns.std() * np.sqrt(252)
    print("Sharpe ratio:", sr_hat)
    
    # Calculate higher moments
    skewy = skew(returns)
    kurt = kurtosis(returns)
    print("Skewness:", skewy)
    print("Kurtosis:", kurt)
    
    # Calculate components of PSR formula
    numerator = (sr_hat - target_sharpe) * (n_obs - 1) ** (1/2)
    denominator = (1 - skewy * sr_hat + (kurt - 1)/4 * sr_hat**2) ** (1/2)

    print("\nPSR Components:")
    print("Numerator:", numerator)
    print("Denominator:", denominator)

    psr = probabilistic_sharpe_ratio(
        observed_sr=sr_hat,
        benchmark_sr=target_sharpe,
        number_of_returns=n_obs,
        skewness_of_returns=skewy,
        kurtosis_of_returns=kurt
    )

    return psr
