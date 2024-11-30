import sys
import os
import pickle
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from feature_storage import FeatureStorage

import pandas as pd
from FinancialMachineLearning.labeling.labeling import meta_labeling, get_events, add_vertical_barrier
from FinancialMachineLearning.features.volatility import daily_volatility, daily_volatility_intraday
from sklearn.ensemble import RandomForestClassifier
from FinancialMachineLearning.sample_weights.bootstrapping import *
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler

from FinancialMachineLearning.sample_weights.concurrency import *
from FinancialMachineLearning.utils.multiprocess import *
from FinancialMachineLearning.sample_weights.attribution import *
from FinancialMachineLearning.cross_validation.combinatorial import CombinatorialPurgedKFold
from feature_analysis import FeatureAnalysis

from typing import Any, Tuple, Union

REGENERATE_CACHE=True

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

def label_and_analyze(features: pd.DataFrame): # Don't hold out any data, put all data into the classifier and K folds
    # X, triple_barrier_events, y = _label(features=features.copy())
    # combined_weights, avg_uniq = _weights(features=features, triple_barrier_events=triple_barrier_events)
    # X_clean, y, combined_weights = _clean_features(X_df=X, combined_weights=combined_weights, y=y)

    # Analyze feature importance for this model
    tfs = FeatureStorage('tbe_before_analyze.parquet')
    # tfs.save_features(triple_barrier_events)
    triple_barrier_events = tfs.load_existing_features()[0]

    fs = FeatureStorage('X_before_analyze.parquet')
    # fs.save_features(X_clean)
    X_clean = fs.load_existing_features()[0]

    fsy = FeatureStorage('y_before_analyze.parquet')
    print("Y from label_and_analyze")
    # print(y)
    # fsy.save_features(pd.DataFrame({y.name: y}))
    y = fsy.load_existing_features()[0].squeeze()

    fsw = FeatureStorage('weights_before_analyze.parquet')
    # fsw.save_features(pd.DataFrame({combined_weights.name: combined_weights}))
    combined_weights = fsw.load_existing_features()[0].squeeze()
    
    fsu = FeatureStorage('uniq_before_analysis.parquet')
    # fsu.save_features(avg_uniq)
    avg_uniq = fsu.load_existing_features()[0]

    _analyze_features(triple_barrier_events=triple_barrier_events, X=X_clean, y=y, combined_weights=combined_weights, avg_uniqueness=avg_uniq)

def label_and_train(features: pd.DataFrame):
    X, triple_barrier_events, y = _label(features=features.copy())
    combined_weights, avg_uniq = _weights(features=features, triple_barrier_events=triple_barrier_events)
    X_clean, y, combined_weights = _clean_features(X_df=X, combined_weights=combined_weights, y=y)
    _train_model(triple_barrier_events=triple_barrier_events, X=X_clean, y=y, combined_weights=combined_weights, avg_uniqueness=avg_uniq)

# Data cleaning and validation
def _clean_features(X_df: pd.DataFrame, combined_weights: pd.Series, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:

    X_scaled = X_df.copy()    
    # First handle any infinite values
    X_scaled = X_scaled.replace([np.inf, -np.inf], np.nan)
    # For each column, fill NaN with median and apply robust scaling
    scaler = RobustScaler(with_centering=True, with_scaling=True, unit_variance=True)
    for col in X_scaled.columns:
            # Fill NaN with median
            col_median = X_scaled[col].median()
            X_scaled[col] = X_scaled[col].fillna(col_median)
            
            # Clip extreme values to 1st and 99th percentiles
            q1, q99 = X_scaled[col].quantile([0.01, 0.99])
            X_scaled[col] = X_scaled[col].clip(q1, q99)
            
            # Scale the column
            X_scaled[col] = scaler.fit_transform(X_scaled[[col]])

    # Additional data validation and cleaning
    X_scaled = X_scaled.astype(np.float32)  # Convert to float32 explicitly

    # Clip extreme values to prevent overflow
    clip_threshold = np.finfo(np.float32).max / 10
    X_scaled = X_scaled.clip(-clip_threshold, clip_threshold)
        
    print(f"\nAfter scaling - column ranges:")
    print("Max values:", X_scaled.max())
    print("Min values:", X_scaled.min())
    
    # Verify data quality
    print("\nData validation:")
    print("NaN values:", X_scaled.isna().sum().sum())
    print("Infinite values:", np.isinf(X_scaled.values).sum())
    print("Max absolute value:", np.abs(X_scaled.values).max())
    

    if y is not None and combined_weights is not None:
        # Get common valid indices
        valid_idx = X_df.index.intersection(y.index).intersection(combined_weights.index)
        return X_df.loc[valid_idx], y.loc[valid_idx], combined_weights.loc[valid_idx]
    
    return X_df

def _weights(features: pd.DataFrame, triple_barrier_events: Any) -> Tuple[pd.Series, pd.DataFrame]:
    # Get initial weights
    return_weights = weights_by_return(triple_barrier_events[:-5], features['close'][:-5], num_threads=1)
    
    time_decay, avg_unique = weights_by_time_decay(
        triple_barrier_events=triple_barrier_events[:-10],
        close_series=features['close'],
        num_threads=1,
        decay=0.5
    )
    
    # Combine weights
    combined_weights = np.sqrt(return_weights * time_decay)
    
    # Prevent any sample from being completely ignored
    min_weight = 0.01
    combined_weights = np.maximum(combined_weights, min_weight)
    
    # Ensure weights sum to 1
    combined_weights = combined_weights / combined_weights.sum()
    
    # Handle NaN values in avg_uniqueness
    if avg_unique is not None:
        avg_unique = avg_unique.fillna(avg_unique.mean())
    
    # Create a Series with the same index as features
    combined_weights = pd.Series(combined_weights, index=return_weights.index)
    
    # Align weights with features index
    # This is crucial - we only want weights for samples that exist in our feature matrix
    combined_weights = combined_weights.reindex(features.index)
    combined_weights = combined_weights.fillna(combined_weights.mean())
    
    print("\nWeight alignment check:")
    print("Features shape:", features.shape)
    print("Combined weights shape:", combined_weights.shape)
    print("Avg uniqueness shape:", avg_unique.shape if avg_unique is not None else "None")
    
    return combined_weights, avg_unique

def _label(features: pd.DataFrame) -> Tuple[pd.DataFrame, Any, pd.Series]:
    
    fs = FeatureStorage('triple_barrier_events.parquet')
    
    if REGENERATE_CACHE:
        vertical_barrier = add_vertical_barrier(
            features.index,
            features['close'],
            num_days = 7 # expariation limit
        )
        
        # volatility = intraday_volatility(features['close'], lookback = 60)
        volatility = daily_volatility_intraday(features['close'], lookback = 60)
        
        print("Volatility stats:")
        print(volatility.describe())
        print("\nNumber of non-null values:", volatility.count())
        
        t_events = features.index[2:]
        min_ret = 0.004
        
        # Before calling get_events, add these checks
        print("Number of events after filtering:", len(volatility[volatility > min_ret]))
        print("Sample of target values:", volatility[volatility > min_ret].head())
        print("T_events length:", len(t_events))
        
        print(f"Beginning triple_barrier_events at time {datetime.now()}")

        triple_barrier_events = get_events(
            close = features['close'],
            t_events = t_events,
            pt_sl = [2, 1], # profit taking 2, stopping loss 1
            target = volatility, # dynamic threshold
            min_ret = min_ret, # minimum position return
            num_threads = 1, # number of multi-thread 
            vertical_barrier_times = vertical_barrier, # add vertical barrier
            side_prediction = None # betting side prediction (primary model)
        )
        
        triple_barrier_events.dropna(inplace=True)
        
        print("triple_barrier_events")
        print(len(triple_barrier_events))
        print(triple_barrier_events.head())
        print(triple_barrier_events.tail())
        
        print(f"Beginning metaLabeling at time {datetime.now()}")
    else:
        triple_barrier_events = fs.load_existing_features()[0]
    
    labels = meta_labeling(triple_barrier_events=triple_barrier_events, close=features['close'])
    
    print(f"Beginning metaLabelling with side labels at time {datetime.now()}")
    
    triple_barrier_events['side'] = labels['bin']
    meta_labels = meta_labeling(
        triple_barrier_events, # with side labels
        features['close']
    )
    
    features['side'] = triple_barrier_events['side'].copy()
    features['label'] = meta_labels['bin'].copy()
    
    features.drop(['open','high','low','close','volume', 'date_time'], axis = 1, inplace = True)
    
    features.dropna(inplace = True)
    matrix = features[features['side'] != 0]
    
    X = matrix.drop(['side','label'], axis = 1)
    y = matrix['label']
    
    print("Features Matrix")
    print(X)
    
    print("Labels matrix")
    print(y)

    fs.save_features(triple_barrier_events)
    
    # Features, When barrier events were started and hit, labels
    # Features and labels are needed as model inputs
    # barrier events are needed for embargo-ing and purging
    return X, triple_barrier_events, y

def _analyze_features(triple_barrier_events: pd.DataFrame, X: pd.DataFrame, y: pd.Series, avg_uniqueness: pd.DataFrame, combined_weights: pd.Series) -> Tuple[BaggingClassifier, dict]:
    
    print("\nShape check before model fitting:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("combined_weights shape:", combined_weights.shape)
    
    # Update avg_uniqueness value for max_samples
    if avg_uniqueness is not None:
        max_samples = avg_uniqueness.loc[X.index].mean()[0]
    else:
        max_samples = 1.0  # fallback value
    
    random_forest_weak = RandomForestClassifier(
        n_estimators = 1,
        criterion = 'entropy',
        bootstrap = False,
        class_weight = 'balanced_subsample',
        min_weight_fraction_leaf = 0.05,
        max_features = min(3, X.shape[1]),
        random_state = 42
    )
    forest_bagging = BaggingClassifier(
        base_estimator = random_forest_weak,
        n_estimators = 1000,
        max_samples = max_samples,
        max_features = .5,
        random_state = 42
    )

    # Extract the t1 series from triple_barrier_events
    # Make sure we only include events for our feature set
    samples_info_sets = triple_barrier_events.loc[X.index, 't1']

    comb_purge_fold = CombinatorialPurgedKFold(
        n_splits = 5,
        n_test_splits = 2,
        samples_info_sets = samples_info_sets,
        pct_embargo = 0.01
    )

    feature_analyzer = FeatureAnalysis(X=X, y=y, model=forest_bagging, cv_folds=comb_purge_fold, combined_weights=combined_weights)
    # feature_importance = feature_analyzer.analyze_feature_importance()
    feature_analyzer.analyze_shap()

    # return feature_importance

def _train_model(triple_barrier_events: pd.DataFrame, X: pd.DataFrame, y: pd.Series, avg_uniqueness: pd.Series, combined_weights: pd.Series) -> Tuple[BaggingClassifier, dict]:
    
    # 1. Handle NaN and infinite values
    # First, let's see where the problematic values are
    print("\nChecking for NaN values in features:")
    print(X.isna().sum()[X.isna().sum() > 0])
    
    print("\nChecking for infinite values in features:")
    print(np.isinf(X.select_dtypes(include=np.number)).sum()[np.isinf(X.select_dtypes(include=np.number)).sum() > 0])
    
    X_train, X_test = _split_data(X, "2023-01-01")
    y_train, y_test = _split_data(y, "2023-01-01")
    
    samples_info_sets = triple_barrier_events.loc[X_train.index].loc[:'2023-01-01', 't1']

    comb_purge_fold = CombinatorialPurgedKFold(
        n_splits = 5,
        n_test_splits = 2,
        samples_info_sets = samples_info_sets,
        pct_embargo = 0.01
    )
    
    # Check sizes at each step
    print("Original X_train shape:", X_train.shape)
    print("Triple barrier event shape:", triple_barrier_events.shape)
    print("Samples after first filter:", triple_barrier_events.loc[X_train.index].shape)
    print("Samples after date filter:", triple_barrier_events.loc[X_train.index].loc[:'2023-01-01'].shape)
    print("Final samples_info_sets shape:", samples_info_sets.shape)

    # Check date ranges
    print("X_train index range:", X_train.index.min(), "to", X_train.index.max())
    print("Triple barrier dates:", triple_barrier_events.index.min(), "to", triple_barrier_events.index.max())

    # Check for NaN values
    print("NaN values in t1:", triple_barrier_events['t1'].isna().sum())

    for train_indices, test_indices in comb_purge_fold.split(X_train, y_train):
        X_train_valid, X_test_valid = X_train.iloc[train_indices], X_train.iloc[test_indices]
        y_train_valid, y_test_valid = y_train.iloc[train_indices], y_train.iloc[test_indices]

        clf = RandomForestClassifier(random_state = 42)
        clf.fit(X_train_valid, y_train_valid)
        
        y_pred = clf.predict(X_test_valid)
        accuracy = accuracy_score(y_test_valid, y_pred)
        print(f'Accuracy: {accuracy:.4f}')

        random_forest_weak = RandomForestClassifier(
            n_estimators = 1,
            criterion = 'entropy',
            bootstrap = False,
            class_weight = 'balanced_subsample',
            min_weight_fraction_leaf = 0.05,
            max_features = 3,
            random_state = 42
        )
        forest_bagging = BaggingClassifier(
            base_estimator = random_forest_weak,
            n_estimators = 1000,
            max_samples = avg_uniqueness.loc[X_train.index].mean()[0],
            max_features = .5,
            random_state = 42
        )
        forest_bagging_fit = forest_bagging.fit(
            X = X_train,
            y = y_train,
            sample_weight = combined_weights
        )

        y_pred = forest_bagging_fit.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy}')
