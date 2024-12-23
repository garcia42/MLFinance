# Standard library modules
from datetime import datetime
from typing import Any, Tuple

# Third-party modules
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.preprocessing import RobustScaler

# FinancialMachineLearning modules
from FinancialMachineLearning.cross_validation.combinatorial import CombinatorialPurgedKFold
from FinancialMachineLearning.features.volatility import daily_volatility_intraday
from FinancialMachineLearning.labeling.labeling import meta_labeling, get_events, add_vertical_barrier
from FinancialMachineLearning.sample_weights.attribution import *
from FinancialMachineLearning.sample_weights.bootstrapping import *
from FinancialMachineLearning.sample_weights.concurrency import *
from FinancialMachineLearning.utils.multiprocess import *

# Claude modules
from claude.feature_analysis import FeatureAnalysis
from claude.feature_storage import FeatureStorage


REGENERATE_CACHE=True


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

def build_model(features: pd.DataFrame, use_cache = REGENERATE_CACHE, path='./Data') -> Model:
    tfs = FeatureStorage(path + '/tbe_before_analyze.parquet')
    fs = FeatureStorage(path + '/X_before_analyze.parquet')
    fsyd = FeatureStorage(path + '/y_side_before_analyze.parquet')
    fsyz = FeatureStorage(path + '/y_size_before_analyze.parquet')
    fsw = FeatureStorage(path + '/weights_before_analyze.parquet')
    fsu = FeatureStorage(path + '/uniq_before_analysis.parquet')
    
    if not use_cache:
        X, triple_barrier_events, y_side, y_size = _label(features=features.copy(), path=path)
        combined_weights, avg_uniq = _weights(features=features, triple_barrier_events=triple_barrier_events)
        X_clean, y_side, y_size, combined_weights = _clean_features(X_df=X, combined_weights=combined_weights, y_side=y_side, y_size=y_size)
        
        fsu.save_features(avg_uniq)
        fsw.save_features(pd.DataFrame({combined_weights.name: combined_weights}))
        fsyd.save_features(pd.DataFrame({y_side.name: y_side}))
        fsyz.save_features(pd.DataFrame({y_size.name: y_size}))
        fs.save_features(X_clean)
        tfs.save_features(triple_barrier_events)

    # Analyze feature importance for this model
    triple_barrier_events = tfs.load_existing_features()[0]
    X_clean = fs.load_existing_features()[0]
    y_side = fsyd.load_existing_features()[0].squeeze()
    y_size = fsyz.load_existing_features()[0].squeeze()
    combined_weights = fsw.load_existing_features()[0].squeeze()
    avg_uniq = fsu.load_existing_features()[0]
    
    fit_side_model, side_model = _train_model(triple_barrier_events=triple_barrier_events, X=X_clean, y=y_side, combined_weights=combined_weights, avg_uniqueness=avg_uniq)
    fit_size_model, size_model = _train_model(triple_barrier_events=triple_barrier_events, X=X_clean, y=y_size, combined_weights=combined_weights, avg_uniqueness=avg_uniq)
    return Model(triple_barrier_events=triple_barrier_events, X_clean=X_clean, y_side=y_side, y_size=y_size, combined_weights=combined_weights, avg_uniq=avg_uniq, fit_size_model=fit_size_model, size_model=size_model, fit_side_model=fit_side_model, side_model=side_model)

def label_and_analyze(features: pd.DataFrame): # Don't hold out any data, put all data into the classifier and K folds
    model = build_model(features=features)
    _analyze_features(model)

# Data cleaning and validation
def _clean_features(X_df: pd.DataFrame, combined_weights: pd.Series, y_side: pd.DataFrame, y_size: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:

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
    

    if y_side is not None and combined_weights is not None:
        # Get common valid indices
        valid_idx = X_df.index.intersection(y_side.index).intersection(combined_weights.index)
        return X_df.loc[valid_idx], y_side.loc[valid_idx], y_size.loc[valid_idx], combined_weights.loc[valid_idx]
    
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
    
    # Create a Series with the same index as features
    combined_weights = pd.Series(combined_weights, index=return_weights.index)
    
    # Align weights with features index
    # This is crucial - we only want weights for samples that exist in our feature matrix
    combined_weights = combined_weights.reindex(features.index)
    combined_weights = combined_weights.fillna(combined_weights.mean())
    avg_unique = avg_unique.reindex(features.index)
    avg_unique = avg_unique.fillna(avg_unique.mean())
    
    print("\nWeight alignment check:")
    print("Features shape:", features.shape)
    print("Combined weights shape:", combined_weights.shape)
    print("Avg uniqueness shape:", avg_unique.shape if avg_unique is not None else "None")
    
    return combined_weights, avg_unique

def _label(features: pd.DataFrame, path="./Data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    fs = FeatureStorage(path + '/triple_barrier_events.parquet')
    
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
    triple_barrier_events['ret'] = labels['ret']  # Save the returns!
    meta_labels = meta_labeling(
        triple_barrier_events, # with side labels
        features['close']
    )
    
    features['side'] = triple_barrier_events['side'].copy()
    features['label'] = meta_labels['bin'].copy()
    
    features.drop(['open','high','low','close','volume', 'date_time', 'Date'], axis = 1, inplace = True)
    
    features.dropna(inplace = True)
    matrix = features[features['side'] != 0]
    
    # First get both labels
    y_side = matrix['side']  # Direction labels
    y_size = matrix['label'] # Size/probability labels

    # Then drop the label columns from features
    X = matrix.drop(['side', 'label'], axis=1)

    
    print("Features Matrix")
    print(X)
    
    print("Side Labels")
    print(y_side)
    
    print("Size Labels")
    print(y_size)

    fs.save_features(triple_barrier_events)
    
    # Return features, events, and both sets of labels
    return X, triple_barrier_events, y_side, y_size

def _analyze_features(model: Model) -> Tuple[BaggingClassifier, dict]:
    # Extract the t1 series from triple_barrier_events
    # Make sure we only include events for our feature set
    samples_info_sets = model.triple_barrier_events.loc[model.X_clean.index, 't1']

    comb_purge_fold = CombinatorialPurgedKFold(
        n_splits = 5,
        n_test_splits = 2,
        samples_info_sets = samples_info_sets,
        pct_embargo = 0.01
    )

    feature_analyzer = FeatureAnalysis(X=model.X, y=model.y_size, cv_folds=comb_purge_fold, combined_weights=model.combined_weights)
    # feature_importance = feature_analyzer.analyze_feature_importance()
    feature_analyzer.analyze_shap()

    # return feature_importance

def _train_model(triple_barrier_events: pd.DataFrame, X: pd.DataFrame, y: pd.Series, avg_uniqueness: pd.Series, combined_weights: pd.Series) -> Tuple[BaggingClassifier, BaggingClassifier]:
    ### Train the size learner
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
        max_samples = avg_uniqueness.loc[X.index].mean()[0],
        max_features = .5,
        random_state = 42
    )
    forest_bagging_fit = forest_bagging.fit(
        X = X,
        y = y,
        sample_weight = combined_weights
    )
    return forest_bagging_fit, forest_bagging
