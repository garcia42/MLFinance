import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from FinancialMachineLearning.cross_validation.cross_validation import cross_val_score
from tqdm import tqdm
from typing import Dict, List, Optional, Callable, Union
from sklearn.base import BaseEstimator

def mean_decrease_impurity(model, feature_names):
    feature_imp_df = {i: tree.feature_importances_ for i, tree in enumerate(model.estimators_)}
    feature_imp_df = pd.DataFrame.from_dict(feature_imp_df, orient='index')
    feature_imp_df.columns = feature_names
    feature_imp_df = feature_imp_df.replace(0, np.nan)

    importance = pd.concat({'mean': feature_imp_df.mean(),
                            'std': feature_imp_df.std() * feature_imp_df.shape[0] ** -0.5},
                           axis=1)
    importance /= importance['mean'].sum()
    return importance

def mean_decrease_impurity_modified(model, feature_names): 
    # Get feature importances from each tree
    feature_importances = []
    for tree in model.estimators_:
        feature_importances.append(tree.feature_importances_)
    
    # Create DataFrame with proper shape
    feature_imp_df = pd.DataFrame(feature_importances, columns=feature_names)
    feature_imp_df = feature_imp_df.replace(0, np.nan)

    # Calculate mean and std
    importance = pd.concat({
        'mean': feature_imp_df.mean(),
        'std': feature_imp_df.std() * feature_imp_df.shape[0] ** -0.5
    }, axis=1)
    
    # Normalize
    importance /= importance['mean'].sum()
    return importance

def mean_decrease_accuracy(model, X, y, cv_gen, sample_weight=None, scoring=log_loss) -> pd.DataFrame:
    if sample_weight is None:
        sample_weight = np.ones((X.shape[0],))
    
    # Log input shapes and basic info
    print(f"Input shapes - X: {X.shape}, y: {y.shape}, sample_weight: {sample_weight.shape}")
    print(f"Unique y values: {np.unique(y)}")
    print(f"Scoring function: {scoring.__name__}")
    
    fold_metrics_values, features_metrics_values = pd.Series(dtype=float), pd.DataFrame(columns=X.columns)
    
    for i, (train, test) in tqdm(enumerate(cv_gen.split(X=X))):
        print(f"\nFold {i} - Train size: {len(train)}, Test size: {len(test)}")
        
        fit = model.fit(X=X.iloc[train, :], y=y.iloc[train], sample_weight=sample_weight[train])
        pred = fit.predict(X.iloc[test, :])
        
        if scoring == log_loss:
            prob = fit.predict_proba(X.iloc[test, :])
            fold_score = -scoring(y.iloc[test], prob, sample_weight=sample_weight[test], labels=model.classes_)
            print(f"Fold {i} base log_loss: {-fold_score:.4f}")
        else:
            fold_score = scoring(y.iloc[test], pred, sample_weight=sample_weight[test])
            print(f"Fold {i} base score: {fold_score:.4f}")
            
        fold_metrics_values.loc[i] = fold_score
        
        # Log feature permutation scores
        feature_scores = {}
        for j in X.columns:
            X1_ = X.iloc[test, :].copy(deep=True)
            np.random.shuffle(X1_[j].values)
            
            if scoring == log_loss:
                prob = fit.predict_proba(X1_)
                feature_score = -scoring(y.iloc[test], prob, sample_weight=sample_weight[test], labels=model.classes_)
            else:
                pred = fit.predict(X1_)
                feature_score = scoring(y.iloc[test], pred, sample_weight=sample_weight[test])
                
            features_metrics_values.loc[i, j] = feature_score
            feature_scores[j] = feature_score
        
        print(f"Fold {i} feature scores range: {min(feature_scores.values()):.4f} to {max(feature_scores.values()):.4f}")
    
    # Calculate importance
    importance = (-features_metrics_values).add(fold_metrics_values, axis=0)
    print("\nPre-normalization importance stats:")
    print(importance.describe())
    
    if scoring == log_loss:
        # Log denominators before division
        print("\nDenominator (-features_metrics_values) stats:")
        print((-features_metrics_values).describe())
        importance = importance / -features_metrics_values
    else:
        # Log denominators before division
        print("\nDenominator (1.0 - features_metrics_values) stats:")
        print((1.0 - features_metrics_values).describe())
        importance = importance / (1.0 - features_metrics_values)
    
    print("\nPost-normalization importance stats:")
    print(importance.describe())
    
    # Calculate final statistics
    importance = pd.concat({
        'mean': importance.mean(),
        'std': importance.std() * importance.shape[0] ** -.5
    }, axis=1)
    
    print("\nFinal importance values before replacing inf/nan:")
    print(importance)
    
    # Log any inf or nan values being replaced
    inf_mask = np.isinf(importance)
    nan_mask = np.isnan(importance)
    if inf_mask.any().any() or nan_mask.any().any():
        print("\nReplacing the following inf/nan values:")
        print("Inf values at:", np.where(inf_mask))
        print("NaN values at:", np.where(nan_mask))
    
    importance.replace([-np.inf, np.nan], 0, inplace=True)
    return importance

def single_feature_importance(clf, X, y, cv_gen, sample_weight = None, scoring = log_loss):
    feature_names = X.columns
    if sample_weight is None:
        sample_weight = np.ones((X.shape[0],))

    imp = pd.DataFrame(columns=['mean', 'std'])
    for feat in tqdm(feature_names) :
        feat_cross_val_scores = cross_val_score(
            clf,
            X=X[[feat]],
            y=y,
            sample_weight=sample_weight,
            scoring=scoring,
            cv_gen=cv_gen
        )
        imp.loc[feat, 'mean'] = feat_cross_val_scores.mean()
        imp.loc[feat, 'std'] = feat_cross_val_scores.std() * feat_cross_val_scores.shape[0] ** -.5
    return imp

def clustered_mean_decrease_importance_detailed(fit, feature_names, clstrs) -> pd.DataFrame:
    """
    Calculate clustered feature importance scores and list features in each cluster
    
    Parameters:
    -----------
    fit : fitted model
        The tree-based model (e.g. RandomForest)
    feature_names : list-like
        Names of features
    clstrs : dict
        Dictionary mapping cluster indices to lists of feature names
    
    Returns:
    --------
    pd.DataFrame with importance scores and feature lists for each cluster
    """
    # Calculate importances as before
    df0 = {
        i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)
    }
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    df0.columns = feature_names
    df0 = df0.replace(0, np.nan)
    
    # Calculate mean and std
    imp = pd.DataFrame(columns=['mean', 'std', 'features'])
    for i, j in clstrs.items():
        df1 = df0[j].sum(axis=1)
        imp.loc['C_' + str(i), 'mean'] = df1.mean()
        imp.loc['C_' + str(i), 'std'] = df1.std() * df1.shape[0] ** (-0.5)
        imp.loc['C_' + str(i), 'features'] = ', '.join(j)  # Add list of features
    
    # Normalize importance scores
    imp[['mean', 'std']] /= imp['mean'].sum()
    
    return imp

def clustered_mean_decrease_accuracy(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    cv_gen: Union[type, object],  # Cross-validator object with split method
    clusters: Dict[int, List[str]],
    sample_weight: Optional[np.ndarray] = None,
    scoring: Callable = log_loss
) -> pd.DataFrame:
    """
    Calculate clustered Mean Decrease Accuracy with feature lists
    
    Parameters:
    -----------
    model : BaseEstimator
        The tree-based model (e.g. RandomForest) that implements fit/predict/predict_proba
    X : pd.DataFrame
        Features used for training
    y : Union[pd.Series, np.ndarray]
        Target variable
    cv_gen : Union[type, object]
        Cross-validation generator that implements split method
    clusters : Dict[int, List[str]]
        Dictionary mapping cluster indices to lists of feature names
    sample_weight : Optional[np.ndarray]
        Sample weights, defaults to None
    scoring : Callable
        Scoring function (default: log_loss)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns:
            - mean: float - Mean importance score
            - std: float - Standard error of importance
            - features: str - Comma-separated list of features in cluster
    """
    if sample_weight is None:
        sample_weight = np.ones((X.shape[0],))
        
    fold_metrics_values: pd.Series = pd.Series(dtype=float)
    features_metrics_values: pd.DataFrame = pd.DataFrame(columns=X.columns)
    
    # Calculate base scores and feature-level MDA
    for i, (train, test) in tqdm(enumerate(cv_gen.split(X=X))):
        # Fit model on training data
        fit = model.fit(X=X.iloc[train, :], y=y.iloc[train], 
                       sample_weight=sample_weight[train])
        
        # Calculate base score for this fold
        pred = fit.predict(X.iloc[test, :])
        if scoring == log_loss:
            prob = fit.predict_proba(X.iloc[test, :])
            fold_metrics_values.loc[i] = -scoring(y.iloc[test], prob, 
                                                sample_weight=sample_weight[test],
                                                labels=model.classes_)
        else:
            fold_metrics_values.loc[i] = scoring(y.iloc[test], pred, 
                                               sample_weight=sample_weight[test])
        
        # For each cluster, shuffle all features in the cluster together
        for cluster_idx, feature_list in clusters.items():
            X1_: pd.DataFrame = X.iloc[test, :].copy(deep=True)
            
            # Generate single permutation for all features in cluster
            perm_idx: np.ndarray = np.random.permutation(len(X1_))
            X1_[feature_list] = X1_[feature_list].iloc[perm_idx].values
            
            # Calculate score with permuted cluster
            if scoring == log_loss:
                prob = fit.predict_proba(X1_)
                cluster_score: float = -scoring(y.iloc[test], prob, 
                                             sample_weight=sample_weight[test],
                                             labels=model.classes_)
            else:
                pred = fit.predict(X1_)
                cluster_score: float = scoring(y.iloc[test], pred,
                                            sample_weight=sample_weight[test])
            
            # Assign same score to all features in cluster
            for feature in feature_list:
                features_metrics_values.loc[i, feature] = cluster_score
    
    # Calculate importance scores
    importance: pd.DataFrame = (-features_metrics_values).add(fold_metrics_values, axis=0)
    if scoring == log_loss:
        importance = importance / -features_metrics_values
    else:
        importance = importance / (1.0 - features_metrics_values)
    
    # Calculate mean and std for clusters with feature lists
    cluster_importance: pd.DataFrame = pd.DataFrame(columns=['mean', 'std', 'features'])
    for cluster_idx, feature_list in clusters.items():
        cluster_scores: pd.Series = importance[feature_list].mean(axis=1)
        cluster_importance.loc[f'C_{cluster_idx}', 'mean'] = float(cluster_scores.mean())
        cluster_importance.loc[f'C_{cluster_idx}', 'std'] = float(cluster_scores.std() * len(cluster_scores)**(-0.5))
        cluster_importance.loc[f'C_{cluster_idx}', 'features'] = ', '.join(feature_list)
    
    # Clean up and normalize
    cluster_importance[['mean', 'std']] = cluster_importance[['mean', 'std']].replace([-np.inf, np.nan], 0)
    cluster_importance[['mean', 'std']] /= cluster_importance['mean'].sum()
    
    return cluster_importance

def plot_feature_importance(importance_df, oob_score, oos_score, save_fig=False, output_path=None):
    plt.figure(figsize=(10, importance_df.shape[0] / 5))
    importance_df.sort_values('mean', ascending=True, inplace=True)
    importance_df['mean'].plot(kind='barh', color='b', alpha=0.25, xerr=importance_df['std'], error_kw={'ecolor': 'r'})
    plt.grid(False)
    plt.axvline(x = 0, color = 'lightgray', ls = '-.', lw = 1, alpha = 0.75)
    plt.title('Feature importance. OOB Score:{}; OOS score:{}'.format(round(oob_score, 4), round(oos_score, 4)))

    plt.savefig(output_path)
    if save_fig is True:
        plt.show()
