import pandas as pd
import numpy as np
from scipy.stats import weightedtau, kendalltau, spearmanr, pearsonr

def get_eigen_vector(dot_matrix, variance_thresh): 
    eigen_val, eigen_vec = np.linalg.eigh(dot_matrix)
    idx = eigen_val.argsort()[::-1]
    eigen_val, eigen_vec = eigen_val[idx], eigen_vec[:, idx]
    eigen_val = pd.Series(eigen_val, index=['PC_' + str(i + 1) for i in range(eigen_val.shape[0])])
    eigen_vec = pd.DataFrame(eigen_vec, index=dot_matrix.index, columns=eigen_val.index)
    eigen_vec = eigen_vec.loc[:, eigen_val.index]
    cum_var = eigen_val.cumsum() / eigen_val.sum()
    dim = cum_var.values.searchsorted(variance_thresh)
    eigen_val, eigen_vec = eigen_val.iloc[:dim + 1], eigen_vec.iloc[:, :dim + 1]
    return eigen_val, eigen_vec

def _standardize_df(data_frame: pd.DataFrame):
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    return (data_frame - data_frame.mean(axis=0)) / (data_frame.std(axis=0) + epsilon)

def get_orthogonal_features(feature_df, variance_thresh=0.95) -> pd.DataFrame:
    """
    Compute orthogonal features using Principal Component Analysis (PCA).
    
    Parameters:
    feature_df (pd.DataFrame): The input feature DataFrame.
    variance_thresh (float): The minimum percentage of variance to be explained by the selected principal components.
    
    Returns:
    pd.DataFrame: A DataFrame containing the orthogonal features, with the original feature names as columns and the original index preserved.
    """
    # Remove any constant columns first
    non_constant_cols = feature_df.columns[feature_df.std() != 0]
    feature_df = feature_df[non_constant_cols]
    feature_df_standard = _standardize_df(feature_df)
    
    # Create correlation matrix
    dot_matrix = pd.DataFrame(
        np.dot(feature_df_standard.T, feature_df_standard),
        index=feature_df.columns,
        columns=feature_df.columns
    )
    
    # Get eigenvectors
    eigen_val, eigen_vec = get_eigen_vector(dot_matrix, variance_thresh)
    
    # Create PCA features with PC names first
    pca_features = pd.DataFrame(
        np.dot(feature_df_standard, eigen_vec),
        index=feature_df.index,
        columns=eigen_vec.columns
    )
    
    # Transform back to original feature space
    orthogonal_features = pd.DataFrame(
        np.dot(pca_features, eigen_vec.T),
        index=feature_df.index,
        columns=feature_df.columns
    )
    
    return orthogonal_features

def get_pca_rank_weighted_kendall_tau(feature_imp, pca_rank):
    return weightedtau(feature_imp, pca_rank ** -1.0)

def feature_pca_analysis(feature_df, feature_importance, variance_thresh=0.95):
    feature_df_standard = _standardize_df(feature_df)
    dot = pd.DataFrame(np.dot(feature_df_standard.T, feature_df_standard), index=feature_df.columns,
                       columns=feature_df.columns)
    eigen_val, eigen_vec = get_eigen_vector(dot, variance_thresh)
    all_eigen_values = []
    corr_dict = {'Pearson': [], 'Spearman': [], 'Kendall': []}
    for vec in eigen_vec.columns:
        all_eigen_values.extend(abs(eigen_vec[vec].values * eigen_val[vec]))
    repeated_importance_array = np.tile(feature_importance['mean'].values, len(eigen_vec.columns))

    for corr_type, function in zip(corr_dict.keys(), [pearsonr, spearmanr, kendalltau]):
        corr_coef = function(repeated_importance_array, all_eigen_values)
        corr_dict[corr_type] = corr_coef
    feature_pca_rank = (eigen_val * eigen_vec).abs().sum(axis=1).rank(ascending=False)
    corr_dict['Weighted_Kendall_Rank'] = get_pca_rank_weighted_kendall_tau(feature_importance['mean'].values,
                                                                           feature_pca_rank.values)
    return corr_dict
