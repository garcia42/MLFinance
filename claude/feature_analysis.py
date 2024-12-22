import sys
import os
import pickle
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from typing import Dict, Tuple, List
from FinancialMachineLearning.cross_validation.combinatorial import CombinatorialPurgedKFold
from FinancialMachineLearning.feature_importance.orthogonal import get_orthogonal_features, get_pca_rank_weighted_kendall_tau
from sklearn.ensemble import BaggingClassifier
from FinancialMachineLearning.feature_importance.importance import mean_decrease_impurity, mean_decrease_accuracy, plot_feature_importance, cross_val_score, single_feature_importance, clustered_mean_decrease_accuracy, clustered_mean_decrease_importance_detailed, clustered_mean_decrease_accuracy
from FinancialMachineLearning.machine_learning.clustering import clusterKMeansBase
import matplotlib.cm
import seaborn as sns
import matplotlib.pyplot as plt
from feature_storage import FeatureStorage

class FeatureAnalysis:
    def __init__(self, X: pd.DataFrame, y: pd.Series, cv_folds: CombinatorialPurgedKFold, combined_weights: pd.Series):
        self.X = X
        self.y = y
        self.cv_folds = cv_folds
        self.model = RandomForestClassifier(
            criterion = 'entropy',
            class_weight = 'balanced_subsample',
            min_weight_fraction_leaf = 0.0,
            random_state = 42,
            n_estimators = 1000,
            max_features = 1,
            oob_score = True,
            n_jobs = 1
        )

        # Add validation and debugging for orthogonal features
        print("\nBefore orthogonal transformation:")
        print("X shape:", X.shape)
        print("X NaN count:", X.isna().sum().sum())
        print("X inf count:", np.isinf(X.values).sum())

        self.ortho_features = get_orthogonal_features(X, variance_thresh=0.95)
        print("\nAfter orthogonal transformation:")
        print("Ortho features shape:", self.ortho_features.shape)
        print("Ortho features NaN count:", np.isnan(self.ortho_features).sum())
        print("Ortho features inf count:", np.isinf(self.ortho_features).sum())
        
        # Final cleanup if needed
        if np.any(np.isnan(self.ortho_features)) or np.any(np.isinf(self.ortho_features)):
            print("Warning: Cleaning remaining NaN/inf values in orthogonal features")
            self.ortho_features = self.ortho_features.fillna(0)  # or use method='ffill'
            self.ortho_features = self.ortho_features.replace([np.inf, -np.inf], 0)
        
        # Calculate OOS score with proper error handling
        try:
            self.oos_score = cross_val_score(
                classifier=self.model,
                X=self.ortho_features,
                y=self.y,
                cv_gen=self.cv_folds,
                scoring=accuracy_score
            ).mean()
        except Exception as e:
            print(f"Error during cross validation: {str(e)}")
            self.oos_score = 0.0
        
        fs = FeatureStorage("./Data/ortho.parquet")
        fs.save_features(self.ortho_features)
        
        self.fit_model = self.model.fit(
            X = self.ortho_features,
            y = y,
        )
        
    def group_mean_std(self, df0, clstrs) -> pd.DataFrame :
        out = pd.DataFrame(columns = ['mean','std'])
        for i, j in clstrs.items() :
            df1 = df0[j].sum(axis = 1)
            out.loc['C_' + str(i), 'mean'] = df1.mean()
            out.loc['C_' + str(i), 'std'] = df1.std() * df1.shape[0] ** (-0.5)
        return out
    
    def analyze_shap(self):
        forest_explain = shap.TreeExplainer(self.fit_model)
        shap_values_train = forest_explain(self.X)
        
        plt.grid(False)
        plt.title('SHAP Value')
        shap.plots.beeswarm(
            shap_values_train[:,:,1],
            max_display = len(shap_values_train.feature_names),
            plot_size = (8, 8)
        )
        plt.savefig('shap')

    def analyze_feature_importance(self):
        print (f"Beginning analyze_feature_importance at time {datetime.now()}")
        importance_results = {}
        # MDI Analysis
        importance_results['mdi'] = mean_decrease_impurity(
            model=self.fit_model,
            feature_names=self.ortho_features.columns
        )
        plot_feature_importance(
            importance_results['mdi'],
            oob_score=self.fit_model.oob_score_,
            oos_score=self.oos_score,
            save_fig=True,
            output_path=f'./Data/images/Ortho_MDI_feature_importance.png'
        )
        
        print(self.y)
        # MDA Analysis
        importance_results['mda'] = mean_decrease_accuracy(
            model=self.fit_model,
            X=self.ortho_features,
            y=self.y,
            cv_gen=self.cv_folds
        )
        plot_feature_importance(
            importance_results['mda'],
            oob_score=self.fit_model.oob_score_,
            oos_score=self.oos_score,
            save_fig=True,
            output_path=f'./Data/images/Ortho_MDA_feature_importance.png'
        )
        
        # SFI Analysis
        importance_results['sfi'] = single_feature_importance(
            clf=self.fit_model,
            X=self.ortho_features,
            y=self.y,
            cv_gen=self.cv_folds,
            scoring=accuracy_score
        )
        plot_feature_importance(
            importance_results['sfi'],
            oob_score=self.fit_model.oob_score_,
            oos_score=self.oos_score,
            save_fig=True,
            output_path=f'./Data/images/Ortho_SFI_feature_importance.png'
        )
        
        # Clustered Mean decrease importance
        corr0, clusters, _ = clusterKMeansBase(self.ortho_features.corr(), maxNumClusters = 10, n_init = 10)
        sns.heatmap(corr0, cmap = 'viridis')
        plt.savefig('./Data/images/correlation_matrix.png')  # PNG format
        plt.close()

        c_mdi = clustered_mean_decrease_importance_detailed(fit=self.fit_model, feature_names=self.ortho_features.columns, clstrs = clusters)
        importance_results['c_mdi'] = c_mdi
        print("Clustered Mean decrease importance")
        print(c_mdi)
        plot_feature_importance(
            importance_results['c_mdi'],
            oob_score=self.fit_model.oob_score_,
            oos_score=self.oos_score,
            save_fig=True,
            output_path=f'./Data/images/Ortho_CMDI_feature_importance.png'
        )
        
        # Clustered MDA
        c_mda = clustered_mean_decrease_accuracy(model=self.fit_model, X=self.ortho_features, y = self.y, cv_gen=self.cv_folds, clusters=clusters)
        importance_results['c_mda'] = c_mda
        print("Clustered Mean decrease accuracy")
        print(c_mda)
        plot_feature_importance(
            importance_results['c_mda'],
            oob_score=self.fit_model.oob_score_,
            oos_score=self.oos_score,
            save_fig=True,
            output_path=f'./Data/images/Ortho_CMDA_feature_importance.png'
        )

        return importance_results