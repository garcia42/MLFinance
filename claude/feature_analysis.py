# Standard library modules
from datetime import datetime

# Third-party modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# FinancialMachineLearning modules
from FinancialMachineLearning.cross_validation.combinatorial import CombinatorialPurgedKFold
from FinancialMachineLearning.feature_importance.importance import mean_decrease_impurity, mean_decrease_accuracy, plot_feature_importance, cross_val_score, single_feature_importance, clustered_mean_decrease_accuracy, clustered_mean_decrease_importance_detailed, clustered_mean_decrease_accuracy
from FinancialMachineLearning.feature_importance.orthogonal import get_orthogonal_features
from FinancialMachineLearning.machine_learning.clustering import clusterKMeansBase

# Claude modules
from claude.feature_storage import FeatureStorage
from claude.model import Model

class FeatureAnalysis:
    def __init__(self, model: Model, path="./Data"):
        self.X = model.X_clean
        self.y = model.y_size
        # Extract the t1 series from triple_barrier_events
        # Make sure we only include events for our feature set
        samples_info_sets = model.triple_barrier_events.loc[model.X_clean.index, 't1']

        comb_purge_fold = CombinatorialPurgedKFold(
            n_splits = 5,
            n_test_splits = 2,
            samples_info_sets = samples_info_sets,
            pct_embargo = 0.06
        )
        self.cv_folds = comb_purge_fold
        self.model = model.y_size

        # Add validation and debugging for orthogonal features
        print("\nBefore orthogonal transformation:")
        print("X shape:", model.X_clean.shape)
        print("X NaN count:", model.X_clean.isna().sum().sum())
        print("X inf count:", np.isinf(model.X_clean.values).sum())

        self.ortho_features = get_orthogonal_features(model.X_clean, variance_thresh=0.95)
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
        
        fs = FeatureStorage(path + "/ortho.parquet")
        fs.save_features(self.ortho_features)
        
        self.fit_model = self.model.fit(
            X = self.ortho_features,
            y = model.y_size,
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
        plt.show()
        plt.savefig('shap')
    
    def mdi(self, path="./Data/images"):
        mdi_results = mean_decrease_impurity(
            model=self.fit_model,
            feature_names=self.ortho_features.columns
        )
        plot_feature_importance(
            mdi_results,
            oob_score=self.fit_model.oob_score_,
            oos_score=self.oos_score,
            save_fig=True,
            output_path=f'{path}/Ortho_MDI_feature_importance.png'
        )
    def mda(self, path="./Data/images", scoring=accuracy_score):
        # MDA Analysis
        mda_results = mean_decrease_accuracy(
            model=self.fit_model,
            X=self.ortho_features,
            y=self.y,
            cv_gen=self.cv_folds,
            scoring = scoring # optimizing to accuracy score
        )
        plot_feature_importance(
            mda_results,
            oob_score=self.fit_model.oob_score_,
            oos_score=self.oos_score,
            save_fig=True,
            output_path=f'{path}/Ortho_MDA_feature_importance.png',
        )
        return mda_results
    
    def sfi(self, path="./Data/images"):
        # SFI Analysis
        sfi_results = single_feature_importance(
            clf=self.fit_model,
            X=self.ortho_features,
            y=self.y,
            cv_gen=self.cv_folds,
            scoring=accuracy_score
        )
        plot_feature_importance(
            sfi_results,
            oob_score=self.fit_model.oob_score_,
            oos_score=self.oos_score,
            save_fig=True,
            output_path=f'{path}/Ortho_SFI_feature_importance.png'
        )

    def analyze_feature_importance(self):
        print (f"Beginning analyze_feature_importance at time {datetime.now()}")
        
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