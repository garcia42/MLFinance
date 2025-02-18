import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class FeatureIssue:
    feature_name: str
    issue_type: str
    details: str
    severity: str  # 'high', 'medium', 'low'
    suggested_action: str

class FeatureAnalyzer:
    """
    A utility class to analyze feature matrices for common problems
    that could lead to overfitting in machine learning models.
    """
    def __init__(self, X, y=None, feature_names=None):
        self.X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=feature_names)
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.issues: List[FeatureIssue] = []
        
    def basic_stats(self):
        """Returns basic statistics about the feature matrix and identifies problematic features."""
        stats = {
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'samples_to_features_ratio': self.n_samples / self.n_features
        }
        
        # Check missing values by feature
        missing_by_feature = self.X.isnull().sum()
        for feature, count in missing_by_feature[missing_by_feature > 0].items():
            severity = 'high' if count/self.n_samples > 0.1 else 'medium'
            self.issues.append(FeatureIssue(
                feature_name=feature,
                issue_type='missing_values',
                details=f'{count} missing values ({count/self.n_samples:.1%} of samples)',
                severity=severity,
                suggested_action='Consider imputation or feature removal if >10% missing'
            ))

        # Check infinite values by feature
        inf_by_feature = np.isinf(self.X.select_dtypes(include=np.number)).sum()
        for feature, count in inf_by_feature[inf_by_feature > 0].items():
            self.issues.append(FeatureIssue(
                feature_name=feature,
                issue_type='infinite_values',
                details=f'{count} infinite values detected',
                severity='high',
                suggested_action='Replace infinities with NaN or remove feature'
            ))

        # Check constant or near-constant features
        for feature in self.X.columns:
            unique_ratio = len(self.X[feature].unique()) / self.n_samples
            if unique_ratio < 0.01:
                self.issues.append(FeatureIssue(
                    feature_name=feature,
                    issue_type='low_variance',
                    details=f'Only {len(self.X[feature].unique())} unique values in {self.n_samples} samples',
                    severity='high',
                    suggested_action='Consider removing this near-constant feature'
                ))

        return stats
    
    def check_correlations(self, threshold=0.8):
        """Identifies highly correlated feature pairs."""
        corr_matrix = self.X.corr()
        
        # Track features that appear in multiple correlations
        correlation_counts = {}
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                correlation = abs(corr_matrix.iloc[i, j])
                if correlation > threshold:
                    feat1, feat2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    
                    # Update correlation counts
                    correlation_counts[feat1] = correlation_counts.get(feat1, 0) + 1
                    correlation_counts[feat2] = correlation_counts.get(feat2, 0) + 1
                    
                    severity = 'high' if correlation > 0.95 else 'medium'
                    self.issues.append(FeatureIssue(
                        feature_name=f'{feat1}, {feat2}',
                        issue_type='high_correlation',
                        details=f'Correlation coefficient: {correlation:.3f}',
                        severity=severity,
                        suggested_action='Consider removing one of these features or combining them'
                    ))
        
        # Identify features with multiple correlations
        for feature, count in correlation_counts.items():
            if count > 2:
                self.issues.append(FeatureIssue(
                    feature_name=feature,
                    issue_type='multiple_correlations',
                    details=f'Correlated with {count} other features',
                    severity='high',
                    suggested_action='This feature is a prime candidate for removal'
                ))
    
    def check_outliers(self, zscore_threshold=3):
        """Identifies features with significant outliers using z-score method."""
        for column in self.X.select_dtypes(include=np.number).columns:
            z_scores = np.abs(stats.zscore(self.X[column]))
            outlier_count = np.sum(z_scores > zscore_threshold)
            outlier_percentage = outlier_count / self.n_samples
            
            if outlier_count > 0:
                severity = 'high' if outlier_percentage > 0.05 else 'medium'
                self.issues.append(FeatureIssue(
                    feature_name=column,
                    issue_type='outliers',
                    details=f'{outlier_count} outliers ({outlier_percentage:.1%} of samples)',
                    severity=severity,
                    suggested_action='Consider outlier removal, capping, or robust scaling'
                ))
    
    def check_feature_importance(self, importance_threshold=0.01):
        """Analyzes feature importance and identifies low-importance features."""
        if self.y is None:
            return None
            
        importance_scores = mutual_info_classif(self.X, self.y)
        importance_df = pd.DataFrame({
            'feature': self.X.columns,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        # Identify low-importance features
        for _, row in importance_df[importance_df['importance'] < importance_threshold].iterrows():
            self.issues.append(FeatureIssue(
                feature_name=row['feature'],
                issue_type='low_importance',
                details=f'Feature importance score: {row["importance"]:.4f}',
                severity='medium',
                suggested_action='Consider removing this low-importance feature'
            ))
        
        return importance_df
    
    def analyze_and_report(self):
        """Runs all analyses and generates a comprehensive report."""
        print("\n=== Feature Matrix Analysis Report ===")
        
        # Run all checks
        stats = self.basic_stats()
        self.check_correlations()
        self.check_outliers()
        self.check_feature_importance()
        
        # Print basic dataset information
        print(f"\nDataset Overview:")
        print(f"- Samples: {stats['n_samples']}")
        print(f"- Features: {stats['n_features']}")
        print(f"- Samples-to-features ratio: {stats['samples_to_features_ratio']:.2f}")
        
        if stats['samples_to_features_ratio'] < 10:
            print("\nWARNING: Low samples-to-features ratio may lead to overfitting")
        
        # Group and print issues by severity
        if self.issues:
            print("\nIdentified Issues:")
            for severity in ['high', 'medium', 'low']:
                severity_issues = [issue for issue in self.issues if issue.severity == severity]
                if severity_issues:
                    print(f"\n{severity.upper()} Severity Issues:")
                    for issue in severity_issues:
                        print(f"\nFeature(s): {issue.feature_name}")
                        print(f"Issue Type: {issue.issue_type}")
                        print(f"Details: {issue.details}")
                        print(f"Suggested Action: {issue.suggested_action}")
        else:
            print("\nNo significant issues found in the feature matrix.")
            
        # Provide summary of most problematic features
        problem_features = {}
        for issue in self.issues:
            for feature in issue.feature_name.split(', '):
                problem_features[feature] = problem_features.get(feature, 0) + 1
                
        if problem_features:
            print("\nMost Problematic Features:")
            for feature, issue_count in sorted(problem_features.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"- {feature}: {issue_count} issues identified")

# Example usage:
if __name__ == "__main__":
    # Generate sample data with some intentional issues
    np.random.seed(42)
    n_samples = 1000
    
    # Create feature matrix with some problems
    X = pd.DataFrame({
        'good_feature': np.random.randn(n_samples),
        'constant_feature': np.ones(n_samples),
        'highly_correlated_1': np.random.randn(n_samples),
        'highly_correlated_2': np.random.randn(n_samples) * 1.1 + 0.1,  # Correlated with highly_correlated_1
        'outlier_feature': np.random.randn(n_samples),
    })
    
    # Add some outliers
    X.loc[0:10, 'outlier_feature'] = 100
    
    # Add some missing values
    X.loc[0:50, 'good_feature'] = np.nan
    
    # Create binary target
    y = np.random.randint(0, 2, n_samples)
    
    # Run analysis
    analyzer = FeatureAnalyzer(X, y)
    analyzer.analyze_and_report()