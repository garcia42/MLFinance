import numpy as np
from scipy import special

def normal_cdf(z):
    """Normal CDF accurate to 7.5e-8."""
    return 0.5 * (1 + special.erf(z / np.sqrt(2)))

def half_normal_cdf(s):
    """
    Half-normal CDF == CDF of |Sn| / sqrt(n)
    """
    return 2 * normal_cdf(s) - 1

def inverse_normal_cdf(p):
    """Inverse normal CDF accurate to 4.5e-4."""
    return special.ndtri(p)

def anderson_darling_cdf(z):
    """Anderson-Darling CDF approximation."""
    if z < 0.01:
        return 0.0
    if z <= 2.0:
        return 2.0 * np.exp(-1.2337/z) * (1.0 + z/8.0 - 0.04958*z*z/(1.325+z)) / np.sqrt(z)
    if z <= 4.0:
        return 1.0 - 0.6621361 * np.exp(-1.091638*z) - 0.95095 * np.exp(-2.005138*z)
    return 1.0 - 0.4938691 * np.exp(-1.050321*z) - 0.5946335 * np.exp(-1.527198*z)

def t_test_one_sample(x):
    """Student's t-test for one sample."""
    n = len(x)
    mean = np.mean(x)
    std = np.sqrt(np.sum((x - mean)**2) / (n * (n - 1)))
    return mean / (std + 1e-60)

def t_test_two_sample(x1, x2):
    """Student's t-test for two independent samples."""
    n1, n2 = len(x1), len(x2)
    mean1, mean2 = np.mean(x1), np.mean(x2)
    ss1 = np.sum((x1 - mean1)**2)
    ss2 = np.sum((x2 - mean2)**2)
    
    std = np.sqrt((ss1 + ss2) / (n1 + n2 - 2) * (1.0/n1 + 1.0/n2))
    return (mean1 - mean2) / (std + 1e-60)

def anova_1way(x, groups):
    """One-way ANOVA test.
    
    Args:
        x: array of measurements
        groups: array of group IDs (0 to K-1)
        
    Returns:
        F-ratio, proportion of variance accounted for, p-value
    """
    x = np.array(x)
    groups = np.array(groups)
    K = len(np.unique(groups))
    n = len(x)
    
    # Compute means
    grand_mean = np.mean(x)
    group_means = [np.mean(x[groups == k]) for k in range(K)]
    group_counts = [np.sum(groups == k) for k in range(K)]
    
    # Between groups variance
    between = sum(count * (mean - grand_mean)**2 
                 for count, mean in zip(group_counts, group_means))
    between /= K - 1
    
    # Within groups variance
    within = sum((x - group_means[groups[i]])**2 for i in range(n))
    within /= n - K
    
    F = between / (within + 1e-60)
    account = between / (between + within)
    pval = 1.0 - special.fdtr(K-1, n-K, F)
    
    return F, account, pval

def roc_area(pred, target, center=False):
    """Compute area under ROC curve."""
    pred = np.array(pred)
    target = np.array(target)
    
    if center:
        target = target - np.mean(target)
    
    # Sort by predictions
    idx = np.argsort(pred)
    target = target[idx]
    
    win_sum = np.sum(target[target > 0])
    lose_sum = -np.sum(target[target < 0])
    
    if win_sum == 0 or lose_sum == 0:
        return 0.5
        
    win = 0
    roc = 0
    
    for t in target[::-1]:  # Iterate in reverse order
        if t > 0:
            win += t / win_sum
        elif t < 0:
            roc -= win * t / lose_sum
            
    return roc

