import numpy as np
from typing import Tuple

def basic_stats(x: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Calculate basic statistics of an array.
    
    Parameters:
    -----------
    x : np.ndarray
        Input array of values
        
    Returns:
    --------
    Tuple[float, float, float, float]
        (mean, minimum, maximum, interquartile range)
    """
    # Calculate mean
    mean = np.mean(x)
    
    # Sort values for min, max, and IQR
    sorted_x = np.sort(x)
    min_val = sorted_x[0]
    max_val = sorted_x[-1]
    
    # Calculate interquartile range
    n = len(x)
    k25 = int(0.25 * (n + 1))
    k75 = n - 1 - k25
    iqr = sorted_x[k75] - sorted_x[k25]
    
    return mean, min_val, max_val, iqr

def entropy(x: np.ndarray) -> float:
    """
    Calculate normalized entropy using adaptive binning based on sample size.
    
    Parameters:
    -----------
    x : np.ndarray
        Input array of values
        
    Returns:
    --------
    float
        Normalized entropy value between 0 and 1
    """
    n = len(x)
    
    # Determine number of bins based on sample size
    if n >= 10000:
        nbins = 20
    elif n >= 1000:
        nbins = 10
    elif n >= 100:
        nbins = 5
    else:
        nbins = 3
        
    # Find min and max
    xmin = np.min(x)
    xmax = np.max(x)
    
    # Calculate bin factor
    factor = (nbins - 0.00000000001) / (xmax - xmin + 1e-60)
    
    # Assign values to bins
    bins = np.minimum(
        (factor * (x - xmin)).astype(int),
        nbins - 1  # Ensure we don't exceed nbins
    )
    
    # Count occurrences in each bin
    counts = np.bincount(bins, minlength=nbins)
    
    # Calculate entropy
    # Only include non-zero counts to avoid log(0)
    p = counts[counts > 0] / n
    entropy_sum = -np.sum(p * np.log(p))
    
    # Normalize by maximum possible entropy (log of number of bins)
    return entropy_sum / np.log(nbins)
