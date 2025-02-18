import numpy as np
from scipy.stats import norm

def atr(multiplier: int, current_idx: int, atr_length: int, 
        open_prices: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
    """
    Calculate Average True Range (ATR)
    """
    if current_idx < atr_length:
        return 0.0
    
    true_ranges = []
    for i in range(current_idx - atr_length + 1, current_idx + 1):
        high_low = high[i] - low[i]
        high_close = abs(high[i] - close[i-1])
        low_close = abs(low[i] - close[i-1])
        true_range = max(high_low, high_close, low_close)
        true_ranges.append(true_range)
    
    return multiplier * np.mean(true_ranges)

def trend(n: int, lookback: int, atr_length: int,
          open_prices: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    Compute linear trend using Legendre polynomials
    
    Parameters:
    -----------
    n : int
        Number of data points
    lookback : int
        Lookback period for trend calculation
    atr_length : int
        Length for ATR calculation
    open_prices, high, low, close : np.ndarray
        Price data arrays
    
    Returns:
    --------
    np.ndarray
        Trend indicator values
    """
    # Initialize output array
    output = np.zeros(n)
    work = np.zeros(lookback)
    
    # Set front padding
    front_bad = max(lookback - 1, atr_length)
    
    # Compute first-order Legendre coefficients
    x = np.linspace(-1, 1, lookback)
    work = 2.0 * x / (lookback - 1.0) - 1.0
    work /= np.sqrt(np.sum(work ** 2))
    
    # Compute trend for remaining bars
    for icase in range(front_bad, n):
        # Get price window
        prices = np.log(close[icase-lookback+1:icase+1])
        
        # Calculate mean and dot product
        mean = np.mean(prices)
        dot_prod = np.sum(prices * work)
        
        # Calculate denominator using ATR
        k = lookback - 1 if lookback != 2 else 2
        denom = atr(1, icase, atr_length, open_prices, high, low, close) * k
        
        # Calculate initial output
        output[icase] = dot_prod * 2.0 / (denom + 1e-60)
        
        # Calculate R-squared
        pred = dot_prod * work
        diff = prices - mean
        yss = np.sum(diff ** 2)
        rsq = 1.0 - np.sum((diff - pred) ** 2) / (yss + 1e-60)
        rsq = max(0.0, rsq)
        
        # Apply R-squared and normalize
        output[icase] *= rsq
        output[icase] = 100.0 * norm.cdf(output[icase]) - 50.0
        
    return output

def cmma(n: int, lookback: int, atr_length: int,
         open_prices: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    Compute Close Minus Moving Average (CMMA)
    
    Parameters:
    -----------
    n : int
        Number of data points
    lookback : int
        Lookback period for MA calculation
    atr_length : int
        Length for ATR calculation
    open_prices, high, low, close : np.ndarray
        Price data arrays
    
    Returns:
    --------
    np.ndarray
        CMMA indicator values
    """
    # Initialize output array
    output = np.zeros(n)
    
    # Set front padding
    front_bad = max(lookback, atr_length)
    
    # Compute CMMA for remaining bars
    for icase in range(front_bad, n):
        # Calculate moving average
        ma = np.mean(np.log(close[icase-lookback:icase]))
        
        # Calculate denominator
        denom = atr(1, icase, atr_length, open_prices, high, low, close)
        
        if denom > 0.0:
            denom *= np.sqrt(lookback + 1.0)
            output[icase] = (np.log(close[icase]) - ma) / denom
            output[icase] = 100.0 * norm.cdf(1.0 * output[icase]) - 50.0
        else:
            output[icase] = 0.0
            
    return output
