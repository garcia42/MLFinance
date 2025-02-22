import numpy as np
from scipy.stats import norm, f
from typing import List, Tuple, Optional
import pandas as pd
from indicators.Single.legendre import legendre_3
from indicators.Single.inform import MutInf

def atr(use_log: bool, icase: int, length: int, prices: pd.DataFrame) -> float:
    """
    Compute Average True Range (ATR)
    """
    assert icase >= length
    
    if length == 0:
        if use_log:
            return np.log(prices['high'].iloc[icase] / prices['low'].iloc[icase])
        else:
            return prices['high'].iloc[icase] - prices['low'].iloc[icase]
    
    sum_tr = 0.0
    for i in range(icase - length + 1, icase + 1):
        if use_log:
            term = prices['high'].iloc[i] / prices['low'].iloc[i]
            term = max(term, prices['high'].iloc[i] / prices['close'].iloc[i-1])
            term = max(term, prices['close'].iloc[i-1] / prices['low'].iloc[i])
            sum_tr += np.log(term)
        else:
            term = prices['high'].iloc[i] - prices['low'].iloc[i]
            term = max(term, prices['high'].iloc[i] - prices['close'].iloc[i-1])
            term = max(term, prices['close'].iloc[i-1] - prices['low'].iloc[i])
            sum_tr += term
            
    return sum_tr / length

def price_intensity(prices: pd.DataFrame, n_to_smooth: int = 1) -> pd.Series:
    """
    Calculate Price Intensity Indicator
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with 'open', 'high', 'low', 'close', and 'volume' columns
    n_to_smooth : int
        Exponential smoothing period
        
    Returns:
    --------
    pd.Series
        Price Intensity values
    """
    high, low = prices['high'], prices['low']
    close, open_price = prices['close'], prices['open']
    volume = prices['volume']
    
    # Calculate raw intensity
    denom = high - low
    denom = np.where(denom < 1e-60, 1e-60, denom)
    intensity = 100.0 * (2.0 * close - high - low) / denom * volume
    
    # Apply exponential smoothing if requested
    if n_to_smooth > 1:
        alpha = 2.0 / (n_to_smooth + 1.0)
        intensity = intensity.ewm(alpha=alpha, adjust=False).mean()
    
    # Final transformation
    return 100.0 * norm.cdf(0.8 * np.sqrt(n_to_smooth) * intensity) - 50.0

def reactivity(prices: pd.DataFrame, lookback: int, multiplier: int) -> pd.Series:
    """
    Calculate Reactivity Indicator
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with 'high', 'low', 'close', and 'volume' columns
    lookback : int
        Lookback period
    multiplier : int
        Multiplier for exponential smoothing
        
    Returns:
    --------
    pd.Series
        Reactivity values
    """
    alpha = 2.0 / (lookback * multiplier + 1)
    
    # Initialize smoothed values
    smoothed_range = prices['high'].iloc[0] - prices['low'].iloc[0]
    smoothed_volume = prices['volume'].iloc[0]
    
    result = []
    for i in range(len(prices)):
        if i < lookback:
            result.append(0.0)
            continue
            
        # Calculate range
        window = prices.iloc[i-lookback:i+1]
        price_range = window['high'].max() - window['low'].min()
        
        # Update smoothed values
        smoothed_range = alpha * price_range + (1.0 - alpha) * smoothed_range
        smoothed_volume = alpha * prices['volume'].iloc[i] + (1.0 - alpha) * smoothed_volume
        
        # Calculate aspect ratio
        aspect_ratio = price_range / smoothed_range
        if prices['volume'].iloc[i] > 0 and smoothed_volume > 0:
            aspect_ratio /= prices['volume'].iloc[i] / smoothed_volume
        else:
            aspect_ratio = 1.0
            
        # Calculate reactivity
        value = aspect_ratio * (prices['close'].iloc[i] - prices['close'].iloc[i-lookback])
        value /= smoothed_range
        
        result.append(100.0 * norm.cdf(0.6 * value) - 50.0)
        
    return pd.Series(result, index=prices.index)

def variance(use_change: bool, prices: pd.Series, length: int) -> float:
    """
    Compute historical variance of prices or price changes
    
    Parameters:
    -----------
    use_change : bool
        If True, compute variance of price changes, else of prices
    prices : pd.Series
        Price series
    length : int
        Lookback period
        
    Returns:
    --------
    float
        Variance value
    """
    if use_change:
        # Calculate log price changes
        changes = np.log(prices[-length:] / prices[-length-1:-1])
        return np.var(changes)
    else:
        # Calculate log prices
        log_prices = np.log(prices[-length:])
        return np.var(log_prices)

def change_variance_ratio(prices: pd.DataFrame, short_length: int, mult: int = 2) -> pd.Series:
    """
    Calculate Change Variance Ratio indicator
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with 'close' column
    short_length : int
        Short-term lookback period
    mult : int, default=2
        Multiplier for long-term period (long_length = short_length * mult)
        
    Returns:
    --------
    pd.Series
        Change Variance Ratio indicator values
    """
    if mult < 2:
        mult = 2
    long_length = short_length * mult
    
    result = pd.Series(index=prices.index, dtype=float)
    result[:long_length] = 0.0  # Initialize front values
    
    close_prices = prices['close']
    
    # Calculate for each valid point
    for i in range(long_length, len(prices)):
        window = close_prices.iloc[i-long_length:i+1]
        
        # Calculate short and long-term variances
        long_var = variance(True, window, long_length)
        if long_var > 0:
            short_var = variance(True, window[-short_length-1:], short_length)
            ratio = short_var / long_var
            
            # Transform using F distribution CDF
            from scipy.stats import f
            result.iloc[i] = 100.0 * f.cdf(ratio, 4, 4 * mult) - 50.0
        else:
            result.iloc[i] = 0.0
            
    return result

def linear_trend(prices: pd.DataFrame, lookback: int, atr_length: int) -> pd.Series:
    """
    Calculate Linear Trend indicator using Legendre polynomials
    """
    # Get Legendre polynomials
    p1, _, _ = legendre_3(lookback)
    
    result = pd.Series(index=prices.index, dtype=float)
    result[:max(lookback-1, atr_length)] = 0.0  # Initialize front values
    
    log_prices = np.log(prices['close'])
    
    # Calculate for each valid point
    for i in range(max(lookback-1, atr_length), len(prices)):
        # Get price window
        window = log_prices.iloc[i-lookback+1:i+1].values
        price_mean = np.mean(window)
        
        # Calculate dot product (regression coefficient)
        dot_prod = np.sum(window * p1)
        
        # Calculate R-squared
        pred = dot_prod * p1
        total_ss = np.sum((window - price_mean)**2)
        residual_ss = np.sum((window - price_mean - pred)**2)
        rsq = max(0, 1 - residual_ss / (total_ss + 1e-60))
        
        # Get ATR denominator
        k = lookback - 1 if lookback != 2 else 2
        denom = atr(1, i, atr_length, prices) * k  # Modified ATR call
        
        # Calculate final indicator value
        value = dot_prod * 2.0 / (denom + 1e-60)
        value *= rsq  # Degrade indicator if poor fit
        result.iloc[i] = 100.0 * norm.cdf(value) - 50.0
        
    return result

def close_minus_ma(prices: pd.DataFrame, lookback: int, atr_length: int) -> pd.Series:
    """
    Calculate Close Minus Moving Average indicator, normalized by ATR
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with 'open', 'high', 'low', 'close' columns
    lookback : int
        Lookback period for moving average
    atr_length : int
        Lookback period for ATR normalization
        
    Returns:
    --------
    pd.Series
        Normalized Close Minus MA values
    """
    # Calculate log prices
    log_prices = np.log(prices['close'])
    
    # Calculate moving average of log prices
    ma = log_prices.rolling(window=lookback).mean()
    
    result = pd.Series(index=prices.index, dtype=float)
    result[:max(lookback, atr_length)] = 0.0  # Initialize front values
    
    # Calculate for each valid point
    for i in range(max(lookback, atr_length), len(prices)):
        # Calculate ATR-based denominator
        denom = atr(use_log=0, icase=i, length=atr_length, prices=prices) * np.sqrt(lookback + 1.0)
        
        if denom > 0:
            # Calculate difference and normalize
            diff = log_prices.iloc[i] - ma.iloc[i]
            value = diff / denom
            # Transform to oscillator
            result.iloc[i] = 100.0 * norm.cdf(1.0 * value) - 50.0
        else:
            result.iloc[i] = 0.0
            
    return result

def volume_momentum(prices: pd.DataFrame, short_length: int, mult: int = 2) -> pd.Series:
    """
    Calculate Volume Momentum Indicator
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with 'volume' column
    short_length : int
        Short-term lookback period
    mult : int, default=2
        Multiplier for long-term period (long_length = short_length * mult)
        
    Returns:
    --------
    pd.Series
        Volume Momentum values
    """
    if mult < 2:
        mult = 2
    long_length = short_length * mult
    
    # Calculate short and long moving averages of volume
    short_ma = prices['volume'].rolling(window=short_length).mean()
    long_ma = prices['volume'].rolling(window=long_length).mean()
    
    # Calculate the scaling denominator
    denom = np.exp(np.log(float(mult)) / 3.0)
    
    # Calculate momentum
    result = pd.Series(index=prices.index, dtype=float)
    mask = (long_ma > 0) & (short_ma > 0)
    
    # Where both moving averages are positive, calculate the indicator
    result[mask] = np.log(short_ma[mask] / long_ma[mask]) / denom
    result[mask] = 100.0 * norm.cdf(3.0 * result[mask]) - 50.0
    
    # Fill remaining values with 0
    result[~mask] = 0.0
    
    return result

def normalized_obv(prices: pd.DataFrame, lookback: int, delta_length: Optional[int] = None) -> pd.Series:
    """
    Calculate Normalized On Balance Volume (OBV)
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with 'close' and 'volume' columns
    lookback : int
        Lookback period
    delta_length : Optional[int]
        If provided, calculate delta of OBV over this period
        
    Returns:
    --------
    pd.Series
        Normalized OBV values
    """
    result = []
    for i in range(len(prices)):
        if i < lookback:
            result.append(0.0)
            continue
            
        window = prices.iloc[i-lookback:i+1]
        signed_volume = 0.0
        total_volume = 0.0
        
        # Calculate signed and total volume
        for j in range(1, len(window)):
            vol = window['volume'].iloc[j]
            if window['close'].iloc[j] > window['close'].iloc[j-1]:
                signed_volume += vol
            elif window['close'].iloc[j] < window['close'].iloc[j-1]:
                signed_volume -= vol
            total_volume += vol
            
        if total_volume > 0:
            value = signed_volume / total_volume
            value *= np.sqrt(lookback)
            value = 100.0 * norm.cdf(0.6 * value) - 50.0
        else:
            value = 0.0
            
        result.append(value)
        
    obv = pd.Series(result, index=prices.index)
    
    # Calculate delta if requested
    if delta_length is not None and delta_length > 0:
        obv = obv - obv.shift(delta_length)
        
    return obv

def price_change_oscillator(prices: pd.DataFrame, short_length: int, mult: int = 2) -> pd.Series:
    """
    Calculate Price Change Oscillator
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with 'open', 'high', 'low', 'close' columns
    short_length : int
        Short-term lookback period
    mult : int, default=2
        Multiplier for long-term period (long_length = short_length * mult)
        
    Returns:
    --------
    pd.Series
        Price Change Oscillator values
    """
    if mult < 2:
        mult = 2
    long_length = short_length * mult
    
    result = pd.Series(index=prices.index, dtype=float)
    result[:long_length] = 0.0  # Initialize front values
    
    # Calculate log price changes
    close_prices = prices['close']
    log_changes = np.abs(np.log(close_prices / close_prices.shift(1)))
    
    # Calculate for each valid point
    for i in range(long_length, len(prices)):
        # Calculate short and long-term averages
        short_window = log_changes.iloc[i-short_length+1:i+1]
        long_window = log_changes.iloc[i-long_length+1:i+1]
        
        short_mean = short_window.mean()
        long_mean = long_window.mean()
        
        # Calculate scaling denominator
        denom = 0.36 + 1.0 / short_length
        v = np.log(0.5 * mult) / 1.609  # Equals zero for multiplier = 2, 1 for multiplier = 10
        denom += 0.7 * v  # Good when multiplier = 2-10
        
        # Get ATR denominator
        denom *= atr(1, i, long_length, prices=prices)
        
        if denom > 1e-20:
            value = (short_mean - long_mean) / denom
            result.iloc[i] = 100.0 * norm.cdf(4.0 * value) - 50.0
        else:
            result.iloc[i] = 0.0
            
    return result

def ppo_enhanced(prices: pd.DataFrame, short_length: int, long_length: int, n_to_smooth: int = 1) -> pd.Series:
    """
    Calculate enhanced Percentage Price Oscillator (PPO) with normalization
    This differs from TA-Lib's PPO by including normalization and optional smoothing.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with 'close' column
    short_length : int
        Short-term lookback period
    long_length : int
        Long-term lookback period
    n_to_smooth : int, default=1
        Optional smoothing period
        
    Returns:
    --------
    pd.Series
        Enhanced PPO values
    """
    # Calculate exponential moving averages
    short_alpha = 2.0 / (short_length + 1.0)
    long_alpha = 2.0 / (long_length + 1.0)
    
    short_ema = prices['close'].ewm(alpha=short_alpha, adjust=False).mean()
    long_ema = prices['close'].ewm(alpha=long_alpha, adjust=False).mean()
    
    # Calculate raw PPO
    ppo = 100.0 * (short_ema - long_ema) / (long_ema + 1e-15)
    
    # Apply normalization to reduce outliers
    ppo = 100.0 * norm.cdf(0.2 * ppo) - 50.0
    
    # Apply additional smoothing if requested
    if n_to_smooth > 1:
        alpha = 2.0 / (n_to_smooth + 1.0)
        smoothed = np.zeros_like(ppo)
        smoothed[0] = ppo[0]
        for i in range(1, len(ppo)):
            smoothed[i] = alpha * ppo[i] + (1.0 - alpha) * smoothed[i-1]
        ppo = ppo - smoothed
        
    return ppo

def rsi_exact(prices: pd.DataFrame, lookback: int) -> pd.Series:
    """
    Calculate RSI exactly matching the original C implementation
    Key differences from TA-Lib:
    1. Uses 1e-60 for initialization (vs 0 in TA-Lib)
    2. Divides by (lookback-1) in initialization (vs lookback in TA-Lib)
    3. Sets undefined values to 50.0 (vs 0 in TA-Lib)
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with 'close' column
    lookback : int
        Lookback period
        
    Returns:
    --------
    pd.Series
        RSI values
    """
    result = pd.Series(index=prices.index, dtype=float)
    result[:lookback] = 50.0  # Set undefined values to neutral value
    
    # Initial calculation
    upsum = dnsum = 1e-60  # Initialize with small non-zero value
    close = prices['close']
    
    # Initialize sums
    for i in range(1, lookback):
        diff = close.iloc[i] - close.iloc[i-1]
        if diff > 0.0:
            upsum += diff
        else:
            dnsum -= diff
    
    # Divide by (lookback-1) as per C code
    upsum /= (lookback - 1)
    dnsum /= (lookback - 1)
    
    # Main calculation
    for i in range(lookback, len(prices)):
        diff = close.iloc[i] - close.iloc[i-1]
        if diff > 0:
            upsum = ((lookback - 1) * upsum + diff) / lookback
            dnsum *= (lookback - 1.0) / lookback
        else:
            dnsum = ((lookback - 1) * dnsum - diff) / lookback
            upsum *= (lookback - 1.0) / lookback
            
        result.iloc[i] = 100.0 * upsum / (upsum + dnsum)
        
    return result

def macd_enhanced(prices: pd.DataFrame, short_length: int, long_length: int, n_to_smooth: int = 1) -> pd.Series:
    """
    Calculate enhanced MACD with ATR normalization and additional scaling.
    Key differences from TA-Lib:
    1. Uses ATR for normalization
    2. Includes additional scaling based on lookback periods
    3. Returns values compressed to [-50, 50] range
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with 'open', 'high', 'low', 'close' columns
    short_length : int
        Short-term EMA period
    long_length : int
        Long-term EMA period
    n_to_smooth : int, default=1
        Signal line smoothing period
        
    Returns:
    --------
    pd.Series
        Enhanced MACD values
    """
    # Calculate EMAs
    long_alpha = 2.0 / (long_length + 1.0)
    short_alpha = 2.0 / (short_length + 1.0)
    
    close = prices['close']
    long_ema = close.ewm(alpha=long_alpha, adjust=False).mean()
    short_ema = close.ewm(alpha=short_alpha, adjust=False).mean()
    
    result = pd.Series(index=prices.index, dtype=float)
    result[:long_length + n_to_smooth] = 0.0  # Initialize front values
    
    # Main calculation
    for i in range(long_length + n_to_smooth, len(prices)):
        # Calculate scaling denominator based on lookback differences
        diff = 0.5 * (long_length - 1.0)
        diff -= 0.5 * (short_length - 1.0)
        denom = np.sqrt(abs(diff))
        
        # Calculate ATR normalization
        k = long_length + n_to_smooth
        if k > i:
            k = i
        denom *= atr(0, i, k, prices)
        
        # Calculate MACD and scale
        value = (short_ema.iloc[i] - long_ema.iloc[i]) / (denom + 1e-15)
        result.iloc[i] = 100.0 * norm.cdf(1.0 * value) - 50.0
    
    # Apply additional smoothing if requested
    if n_to_smooth > 1:
        alpha = 2.0 / (n_to_smooth + 1.0)
        smoothed = result.ewm(alpha=alpha, adjust=False).mean()
        result = result - smoothed
    
    return result

def stochastic_exact(prices: pd.DataFrame, lookback: int, n_to_smooth: int = 1) -> pd.Series:
    """
    Calculate Stochastic exactly matching the original C implementation
    Key differences from TA-Lib:
    1. Uses specific 1/3 - 2/3 smoothing ratio
    2. Different handling of initial values
    3. Supports raw (n_to_smooth=0), K (n_to_smooth=1), and D (n_to_smooth=2)
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with 'high', 'low', 'close' columns
    lookback : int
        Lookback period (includes current bar)
    n_to_smooth : int, default=1
        0 for raw, 1 for %K, 2 for %D
        
    Returns:
    --------
    pd.Series
        Stochastic values
    """
    result = pd.Series(index=prices.index, dtype=float)
    # Set undefined values to neutral value
    result[:lookback-1] = 50.0
    
    high, low, close = prices['high'], prices['low'], prices['close']
    
    # Variables for smoothing
    sto_0 = sto_1 = sto_2 = 0.0
    
    # Main calculation
    for i in range(lookback-1, len(prices)):
        # Get window of prices
        high_window = high.iloc[i-lookback+1:i+1]
        low_window = low.iloc[i-lookback+1:i+1]
        
        # Calculate raw stochastic
        highest = high_window.max()
        lowest = low_window.min()
        sto_0 = (close.iloc[i] - lowest) / (highest - lowest + 1e-60)
        
        # n_to_smooth will be 0 for raw, 1 for K, 2 for D
        if n_to_smooth == 0:
            result.iloc[i] = 100.0 * sto_0
        else:
            if i == lookback-1:
                sto_1 = sto_0
                result.iloc[i] = 100.0 * sto_0
            else:
                # First smoothing (K)
                sto_1 = 0.33333333 * sto_0 + 0.66666667 * sto_1
                if n_to_smooth == 1:
                    result.iloc[i] = 100.0 * sto_1
                else:
                    # Second smoothing (D)
                    if i == lookback:
                        sto_2 = sto_1
                        result.iloc[i] = 100.0 * sto_1
                    else:
                        sto_2 = 0.33333333 * sto_1 + 0.66666667 * sto_2
                        result.iloc[i] = 100.0 * sto_2
    
    return result

def stochastic_rsi_exact(prices: pd.DataFrame, rsi_lookback: int, stoch_lookback: int, smooth_periods: int = 0) -> pd.Series:
    """
    Calculate Stochastic RSI exactly matching the original C implementation
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with 'close' column
    rsi_lookback : int
        Lookback period for RSI calculation
    stoch_lookback : int
        Lookback period for Stochastic calculation
    smooth_periods : int, default=0
        Number of periods for final exponential smoothing
        
    Returns:
    --------
    pd.Series
        Stochastic RSI values
    """
    # First calculate RSI
    rsi_values = rsi_exact(prices, rsi_lookback)
    rsi_df = pd.DataFrame({'close': rsi_values})
    
    # Calculate stochastic of RSI values
    result = pd.Series(index=prices.index, dtype=float)
    front_bad = rsi_lookback + stoch_lookback - 1
    result[:front_bad] = 50.0  # Set undefined values to neutral value
    
    # Main calculation
    for i in range(front_bad, len(prices)):
        window = rsi_values.iloc[i-stoch_lookback+1:i+1]
        min_val = window.min()
        max_val = window.max()
        result.iloc[i] = 100.0 * (rsi_values.iloc[i] - min_val) / (max_val - min_val + 1e-60)
    
    # Apply smoothing if requested
    if smooth_periods > 1:
        alpha = 2.0 / (smooth_periods + 1.0)
        smoothed = result.iloc[front_bad]
        for i in range(front_bad + 1, len(prices)):
            smoothed = alpha * result.iloc[i] + (1.0 - alpha) * smoothed
            result.iloc[i] = smoothed
    
    return result

def money_flow(df, period=20):
    """
    Calculate Chaikin's Money Flow (CMF).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'high', 'low', 'close', 'volume' columns
    period : int, optional (default=20)
        The lookback period for the moving average
        
    Returns:
    --------
    pandas.Series
        Chaikin's Money Flow values
    """
    
    # Ensure we have all required columns
    required_columns = ['high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    
    # Find first non-zero volume (initialization point)
    first_valid_idx = df.index[df['volume'] > 0][0]
    
    # Calculate Money Flow Multiplier
    # This is: (2*close - high - low)/(high - low)
    money_flow_multiplier = np.where(
        df['high'] > df['low'],
        (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low']),
        0.0
    )
    
    # Calculate Money Flow Volume
    money_flow_volume = money_flow_multiplier * df['volume']
    
    # Initialize the output series with zeros
    cmf = pd.Series(0.0, index=df.index)
    
    # Calculate Chaikin Money Flow
    for i in range(len(df)):
        if i < period - 1:
            continue
            
        # Skip if we're before the first valid volume
        if df.index[i] < first_valid_idx:
            continue
            
        # Get the window for calculations
        window_mfv = money_flow_volume.iloc[i-period+1:i+1]
        window_vol = df['volume'].iloc[i-period+1:i+1]
        
        # Calculate CMF for this window
        sum_mfv = window_mfv.sum()
        sum_vol = window_vol.sum()
        
        if sum_vol > 0:  # Avoid division by zero
            cmf.iloc[i] = sum_mfv / sum_vol
    
    return cmf

def normalize_cmf(cmf, lookback=20):
    """
    Normalize Chaikin's Money Flow to create an oscillator between -50 and 50.
    
    Parameters:
    -----------
    cmf : pandas.Series
        Raw Chaikin's Money Flow values
    lookback : int, optional (default=20)
        Lookback period for normalization
        
    Returns:
    --------
    pandas.Series
        Normalized CMF values ranging from -50 to 50
    """
    # Calculate rolling mean and standard deviation
    rolling_mean = cmf.rolling(window=lookback).mean()
    rolling_std = cmf.rolling(window=lookback).std()
    
    # Normalize the values
    normalized = (cmf - rolling_mean) / (rolling_std + 1e-10)
    
    # Convert to probability using normal CDF and scale to -50 to 50 range
    from scipy.stats import norm
    normalized = 100 * norm.cdf(normalized * 0.8) - 50
    
    return normalized

def aroon_down(df, period=25):
    """
    Calculate Aroon Down indicator.
    
    The Aroon Down indicator measures the number of periods since the lowest price,
    showing the strength of downward trends.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'low' price column
    period : int, optional (default=25)
        Lookback period for calculating Aroon indicators
        
    Returns:
    --------
    pandas.Series
        Aroon Down values ranging from 0 to 100
    """
    # Ensure we have the required column
    if 'low' not in df.columns:
        raise ValueError("DataFrame must contain 'low' column")
    
    # Initialize output series
    aroon_down = pd.Series(index=df.index, dtype=float)
    aroon_down.iloc[0] = 50.0  # Set first value to neutral
    
    # Calculate Aroon Down for each period
    for i in range(1, len(df)):
        if i < period:
            # For periods less than lookback, use available data
            lookback = i
        else:
            lookback = period
            
        # Get the window for calculations
        window = df['low'].iloc[i-lookback:i+1]
        
        # Find the index of the lowest low in the window
        days_since_low = lookback - window.argmin()
        
        # Calculate Aroon Down
        aroon_down.iloc[i] = 100 * (lookback - days_since_low) / lookback
        
    return aroon_down

def aroon_up(df, period=25):
    """
    Calculate Aroon Up indicator.
    
    The Aroon Up indicator measures the number of periods since the highest price,
    showing the strength of upward trends.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'high' price column
    period : int, optional (default=25)
        Lookback period for calculating Aroon indicators
        
    Returns:
    --------
    pandas.Series
        Aroon Up values ranging from 0 to 100
    """
    # Ensure we have the required column
    if 'high' not in df.columns:
        raise ValueError("DataFrame must contain 'high' column")
    
    # Initialize output series
    aroon_up = pd.Series(index=df.index, dtype=float)
    aroon_up.iloc[0] = 50.0  # Set first value to neutral
    
    # Calculate Aroon Up for each period
    for i in range(1, len(df)):
        if i < period:
            # For periods less than lookback, use available data
            lookback = i
        else:
            lookback = period
            
        # Get the window for calculations
        window = df['high'].iloc[i-lookback:i+1]
        
        # Find the index of the highest high in the window
        days_since_high = lookback - window.argmax()
        
        # Calculate Aroon Up
        aroon_up.iloc[i] = 100 * (lookback - days_since_high) / lookback
        
    return aroon_up

def aroon_difference(df, period=25):
    """
    Calculate Aroon Difference (Aroon Up - Aroon Down).
    
    The Aroon Difference oscillates between -100 and 100, providing a measure
    of trend strength and direction.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'high' and 'low' price columns
    period : int, optional (default=25)
        Lookback period for calculating Aroon indicators
        
    Returns:
    --------
    pandas.Series
        Aroon Difference values ranging from -100 to 100
    """
    # Calculate both Aroon Up and Down
    up = aroon_up(df, period)
    down = aroon_down(df, period)
    
    # Calculate the difference
    return up - down

def mutual_information(df: pd.DataFrame, word_length: int, mult: int = 2) -> pd.Series:
    """
    Calculate Mutual Information indicator using the MutInf class.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'close' column
    word_length : int
        Word length for pattern analysis
    mult : int, default=2
        Multiplier for determining number of required data points
        
    Returns:
    --------
    pd.Series
        Mutual Information indicator values scaled between -50 and 50
    """
    # Calculate number of needed data points
    needed = 2 ** word_length  # Count bins
    needed *= mult            # Requires this many points per bin
    needed += 1              # Plus one for computing differences
    
    # Initialize output series
    result = pd.Series(index=df.index, dtype=float)
    result[:needed-1] = 0.0  # Set undefined values to neutral value
    
    # Create MutInf calculator
    mut_inf = MutInf(word_length)
    if not mut_inf.ok:
        return result
        
    # Calculate for each valid point
    close_prices = df['close'].values
    for i in range(needed-1, len(df)):
        # Get window of prices in reverse chronological order
        window = close_prices[i-needed+1:i+1][::-1]
        
        # Calculate mutual information
        value = mut_inf.mut_inf(window, word_length)
        
        # Transform value following C++ implementation
        value = value * mult * np.sqrt(word_length) - 0.12 * word_length - 0.04
        result.iloc[i] = 100.0 * norm.cdf(3.0 * value) - 50.0
        
    return result

def ma_difference(df, param1=10, param2=20, param3=0):
    """
    Calculate Moving Average Difference indicator, matching C++ implementation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'open', 'high', 'low', 'close' columns
    param1 : float, optional (default=10)
        Short-term moving average length
    param2 : float, optional (default=20)
        Long-term moving average length
    param3 : float, optional (default=0)
        Lag for the long-term moving average
    
    Returns:
    --------
    numpy.ndarray
        Moving Average Difference indicator values scaled between -50 and 50
    """
    # Match C++ parameter rounding
    short_length = int(param1 + 0.5)
    long_length = int(param2 + 0.5)
    lag = int(param3 + 0.5)
    
    n = len(df)
    front_bad = long_length + lag
    if front_bad > n:
        front_bad = n
        
    # Convert DataFrame to arrays for C++-like access
    close = df['close'].values
    
    # Initialize output array
    output = np.zeros(n)
    
    # Main calculation loop
    for icase in range(front_bad, n):
        # Calculate long-term moving average
        long_sum = 0.0
        for k in range(icase - long_length + 1, icase + 1):
            long_sum += close[k - lag]
        long_sum /= long_length
        
        # Calculate short-term moving average
        short_sum = 0.0
        for k in range(icase - short_length + 1, icase + 1):
            short_sum += close[k]
        short_sum /= short_length
        
        # Calculate normalizing factor
        diff = 0.5 * (long_length - 1.0) + lag  # Center of long block
        diff -= 0.5 * (short_length - 1.0)      # Minus center of short block
        denom = np.sqrt(abs(diff))              # For random walk variance
        
        # Calculate ATR and apply to denominator
        denom *= atr(use_log=0, icase=icase, length=long_length + lag, prices=df)
        
        # Calculate final output with epsilon for numerical stability
        output[icase] = (short_sum - long_sum) / (denom + 1e-60)
        output[icase] = 100.0 * norm.cdf(1.5 * output[icase]) - 50.0
        
    return output