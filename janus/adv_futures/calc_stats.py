from scipy.stats import skew
import pandas as pd
import numpy as np
from typing import Union, Tuple, Dict, List

NUM_TRADING_DAYS_ANNUAL = 256

def calculate_trading_profits(
    prices: Union[List[float], np.ndarray, pd.Series],
    positions: Union[List[int], np.ndarray, pd.Series],
    multiplier: float,
    fx_rate: float = 1.0,
    return_cumulative: bool = True
) -> Union[np.ndarray, float]:
    """
    Calculate profits from a trading strategy using the buy and hold methodology.

    Parameters:
    -----------
    prices : array-like
        Series of back-adjusted prices (Pt)
    positions : array-like
        Series of positions (Nt) where positive = long, negative = short, 0 = no position
    multiplier : float
        Futures contract multiplier (converts price points to instrument currency)
    fx_rate : float, default=1.0
        Exchange rate to convert from instrument currency to base currency
    return_cumulative : bool, default=True
        If True, returns cumulative profits. If False, returns period-by-period profits.

    Returns:
    --------
    numpy.ndarray or float
        Cumulative profits in base currency (if return_cumulative=True)
        or period profits (if return_cumulative=False)

    Examples:
    ---------
    # Simple buy and hold example from the document
    >>> prices = [4500, 4545]
    >>> positions = [1, 1]  # Long 1 contract throughout
    >>> multiplier = 5
    >>> fx_rate = 0.75  # £1 = $0.75
    >>> profit = calculate_trading_profits(prices, positions, multiplier, fx_rate)
    >>> print(f"Total profit: ${profit[-1]:.2f}")  # Should be $168.75
    """

    # Convert inputs to numpy arrays
    prices = np.array(prices, dtype=float)
    positions = np.array(positions, dtype=int)

    # Validate inputs
    if len(prices) != len(positions):
        raise ValueError("Prices and positions arrays must have the same length")

    if len(prices) < 2:
        raise ValueError("Need at least 2 price points to calculate returns")

    # Calculate period returns in price points
    # R^points_t = N_{t-1} × (P_t - P_{t-1})
    price_changes = np.diff(prices)  # P_t - P_{t-1}
    lagged_positions = positions[:-1]  # N_{t-1}

    period_returns_points = lagged_positions * price_changes

    # Convert to instrument currency
    # R^instr_t = R^points_t × Multiplier
    period_returns_instr = period_returns_points * multiplier

    # Convert to base currency
    # R^base_t = R^instr_t × FX rate
    period_returns_base = period_returns_instr * fx_rate

    if return_cumulative:
        # Calculate cumulative returns
        cumulative_returns = np.cumsum(period_returns_base)
        # Prepend 0 for the initial period (no profit/loss at start)
        return np.concatenate([[0], cumulative_returns])
    else:
        return period_returns_base

def annual_std_dev(returns):
    daily_std = np.std(returns, ddof=1)
    return daily_std * np.sqrt(NUM_TRADING_DAYS_ANNUAL)

def average_drawdown(returns):
    """
    Calculate the average drawdown from a series of returns.

    Average drawdown measures the mean of all daily drawdown values,
    providing a more robust risk measure than maximum drawdown alone.

    Parameters:
    -----------
    returns : array-like
        Series of returns (can be daily, monthly, etc.)
        Can be a list, numpy array, or pandas Series

    Returns:
    --------
    float
        Average drawdown as a decimal (negative value)

    Example:
    --------
    >>> returns = [0.01, -0.02, 0.015, -0.01, 0.005]
    >>> avg_dd = average_drawdown(returns)
    >>> print(f"Average drawdown: {avg_dd:.4f}")
    """

    # Convert to numpy array for easier handling
    returns = np.array(returns)

    # Calculate cumulative returns (wealth index)
    cumulative_returns = np.cumprod(1 + returns)

    # Calculate running maximum (peak values)
    running_max = np.maximum.accumulate(cumulative_returns)

    # Calculate drawdown at each point
    # Drawdown = (Current Value - Peak Value) / Peak Value
    drawdowns = (cumulative_returns - running_max) / running_max

    # Return the average of all drawdown values
    return np.mean(drawdowns)

def mean_annual_return(avg_daily_ret: float):
    return avg_daily_ret * NUM_TRADING_DAYS_ANNUAL

def sharpe_ratio(mean_excess_return: float, std_dev: float):
    return mean_excess_return / std_dev

def measure_fat_tails(daily_returns: Union[list, np.ndarray, pd.Series]) -> Dict[str, float]:
    """
    Measure fat tails in a daily return series using percentile ratios.

    This function implements the method described in the text, which uses
    percentile ratios to measure how extreme the tails are compared to a
    normal Gaussian distribution. The returns are automatically demeaned.

    Parameters:
    -----------
    daily_returns : array-like
        Series of daily returns (as decimals, e.g., 0.02 for 2%)

    Returns:
    --------
    dict : Dictionary containing:
        - 'lower_percentile_ratio': 1st percentile / 30th percentile ratio
        - 'upper_percentile_ratio': 99th percentile / 70th percentile ratio
        - 'lower_tail': Relative lower fat tail ratio (compared to normal)
        - 'upper_tail': Relative upper fat tail ratio (compared to normal)
        - 'percentiles': Dictionary of calculated percentiles
    """

    # Convert to numpy array for easier handling
    daily_returns = np.array(daily_returns)

    # Remove any NaN values
    daily_returns = daily_returns[~np.isnan(daily_returns)]

    if len(daily_returns) == 0:
        raise ValueError("No valid daily returns data provided")

    # Always demean the daily returns as described in the text
    demeaned_returns = daily_returns - np.mean(daily_returns)

    # Calculate required percentiles
    percentiles = {
        '1st': np.percentile(demeaned_returns, 1),
        '30th': np.percentile(demeaned_returns, 30),
        '70th': np.percentile(demeaned_returns, 70),
        '99th': np.percentile(demeaned_returns, 99)
    }

    # Calculate lower percentile ratio (1st / 30th)
    # Note: We use absolute values to handle negative returns properly
    lower_percentile_ratio = abs(percentiles['1st']) / abs(percentiles['30th'])

    # Calculate upper percentile ratio (99th / 70th)
    upper_percentile_ratio = percentiles['99th'] / percentiles['70th']

    # For a normal Gaussian distribution, both ratios equal 4.43
    NORMAL_RATIO = 4.43

    # Calculate relative fat tail ratios
    lower_tail = lower_percentile_ratio / NORMAL_RATIO
    upper_tail = upper_percentile_ratio / NORMAL_RATIO

    return {
        'lower_percentile_ratio': lower_percentile_ratio,
        'upper_percentile_ratio': upper_percentile_ratio,
        'lower_tail': lower_tail,
        'upper_tail': upper_tail,
        'percentiles': percentiles
    }

def calculate_daily_volatility_forecast(returns: Union[np.ndarray, pd.Series],
                                short_span: int = 32,
                                long_span: int = 256,  # NUM_TRADING_DAYS_ANNUAL
                                short_weight: float = 0.7,
                                long_weight: float = 0.3) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate volatility forecast using exponentially weighted moving averages
    and blended short/long term estimates.

    Parameters:
    -----------
    returns : array-like
        Series of returns (can be price returns or percentage returns)
    short_span : int, default 32
        Span for short-term EWMA (corresponds to ~22 day SMA with 11 day half-life)
    long_span : int, default 256
        Span for long-term EWMA (for mean reversion component)
    short_weight : float, default 0.7
        Weight for short-term volatility estimate in blend
    long_weight : float, default 0.3
        Weight for long-term volatility estimate in blend

    Returns:
    --------
    tuple: (short_vol, long_vol, blended_vol)
        - short_vol: Short-term volatility estimates
        - long_vol: Long-term volatility estimates
        - blended_vol: Final blended volatility forecast
    """
    # Convert to pandas Series if numpy array
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    # Calculate EWMA means for both short and long term
    ewma_mean_short = returns.ewm(span=short_span).mean()
    ewma_mean_long = returns.ewm(span=long_span).mean()

    # Calculate squared deviations from EWMA means
    sq_dev_short = (returns - ewma_mean_short) ** 2
    sq_dev_long = (returns - ewma_mean_long) ** 2

    # Calculate EWMA variance (using same spans)
    ewma_var_short = sq_dev_short.ewm(span=short_span).mean()
    ewma_var_long = sq_dev_long.ewm(span=long_span).mean()

    # Calculate volatility (standard deviation)
    short_vol = np.sqrt(ewma_var_short)
    long_vol = np.sqrt(ewma_var_long)

    # Calculate blended estimate
    blended_vol = short_weight * short_vol + long_weight * long_vol

    return short_vol, long_vol, blended_vol
