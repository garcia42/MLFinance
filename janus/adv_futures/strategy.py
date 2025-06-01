import pandas as pd
import numpy as np
from enum import Enum

class StrategyType(Enum):
    BUY_AND_HOLD = "BUY_AND_HOLD"
    EWMAC = "EWMAC"
    EWMAC_LONG_SHORT = "EWMAC_LONG_SHORT"
    TREND_FORECAST = "TREND_FORECAST"

def trend_forecast_strategy(price_data, fast_span=16, slow_span=64,
                           price_col='Close', forecast_scalar=1.9,
                           max_forecast=20, return_signals=True):
    """
    Implements a trend forecasting strategy using risk-adjusted EWMAC signals.

    This strategy calculates a scaled forecast based on the EWMAC crossover
    divided by price volatility, then uses the forecast for position sizing.

    Parameters:
    -----------
    price_data : pd.DataFrame or pd.Series
        Price data with datetime index
    fast_span : int, default=16
        Span for fast EWMA
    slow_span : int, default=64
        Span for slow EWMA
    price_col : str, default='Close'
        Column name for prices if DataFrame is passed
    forecast_scalar : float, default=1.9
        Scaling factor to normalize forecast (1.9 for EWMAC(16,64))
    max_forecast : float, default=20
        Maximum absolute forecast value (caps position size)
    return_signals : bool, default=False
        If True, returns additional columns with forecasts and signals

    Returns:
    --------
    pd.DataFrame
        DataFrame with returns and optionally forecasts/signals
    """

    # Handle both Series and DataFrame inputs
    if isinstance(price_data, pd.DataFrame):
        prices = price_data[price_col].copy()
    else:
        prices = price_data.copy()

    # Calculate EWMAs
    ewma_fast = prices.ewm(span=fast_span, adjust=False).mean()
    ewma_slow = prices.ewm(span=slow_span, adjust=False).mean()

    # Calculate EWMAC signal (crossover)
    ewmac_raw = ewma_fast - ewma_slow

    # Calculate daily returns for volatility estimation
    daily_returns = prices.pct_change()

    # Convert to daily price volatility (sigma_p)
    # sigma_p = price * annualized_vol / 16
    return_vol = daily_returns.ewm(span=25).std()  # Use return volatility directly
    raw_forecast = ewmac_raw / return_vol

    # Apply forecast scalar to get scaled forecast
    scaled_forecast = raw_forecast * forecast_scalar

    # Cap forecasts at maximum absolute value
    scaled_forecast = raw_forecast * forecast_scalar
    capped_forecast = np.clip(scaled_forecast, -max_forecast, max_forecast)

    # Convert forecast to position (normalized by max_forecast for position sizing)
    # Position ranges from -1 to +1
    positions = capped_forecast / max_forecast

    # Shift positions to avoid look-ahead bias
    positions = pd.Series(positions, index=prices.index).shift(1)

    # Calculate strategy returns
    strategy_returns = positions * daily_returns

    # Create results DataFrame
    results = pd.DataFrame({
        'Price': prices,
        'Daily_Returns': daily_returns,
        'Strategy_Returns': strategy_returns,
        'Cumulative_Returns': (1 + strategy_returns).cumprod() - 1,
        'Buy_Hold_Returns': (1 + daily_returns).cumprod() - 1
    })
    
    print(f"EWMAC raw: {ewmac_raw.iloc[-1]:.6f}")
    print(f"Raw forecast: {raw_forecast.iloc[-1]:.6f}")
    print(f"After scalar (*1.9): {(scaled_forecast * 1.9).iloc[-1]:.6f}")
    print(f"After capping (±20): {capped_forecast.iloc[-1]:.6f}")
    print(f"EWMAC raw stats: mean={ewmac_raw.mean():.4f}, std={ewmac_raw.std():.4f}")
    print(f"Raw forecast stats: mean={raw_forecast.mean():.4f}, std={raw_forecast.std():.4f}")
    print(f"EWMAC raw mean: {ewmac_raw.mean():.6f}")  # Should be ~0
    print(f"EWMAC raw std: {ewmac_raw.std():.6f}")

    # Add additional columns if requested
    if return_signals:
        results['EWMA_Fast'] = ewma_fast
        results['EWMA_Slow'] = ewma_slow
        results['EWMAC_Raw'] = ewmac_raw
        results['Raw_Forecast'] = raw_forecast
        results['Scaled_Forecast'] = scaled_forecast
        results['Capped_Forecast'] = capped_forecast
        results['Position'] = positions

    return results

def ewmac_strategy(price_data, fast_span=64, slow_span=256,
                   price_col='Close', can_short=False, return_signals=False):
    """
    Implements the EWMAC (Exponentially Weighted Moving Average Crossover) strategy
    as described in systematic trading literature.

    Parameters:
    -----------
    price_data : pd.DataFrame or pd.Series
        Price data with datetime index
    fast_span : int, default=64
        Span for fast EWMA (equivalent to N=64 day window)
    slow_span : int, default=256
        Span for slow EWMA (equivalent to N=256 day window)
    price_col : str, default='Close'
        Column name for prices if DataFrame is passed
    can_short : bool
        If the option to short is possible
    return_signals : bool, default=False
        If True, returns additional columns with signals and EWMAs

    Returns:
    --------
    pd.DataFrame
        DataFrame with returns and optionally signals/EWMAs
    """

    # Handle both Series and DataFrame inputs
    if isinstance(price_data, pd.DataFrame):
        prices = price_data[price_col].copy()
    else:
        prices = price_data.copy()

    # Calculate EWMAs
    ewma_fast = prices.ewm(span=fast_span, adjust=False).mean()
    ewma_slow = prices.ewm(span=slow_span, adjust=False).mean()

    # Calculate EWMAC signal
    # EWMAC = EWMA_fast - EWMA_slow
    ewmac_signal = ewma_fast - ewma_slow

    # Generate trading positions
    # Go long if EWMAC > 0, else remain flat (or short)
    if can_short:
        positions = np.where(ewmac_signal > 0, 1, -1)  # Always positioned
    else:
        positions = np.where(ewmac_signal > 0, 1, 0)   # Long or flat

    # Shift positions to avoid look-ahead bias
    positions = pd.Series(positions, index=prices.index).shift(1)

    # Calculate daily returns
    daily_returns = prices.pct_change()

    # Calculate strategy returns
    strategy_returns = positions * daily_returns

    # Create results DataFrame
    results = pd.DataFrame({
        'Price': prices,
        'Daily_Returns': daily_returns,
        'Strategy_Returns': strategy_returns,
        'Cumulative_Returns': (1 + strategy_returns).cumprod() - 1,
        'Buy_Hold_Returns': (1 + daily_returns).cumprod() - 1
    })

    # Add additional columns if requested
    if return_signals:
        results['EWMA_Fast'] = ewma_fast
        results['EWMA_Slow'] = ewma_slow
        results['EWMAC_Signal'] = ewmac_signal
        results['Position'] = positions

    return results
