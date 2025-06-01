import pandas as pd
import numpy as np
from scipy.stats import skew
from typing import Union, List, Dict, Optional
import calc_stats

def create_strategy_summary_table(
    daily_returns: Union[List[float], np.ndarray, pd.Series],
    strategy_name: str = "Trading Strategy",
    instrument_name: str = "Instrument",
    risk_free_rate: float = 0.0,
    return_dataframe: bool = True
) -> Union[pd.DataFrame, Dict]:
    """
    Create a comprehensive summary statistics table for a trading strategy.

    Parameters:
    -----------
    daily_returns : array-like
        Series of daily returns as decimals (e.g., 0.02 for 2%)
    strategy_name : str, default="Trading Strategy"
        Name of the trading strategy for the table header
    instrument_name : str, default="Instrument"
        Name of the instrument being traded
    risk_free_rate : float, default=0.0
        Risk-free rate for Sharpe ratio calculation (annualized)
    return_dataframe : bool, default=True
        If True, returns formatted DataFrame. If False, returns dictionary of values.

    Returns:
    --------
    pd.DataFrame or dict
        Summary statistics table or dictionary containing all calculated metrics

    Example:
    --------
    >>> daily_returns = np.random.normal(0.0005, 0.015, 10000)  # Simulated daily returns
    >>> summary = create_strategy_summary_table(daily_returns, "Buy and Hold", "S&P 500")
    >>> print(summary)
    """

    # Convert to numpy array and remove NaN values
    daily_returns = np.array(daily_returns, dtype=float)
    daily_returns = daily_returns[~np.isnan(daily_returns)]

    if len(daily_returns) == 0:
        raise ValueError("No valid daily returns data provided")

    # Calculate years of data (assuming 256 trading days per year)
    years_of_data = len(daily_returns) / calc_stats.NUM_TRADING_DAYS_ANNUAL

    # Basic return statistics
    mean_daily_return = np.mean(daily_returns)
    mean_annual_return_val = calc_stats.mean_annual_return(mean_daily_return)

    # Volatility measures
    annualized_std = calc_stats.annual_std_dev(daily_returns)

    # Risk measures
    avg_drawdown = calc_stats.average_drawdown(daily_returns)

    # Sharpe ratio (using excess return over risk-free rate)
    daily_risk_free = risk_free_rate / calc_stats.NUM_TRADING_DAYS_ANNUAL
    excess_daily_return = mean_daily_return - daily_risk_free
    sharpe = calc_stats.sharpe_ratio(excess_daily_return * calc_stats.NUM_TRADING_DAYS_ANNUAL, annualized_std)

    # Skewness
    returns_skew = skew(daily_returns)

    # Fat tail measures
    try:
        tail_measures = calc_stats.measure_fat_tails(daily_returns)
        lower_tail_ratio = tail_measures['lower_tail']
        upper_tail_ratio = tail_measures['upper_tail']
    except:
        # Fallback if fat tail calculation fails
        lower_tail_ratio = np.nan
        upper_tail_ratio = np.nan

    # Create the summary dictionary
    summary_stats = {
        'Strategy': f"{strategy_name}, single contract",
        'Instrument': instrument_name,
        'Years of data': round(years_of_data),
        'Mean annual return': f"{mean_annual_return_val:.1%}",
        'Average drawdown': f"{avg_drawdown:.1%}",
        'Annualised standard deviation': f"{annualized_std:.1%}",
        'Sharpe ratio': f"{sharpe:.2f}",
        'Skew': f"{returns_skew:.2f}",
        'Lower tail': f"{lower_tail_ratio:.2f}" if not np.isnan(lower_tail_ratio) else "N/A",
        'Upper tail': f"{upper_tail_ratio:.2f}" if not np.isnan(upper_tail_ratio) else "N/A"
    }

    if not return_dataframe:
        return summary_stats

    # Create DataFrame for nice tabular display
    df = pd.DataFrame([summary_stats]).T
    df.columns = [f"{strategy_name} {instrument_name}"]

    return df

def create_multi_strategy_summary(
    strategies_data: Dict[str, Dict],
    risk_free_rate: float = 0.0
) -> pd.DataFrame:
    """
    Create a summary table comparing multiple trading strategies.

    Parameters:
    -----------
    strategies_data : dict
        Dictionary where keys are strategy names and values are dictionaries containing:
        - 'returns': array-like of daily returns
        - 'instrument': str, name of instrument (optional)
    risk_free_rate : float, default=0.0
        Risk-free rate for Sharpe ratio calculation

    Returns:
    --------
    pd.DataFrame
        Comparison table of all strategies

    Example:
    --------
    >>> strategies = {
    ...     'Buy and Hold S&P 500': {'returns': sp500_returns, 'instrument': 'S&P 500'},
    ...     'Moving Average Strategy': {'returns': ma_returns, 'instrument': 'S&P 500'}
    ... }
    >>> comparison = create_multi_strategy_summary(strategies)
    >>> print(comparison)
    """

    all_summaries = {}

    for strategy_name, data in strategies_data.items():
        returns = data['returns']
        instrument = data.get('instrument', 'Unknown')

        # Get raw statistics (as dictionary)
        stats = create_strategy_summary_table(
            returns,
            strategy_name,
            instrument,
            risk_free_rate,
            return_dataframe=False
        )

        # Use strategy name as column header
        all_summaries[strategy_name] = stats

    # Create combined DataFrame
    df = pd.DataFrame(all_summaries)

    # Reorder rows to match the standard format
    row_order = [
        'Years of data',
        'Mean annual return',
        'Average drawdown',
        'Annualised standard deviation',
        'Sharpe ratio',
        'Skew',
        'Lower tail',
        'Upper tail'
    ]

    # Only include rows that exist in the data
    existing_rows = [row for row in row_order if row in df.index]
    df = df.loc[existing_rows]

    return df