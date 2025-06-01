import numpy as np
import pandas as pd

def calculate_risk_adjusted_cost(tick_size, current_price, annualized_volatility, rolls_per_year,
                               multiplier=1.0, commission_pct=0.0, price_impact_pct=0.0,
                               turnover=1, currency_multiplier=1.0,
                               print_results=True):
    """
    Calculate risk-adjusted trading costs.
    
    Risk-adjusted costs account for both direct trading costs and the risk
    associated with holding positions, allowing fair comparison across
    different trading strategies and instruments.
    
    Formula: Risk-adjusted cost = Total cost per trade x σₐ (annualized volatility)
    
    Args:
        tick_size: Size per tick, smallest movement possible, 
        current_price: Current market price
        annualized_volatility: Annualized standard deviation (σₐ)
        multiplier: Contract multiplier (default: 1.0)
        commission_pct: Commission as percentage (default: 0.0)
        price_impact_pct: Price impact as percentage (default: 0.0)
        turnover: Number of trades per year (default: 1)
        currency_multiplier: Currency conversion multiplier (default: 1.0)
        print_results: Whether to print formatted results (default: True)
        
    Returns:
        Dictionary containing all calculated values
    """
    
    # Step 1: Calculate spread cost in points
    spread_points = tick_size / 2
    
    # Step 2: Calculate spread cost in currency terms
    spread_currency = spread_points * multiplier * currency_multiplier
    
    # Step 3: Calculate spread cost as percentage of current price
    spread_pct = (spread_points / current_price) * 100
    
    # Step 4: Calculate total cost per trade as percentage
    total_cost_pct = spread_pct + commission_pct + price_impact_pct
    
    # Step 5: Calculate risk-adjusted cost per trade
    risk_adjusted_cost = total_cost_pct * annualized_volatility
    
    annual_risk_adjusted_holding_cost = risk_adjusted_cost * rolls_per_year * 2
    
    annual_risk_adjusted_trading_cost = risk_adjusted_cost * turnover
    
    # Step 6: Calculate annual risk-adjusted cost
    annual_risk_adjusted_cost = annual_risk_adjusted_holding_cost + annual_risk_adjusted_trading_cost
    
    # Compile results
    results = {
        'spread_cost_points': spread_points,
        'spread_cost_currency': spread_currency,
        'spread_cost_percentage': spread_pct,
        'commission_percentage': commission_pct,
        'price_impact_percentage': price_impact_pct,
        'total_cost_per_trade_percentage': total_cost_pct,
        'annualized_volatility': annualized_volatility,
        'risk_adjusted_cost_per_trade': risk_adjusted_cost,
        'turnover': turnover,
        'annual_risk_adjusted_trading_cost': annual_risk_adjusted_trading_cost,
        'annual_risk_adjusted_holding_cost': annual_risk_adjusted_holding_cost,
        'annual_risk_adjusted_cost': annual_risk_adjusted_cost
    }
    
    # Print formatted results if requested
    if print_results:
        print("Risk-Adjusted Trading Cost Analysis")
        print("=" * 40)
        print(f"Spread cost (points): {spread_points:.4f}")
        print(f"Spread cost (currency): {spread_currency:.2f}")
        print(f"Spread cost (%): {spread_pct:.4f}%")
        
        if commission_pct > 0:
            print(f"Commission (%): {commission_pct:.4f}%")
        if price_impact_pct > 0:
            print(f"Price impact (%): {price_impact_pct:.4f}%")
            
        print(f"Total cost per trade (%): {total_cost_pct:.4f}%")
        print(f"Annualized volatility: {annualized_volatility:.4f}")
        print(f"Risk-adjusted cost per trade: {risk_adjusted_cost:.6f}")
        
        if turnover > 1:
            print(f"Trades per year: {turnover}")
            print(f"Annual risk-adjusted cost: {annual_risk_adjusted_cost:.6f}")
    
    return results

def calculate_annual_turnover(trades_data, position_data):
    """
    Calculate annual turnover based on trading activity and position data.
    
    Turnover = Number of times we turn over our average position per year
    
    Parameters:
    -----------
    trades_data : list or array
        Number of contracts traded in each period
    position_data : list or array  
        Average position size in each period
    
    Returns:
    --------
    float : Annual turnover ratio
    
    Examples:
    ---------
    # From the document example:
    # 60 contracts traded, average position 9.3 contracts
    # Turnover = 60 / 9.3 = 6.5 times
    
    >>> trades = [60]  # contracts traded
    >>> positions = [9.3]  # average position
    >>> calculate_annual_turnover(trades, positions)
    6.451612903225806
    """
    
    trades_array = np.array(trades_data)
    positions_array = np.array(position_data)
    
    if len(trades_array) != len(positions_array):
        raise ValueError("Trades and positions data must have the same length")
            
    # Rolling calculation for multiple periods
    turnovers = []
    for i in range(len(trades_array)):
        if positions_array[i] != 0:
            turnovers.append(trades_array[i] / positions_array[i])
    
    turnover = np.mean(turnovers) if turnovers else 0
    
    return turnover

def calculate_rolling_turnover(trades_df: pd.DataFrame, window_months=6):
    """
    Calculate rolling average turnover over specified time window.
    
    Parameters:
    -----------
    trades_df : pandas.DataFrame
        DataFrame with columns 'trades' and 'avg_position' and datetime index
    window_months : int, default 6
        Rolling window size in months
        
    Returns:
    --------
    pandas.Series : Rolling turnover values
    """
    
    if not isinstance(trades_df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have datetime index")
    
    required_cols = ['trades', 'avg_position']
    if not all(col in trades_df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    # Calculate turnover for each period
    trades_df['turnover'] = trades_df['trades'] / trades_df['avg_position']
    trades_df['turnover'] = trades_df['turnover'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate rolling average
    rolling_turnover = trades_df['turnover'].rolling(
        window=f'{window_months}M', 
        min_periods=1
    ).mean()
    
    return rolling_turnover