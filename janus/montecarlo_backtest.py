import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from units import CONTRACT_UNITS
import os
from dow import calculate_equity_curve_dow_theory
from typing import List, Union

def get_permutation(
    ohlc: Union[pd.DataFrame, List[pd.DataFrame]], start_index: int = 0, seed=None
):
    assert start_index >= 0

    np.random.seed(seed)

    if isinstance(ohlc, list):
        time_index = ohlc[0].index
        for mkt in ohlc:
            assert np.all(time_index == mkt.index), "Indexes do not match"
        n_markets = len(ohlc)
    else:
        n_markets = 1
        time_index = ohlc.index
        ohlc = [ohlc]

    n_bars = len(ohlc[0])

    perm_index = start_index + 1
    perm_n = n_bars - perm_index

    start_bar = np.empty((n_markets, 4))
    relative_open = np.empty((n_markets, perm_n))
    relative_high = np.empty((n_markets, perm_n))
    relative_low = np.empty((n_markets, perm_n))
    relative_close = np.empty((n_markets, perm_n))

    for mkt_i, reg_bars in enumerate(ohlc):
        log_bars = np.log(reg_bars[['Open', 'High', 'Low', 'Close']])

        # Get start bar
        start_bar[mkt_i] = log_bars.iloc[start_index].to_numpy()

        # Open relative to last close
        r_o = (log_bars['Open'] - log_bars['Close'].shift()).to_numpy()
        
        # Get prices relative to this bars open
        r_h = (log_bars['High'] - log_bars['Open']).to_numpy()
        r_l = (log_bars['Low'] - log_bars['Open']).to_numpy()
        r_c = (log_bars['Close'] - log_bars['Open']).to_numpy()

        relative_open[mkt_i] = r_o[perm_index:]
        relative_high[mkt_i] = r_h[perm_index:]
        relative_low[mkt_i] = r_l[perm_index:]
        relative_close[mkt_i] = r_c[perm_index:]

    idx = np.arange(perm_n)

    # Shuffle intrabar relative values (high/low/close)
    perm1 = np.random.permutation(idx)
    relative_high = relative_high[:, perm1]
    relative_low = relative_low[:, perm1]
    relative_close = relative_close[:, perm1]

    # Shuffle last close to open (gaps) seprately
    perm2 = np.random.permutation(idx)
    relative_open = relative_open[:, perm2]

    # Create permutation from relative prices
    perm_ohlc = []
    for mkt_i, reg_bars in enumerate(ohlc):
        perm_bars = np.zeros((n_bars, 4))

        # Copy over real data before start index 
        log_bars = np.log(reg_bars[['Open', 'High', 'Low', 'Close']]).to_numpy().copy()
        perm_bars[:start_index] = log_bars[:start_index]
        
        # Copy start bar
        perm_bars[start_index] = start_bar[mkt_i]

        for i in range(perm_index, n_bars):
            k = i - perm_index
            perm_bars[i, 0] = perm_bars[i - 1, 3] + relative_open[mkt_i][k]
            perm_bars[i, 1] = perm_bars[i, 0] + relative_high[mkt_i][k]
            perm_bars[i, 2] = perm_bars[i, 0] + relative_low[mkt_i][k]
            perm_bars[i, 3] = perm_bars[i, 0] + relative_close[mkt_i][k]

        perm_bars = np.exp(perm_bars)
        perm_bars = pd.DataFrame(perm_bars, index=time_index, columns=['Open', 'High', 'Low', 'Close'])

        perm_ohlc.append(perm_bars)

    if n_markets > 1:
        return perm_ohlc
    else:
        return perm_ohlc[0]

def calculate_returns(data):
    """Calculate daily returns from price data."""
    data['Returns'] = data['Close'].pct_change()
    return data.dropna()

def backtest(returns, signals):
    """
    Backtest a strategy given returns and position signals.
    Returns the equity curve and performance metrics.
    """
    # Calculate strategy returns
    strategy_returns = returns * signals
    
    # Calculate equity curve (cumulative returns)
    equity_curve = (1 + strategy_returns).cumprod()
    
    # Calculate performance metrics
    total_return = equity_curve.iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(equity_curve)) - 1
    daily_std = strategy_returns.std()
    annual_std = daily_std * np.sqrt(252)
    sharpe_ratio = annual_return / annual_std if annual_std != 0 else 0
    max_drawdown = (equity_curve / equity_curve.cummax() - 1).min()
    
    metrics = {
        'Total Return': total_return,
        'Annual Return': annual_return,
        'Annual Volatility': annual_std,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }
    
    return equity_curve, metrics

def monte_carlo_random_strategies(data: pd.DataFrame, n_simulations=100):
    """
    Run Monte Carlo simulation by generating random strategies.
    Each random strategy randomly selects commodities to trade and 
    randomly assigns signals (-1 for short, 0 for cash, 1 for long)
    with equal probability.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The price data for one commodity
    n_simulations : int
        Number of random strategies to generate
    all_commodities : list
        List of all commodities to consider (if None, assumes only using the provided data)
    
    Returns:
    --------
    list of pd.Series
        List of equity curves for each random strategy
    """
    # Store all equity curves
    all_equity_curves = []
    
    # Run simulations
    for i in tqdm(range(n_simulations)):
        # Generate random signals with equal probability of -1, 0, 1
        random_signals = pd.Series(
            np.random.choice([-1, 0, 1], size=len(data)), 
            index=data.index
        )
        
        # Calculate returns
        if 'Returns' not in data.columns:
            returns = data['Close'].pct_change().dropna()
        else:
            returns = data['Returns']
        
        # Calculate strategy returns
        strategy_returns = returns * random_signals[returns.index]
        
        # Calculate equity curve (cumulative returns)
        equity_curve = (1 + strategy_returns).cumprod()
        
        all_equity_curves.append(equity_curve)
    
    return all_equity_curves

def monte_carlo_permutation(returns: pd.DataFrame, n_simulations=1000):
    """
    Run Monte Carlo simulation by permuting the returns between dates.
    Returns all equity curves.
    """
    # Store all equity curves
    all_equity_curves = []
    
    # Original dates
    original_dates = returns.index
    
    # Calculate original returns if not already present
    if 'Returns' not in returns.columns:
        original_returns = returns['Close'].pct_change().dropna()
    else:
        original_returns = returns['Returns']
    
    # Run simulations
    for i in tqdm(range(n_simulations)):
        # Create a copy of the original dataframe
        permuted_data = get_permutation(returns)
        
        # Calculate equity curve with the new price data
        equity_df, _ = calculate_equity_curve_dow_theory(permuted_data)
        
        all_equity_curves.append(equity_df["contract_equity"])
    
    return all_equity_curves

def plot_equity_curves(original_equity, monte_carlo_equities, percentiles=[5, 50, 95]):
    """
    Plot the original equity curve and Monte Carlo simulations with percentiles.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot original equity curve
    plt.plot(original_equity, color='blue', linewidth=2, label='Original Strategy')
    
    # Calculate and plot percentiles
    monte_carlo_df = pd.DataFrame(monte_carlo_equities).T
    for p in percentiles:
        percentile_curve = monte_carlo_df.apply(lambda x: np.percentile(x, p), axis=1)
        plt.plot(percentile_curve, 
                 linestyle='--', 
                 linewidth=1.5, 
                 label=f'{p}th Percentile')
    
    # Plot a sample of Monte Carlo simulations (faded)
    sample_size = min(100, len(monte_carlo_equities))
    sample_indices = np.random.choice(len(monte_carlo_equities), sample_size, replace=False)
    
    for idx in sample_indices:
        plt.plot(monte_carlo_equities[idx], 
                 color='gray', 
                 alpha=0.1, 
                 linewidth=0.5)
    
    plt.title('Monte Carlo Simulation of Strategy Performance')
    plt.xlabel('Date')
    plt.ylabel('Equity Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt

def run_monte_carlo_backtest(data, n_simulations=10):
    """
    Complete workflow for Monte Carlo backtest simulation.
    """
    
    # 3. Apply trading strategy to get signals
    print("Generating trading signals...")
    equity_df, _ = calculate_equity_curve_dow_theory(data)
    
    # 5. Run Monte Carlo simulation
    print(f"Running {n_simulations} Monte Carlo simulations...")
    mc_equities = monte_carlo_permutation(data, n_simulations)
    # mc_equities = monte_carlo_random_strategies(data, n_simulations)
    
    # 6. Plot results
    print("Generating plots...")
    equity_plot = plot_equity_curves(equity_df["contract_equity"], mc_equities)

    return {
        'data': data,
        'mc_equities': mc_equities,
        'equity_plot': equity_plot,
    }

# Example usage
if __name__ == "__main__":
    # Define parameters
    n_simulations = 50  # Reduced for faster execution
    
    file = 'Gold_data.csv'
    csv_path = os.path.join(os.path.dirname(__file__), 'individual_data', file)
    df = pd.read_csv(csv_path)
    
    # Prepare data
    df.columns = [col.capitalize() for col in df.columns]
    df['Date'] = pd.to_datetime(df['Date'])
    df['Open'] = df['Open'] * CONTRACT_UNITS[file]
    df['High'] = df['High'] * CONTRACT_UNITS[file]
    df['Low'] = df['Low'] * CONTRACT_UNITS[file]
    df['Close'] = df['Close'] * CONTRACT_UNITS[file]
    df.set_index('Date', inplace=True)
    
    # Run the Monte Carlo backtest
    results = run_monte_carlo_backtest(df, n_simulations)
    
    # Show plots
    results['equity_plot'].show()
