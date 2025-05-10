import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quantstats as qs
from dow import calculate_equity_curve_dow_theory

# Optional: Enable pandas integration with QuantStats
qs.extend_pandas()

# Define contract units (example values - you'll need to adjust these to your actual values)
CONTRACT_UNITS = {
    'Natural_Gas_data.csv': 10000,        # 10,000 MMBtu per contract
    'Cotton.csv': 50000,                  # 50,000 lbs per contract
    'Gold_data.csv': 100,                 # 100 troy ounces per contract
    'Coffee.csv': 37500,                  # 37,500 lbs per contract
    'Platinum_data.csv': 50,              # 50 troy ounces per contract
    'Soybean_data.csv': 5000,             # 5,000 bushels per contract
    'US 5 Year T-Note Futures Historical Data.csv': 100000,  # $100,000 face value per contract
    'US 10 Year T-Note Futures Historical Data.csv': 100000, # $100,000 face value per contract
}

def calculate_daily_returns_from_equity(equity_curve):
    """
    Calculate daily returns from an equity curve for QuantStats compatibility
    
    Parameters:
    -----------
    equity_curve : pandas Series
        Series containing the equity curve values
        
    Returns:
    --------
    daily_returns : pandas Series
        Series containing the daily returns as percentages
    """
    daily_returns = equity_curve.pct_change().fillna(0)
    return daily_returns

def create_performance_report_dow_theory(dfs, initial_capital=50000, strategy_name="Futures Trading Strategy"):
    """
    Create a comprehensive performance report using Dow Theory and QuantStats.
    
    Parameters:
    -----------
    dfs : list of DataFrames
        List of pandas DataFrames containing futures price data
    initial_capital : float
        Initial capital for the strategy
    strategy_name : str
        Name of the strategy for reporting purposes
    
    Returns:
    --------
    fig : matplotlib figure
        Figure object containing the equity curve
    """
    # Process each futures contract using Dow Theory
    strategy_dfs = []
    trade_dfs = []
    
    for i, df in enumerate(dfs):
        # Apply Dow Theory trading strategy
        strategy_df, trades_df = calculate_equity_curve_dow_theory(
            df, 
            initial_capital=initial_capital/len(dfs),  # Divide capital among contracts
            use_position_sizing=False
        )
        strategy_dfs.append(strategy_df)
        trade_dfs.append(trades_df)
    
    # Combine equity curves
    combined_equity = pd.Series(0, index=strategy_dfs[0].index)
    for strategy_df in strategy_dfs:
        if 'contract_equity' in strategy_df.columns:
            # Align indices to handle potentially different date ranges
            common_index = combined_equity.index.intersection(strategy_df.index)
            combined_equity.loc[common_index] += strategy_df.loc[common_index, 'contract_equity']
    
    # Add initial capital to get total portfolio value
    portfolio_equity = combined_equity + initial_capital
    
    # Calculate daily returns for QuantStats
    portfolio_returns = calculate_daily_returns_from_equity(portfolio_equity)
    portfolio_returns.name = strategy_name
    
    # Optional: Download a benchmark (e.g., S&P 500) for comparison
    try:
        benchmark = qs.utils.download_returns('SPY', period='max')
        # Align dates with portfolio returns
        benchmark = benchmark[benchmark.index.isin(portfolio_returns.index)]
    except Exception as e:
        print(f"Could not download benchmark: {e}")
        benchmark = None
    
    # Generate a tearsheet HTML report
    report_filename = f"{strategy_name.replace(' ', '_').lower()}_report.html"
    print("Creating Report")
    qs.reports.html(
        portfolio_returns,
        benchmark=benchmark,
        title=strategy_name,
        output=report_filename
    )
    print(f"Full performance report saved as '{report_filename}'")
    
    # Calculate key performance metrics
    metrics = {
        'Annual Return': qs.stats.cagr(portfolio_returns),
        'Cumulative Return': qs.stats.comp(portfolio_returns),
        'Sharpe Ratio': qs.stats.sharpe(portfolio_returns),
        'Max Drawdown': qs.stats.max_drawdown(portfolio_returns),
        'Volatility': qs.stats.volatility(portfolio_returns),
        'Sortino Ratio': qs.stats.sortino(portfolio_returns),
        'Calmar Ratio': qs.stats.calmar(portfolio_returns),
        'Win Rate': qs.stats.win_rate(portfolio_returns),
        'Profit Factor': qs.stats.profit_factor(portfolio_returns),
        'Ulcer Index': qs.stats.ulcer_index(portfolio_returns),
        'Expected Return (1-day)': qs.stats.expected_return(portfolio_returns),
        'Expected Return (1-month)': qs.stats.expected_return(portfolio_returns, aggregate='M'),
        'Expected Return (1-year)': qs.stats.expected_return(portfolio_returns, aggregate='A'),
    }
    
    # Print metrics
    print("\n----- Performance Metrics -----")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # Create equity curve
    fig, ax = plt.subplots(figsize=(12, 6))
    portfolio_equity.plot(ax=ax, title=f"{strategy_name} - Equity Curve", linewidth=2)
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_xlabel('Date')
    ax.grid(True)
    
    # Add key metrics to the plot
    metrics_text = (
        f"CAGR: {metrics['Annual Return']:.2%}\n"
        f"Sharpe: {metrics['Sharpe Ratio']:.2f}\n"
        f"Max DD: {metrics['Max Drawdown']:.2%}\n"
        f"Win Rate: {metrics['Win Rate']:.2%}\n"
        f"Profit Factor: {metrics['Profit Factor']:.2f}"
    )
    plt.figtext(0.15, 0.15, metrics_text, bbox=dict(facecolor='white', alpha=0.8))
    
    return fig, portfolio_equity, portfolio_returns

def analyze_individual_futures_dow_theory(dfs, names, initial_capital):
    """
    Analyze each futures contract individually using Dow Theory
    
    Parameters:
    -----------
    dfs : list of DataFrames
        List of pandas DataFrames containing futures price data
    names : list of str
        Names of each futures contract
    initial_capital : float
        Initial capital per contract
    """
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    for df, name in zip(dfs, names):
        print(f"\n----- Analysis of {name} -----")
        
        # Apply Dow Theory trading strategy
        strategy_df, trades_df = calculate_equity_curve_dow_theory(
            df, 
            initial_capital=initial_capital,
            use_position_sizing=False
        )
        
        # Calculate daily returns for QuantStats
        daily_returns = strategy_df['contract_return'].fillna(0)
        daily_returns.name = name
        
        # Calculate and print metrics
        print(f"Sharpe Ratio: {qs.stats.sharpe(daily_returns):.4f}")
        print(f"Max Drawdown: {qs.stats.max_drawdown(daily_returns):.4%}")
        print(f"CAGR: {qs.stats.cagr(daily_returns):.4%}")
        print(f"Volatility: {qs.stats.volatility(daily_returns):.4%}")
        print(f"Sortino Ratio: {qs.stats.sortino(daily_returns):.4f}")
        print(f"Calmar Ratio: {qs.stats.calmar(daily_returns):.4f}")
        
        # Generate plots for each futures contract
        # Performance metrics tearsheet
        qs.reports.metrics(daily_returns, mode='full', display=False, 
                          output=f'plots/{name.replace(" ", "_")}_metrics.png')
        
        # Monthly returns heatmap
        fig = qs.plots.monthly_heatmap(daily_returns, figsize=(10, 6))
        plt.savefig(f'plots/{name.replace(" ", "_")}_monthly_heatmap.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Drawdown plot
        fig = qs.plots.drawdowns_periods(daily_returns, figsize=(10, 6), 
                                        title=f'{name} - Drawdowns')
        plt.savefig(f'plots/{name.replace(" ", "_")}_drawdowns.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Rolling Sharpe ratio
        fig = qs.plots.rolling_sharpe(daily_returns, figsize=(10, 6))
        plt.savefig(f'plots/{name.replace(" ", "_")}_rolling_sharpe.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create basic trade analysis report
        if len(trades_df) > 0:
            # Plot trade P&L distribution
            plt.figure(figsize=(10, 6))
            plt.hist(trades_df['profit'], bins=20, alpha=0.7)
            plt.axvline(0, color='r', linestyle='--')
            plt.title(f'{name} - Trade P&L Distribution')
            plt.xlabel('Profit/Loss ($)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'plots/{name.replace(" ", "_")}_trade_pnl.png', 
                        dpi=300, bbox_inches='tight')
            plt.close()

def create_correlation_matrix(returns_list, names):
    """
    Create a correlation matrix of all futures returns
    
    Parameters:
    -----------
    returns_list : list of Series
        List of pandas Series containing daily returns data
    names : list of str
        Names of each futures contract
    """
    # Create a combined DataFrame for correlation analysis
    combined_returns = pd.DataFrame()
    for i, returns in enumerate(returns_list):
        combined_returns[names[i]] = returns
    
    # Calculate correlation matrix
    corr_matrix = combined_returns.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    plt.title('Correlation Matrix of Futures Contracts', fontsize=16)
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none', aspect='auto')
    
    # Add colorbar
    plt.colorbar(label='Correlation Coefficient')
    
    # Add labels and ticks
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.yticks(range(len(names)), names)
    
    # Add correlation values
    for i in range(len(names)):
        for j in range(len(names)):
            text = plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black")
    
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return corr_matrix

def main():
    # Define the data files to process
    data_files = [
        # 'Natural_Gas_data.csv',
        # 'Cotton.csv',
        # 'Gold_data.csv',
        # 'Coffee.csv',
        # 'Platinum_data.csv',
        'Soybean_data.csv',
        # 'US 5 Year T-Note Futures Historical Data.csv',
        # 'US 10 Year T-Note Futures Historical Data.csv',
    ]
    
    # Create shortened names for readability
    file_names = [
        # 'Natural Gas',
        # 'Cotton',
        # 'Gold',
        # 'Coffee',
        # 'Platinum',
        'Soybean',
        # 'US 5Y T-Note',
        # 'US 10Y T-Note',
    ]
    
    # Initial capital settings
    initial_capital = 50000
    capital_per_contract = initial_capital / len(data_files)
    
    # Load and process all data files
    dfs = []
    for file in data_files:
        path = os.path.join(os.path.dirname(__file__), 'individual_data', file)
        df = pd.read_csv(path)
        df.columns = [col.capitalize() for col in df.columns]
        df['Open'] = df['Open'] * CONTRACT_UNITS[file]
        df['High'] = df['High'] * CONTRACT_UNITS[file]
        df['Low'] = df['Low'] * CONTRACT_UNITS[file]
        df['Close'] = df['Close'] * CONTRACT_UNITS[file]
        
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        dfs.append(df)
    
    # Analyze individual futures using Dow Theory
    analyze_individual_futures_dow_theory(dfs, file_names, capital_per_contract)
    
    # Create combined performance report using Dow Theory
    fig, portfolio_equity, portfolio_returns = create_performance_report_dow_theory(
        dfs, 
        initial_capital=initial_capital,
        strategy_name="Dow Theory Futures Strategy"
    )
    
    # Find the first non-zero value's index
    first_nonzero_idx = (portfolio_returns != 0).idxmax()

    # Create a new DataFrame starting from the first non-zero value
    portfolio_returns = portfolio_returns.loc[first_nonzero_idx:]
    
    # Extract daily returns for correlation analysis
    returns_list = []
    for df in dfs:
        strategy_df, _ = calculate_equity_curve_dow_theory(
            df, 
            initial_capital=capital_per_contract,
            use_position_sizing=False
        )
        returns_list.append(strategy_df['contract_return'])
    
    # Create correlation matrix
    corr_matrix = create_correlation_matrix(returns_list, file_names)
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    
    # Generate full QuantStats report
    qs.reports.html(
        portfolio_returns,
        # benchmark="SPY",
        output='dow_theory_futures_full_report.html',
        title='Dow Theory Futures Trading Strategy Analysis'
    )
    print("\nQuantStats full report saved as 'dow_theory_futures_full_report.html'")
    
    # Create a full tearsheet
    qs.reports.full(portfolio_returns, output='dow_theory_futures_tearsheet.png')
    print("QuantStats tearsheet saved as 'dow_theory_futures_tearsheet.png'")
    
    # Save equity curve
    plt.figure(figsize=(12, 6))
    portfolio_equity.plot(title="Dow Theory Futures Portfolio - Equity Curve", linewidth=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('dow_theory_futures_equity.png', dpi=300, bbox_inches='tight')
    print("Equity curve saved as 'dow_theory_futures_equity.png'")
    
    # Optional: Show plots
    # plt.show()

if __name__ == "__main__":
    main()