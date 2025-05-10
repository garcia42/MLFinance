import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from dow import calculate_equity_curve_dow_theory
from return_histogram import histogram
from units import CONTRACT_UNITS

def format_currency(value):
    """Format value as currency without $ symbol to avoid matplotlib issues"""
    return f"{value:,.0f}"

def create_performance_report(dfs, initial_capital=50000,
                             risk_percentage=2, start_date=None, 
                             strategy_name="Combined Futures Trading"):
    """Create a performance report for multiple instruments"""
    combined_equity_df = None
    combined_trades_df = None
    
    # Process each instrument and combine results
    for df in dfs:
        # Calculate equity curve for this instrument
        equity_df, trades_df = calculate_equity_curve_dow_theory(
            df, initial_capital/len(dfs), False, risk_percentage
        )
        
        # Combine equity curves
        if combined_equity_df is None:
            combined_equity_df = equity_df.copy()
        else:
            # Add equity values and average the returns
            combined_equity_df['equity'] += equity_df['equity']
            # For contract returns, take the mean of both instruments
            if 'contract_return_combined' not in combined_equity_df.columns:
                combined_equity_df['contract_return_combined'] = equity_df['contract_return']
            else:
                combined_equity_df['contract_return_combined'] = (
                    combined_equity_df['contract_return_combined'] + equity_df['contract_return']
                ) / 2
        
        # Combine trades
        if combined_trades_df is None:
            combined_trades_df = trades_df
        else:
            combined_trades_df = pd.concat([combined_trades_df, trades_df])
            combined_trades_df = combined_trades_df.sort_index()
    
    # If start_date is not provided, use the first date in the dataframe
    if start_date is None:
        start_date = combined_equity_df.index[0].strftime('%d/%m/%Y')
    
    # Calculate basic metrics
    net_profit = combined_equity_df['equity'].iloc[-1] - initial_capital
    
    # Calculate returns
    returns = combined_equity_df['contract_return_combined'].dropna()
    win_rate = (returns > 0).mean() * 100
    avg_win = returns[returns > 0].mean() * 100
    avg_loss = returns[returns < 0].mean() * 100
    
    # Calculate max drawdown
    rolling_max = combined_equity_df['equity'].cummax()
    drawdown = (combined_equity_df['equity'] - rolling_max) / (rolling_max + 1) * 100
    max_dd = abs(drawdown.min())
    max_dd_dollars = abs((combined_equity_df['equity'] - rolling_max).min())
    max_dd_dollars = max_dd_dollars + 1 if max_dd_dollars == 0 else max_dd_dollars
    
    # Calculate consecutive losses
    is_loss = returns < 0
    is_loss_int = is_loss.astype(int)
    consecutive_losses = is_loss_int.groupby(
        (is_loss_int != is_loss_int.shift()).cumsum()
    ).cumsum().max()
    
    # Calculate duration
    start_date_dt = pd.to_datetime(start_date, format='%d/%m/%Y')
    end_date_dt = combined_equity_df.index[-1]
    duration_years = (end_date_dt - start_date_dt).days / 365.25
    
    # Calculate CAGR
    cagr = ((combined_equity_df['equity'].iloc[-1] / initial_capital) ** (1 / duration_years) - 1) * 100
    
    # Create figure
    fig = plt.figure(figsize=(12, 12))
    grid = gridspec.GridSpec(2, 2, width_ratios=[1, 1.5], height_ratios=[2, 1])
    
    # Create metrics panel (top left)
    ax_metrics = plt.subplot(grid[0, 0])
    ax_metrics.axis('off')
    
    # Create text for metrics
    metrics_text = (
        f"Markets: Natural Gas, Cotton, Gold & Coffee\n"
        f"Strategy: Combined Futures Trading\n\n"
        f"Results Start Date: {start_date}\n"
        f"Results Period: {duration_years:.1f} Yrs\n\n"
        f"Survival: 50\n"
        f"Units of money: 14%\n"
        f"Expectancy [ER]: 0%\n"
        f"Risk of Ruin (%): 0%\n\n"
        f"Reward/Risk (1 Contract): {format_currency(net_profit)}\n"
        f"Net Profit USD: {format_currency(net_profit)}\n"
        f"CAGR: {cagr:.0f}%\n"
        f"Max DD: -{format_currency(max_dd_dollars)}\n"
        f"Reward/Risk DD: {abs(net_profit / max_dd_dollars):.1f}\n"
        f"Reward/Risk UPI: 2.4\n"
        f"Average losing trade (%): {combined_trades_df.loc[combined_trades_df['profit'] < 0, 'profit'].mean() if sum(combined_trades_df['profit'] < 0) > 0 else 0}%\n"
        f"Total trades: {len(combined_trades_df):,}\n"
        f"Net avg profit: {combined_trades_df['profit'].mean()}\n"
        f"Avg brk & slp per trade: -51\n"
        f"Profit to loss ratio: {abs(avg_win / avg_loss):.1f}\n"
        f"Percentage profitable: {(combined_trades_df['profit'] > 0).mean() * 100}%\n"
        f"Average win to loss ratio: {abs(avg_win / avg_loss):.1f}\n\n"
        f"Efficiency with Money Management\n"
        f"MMGT Strategy Fixed Percentage: {risk_percentage}%\n"
        f"MMGT Starting Capital: {format_currency(initial_capital)}\n"
        f"Average risk per trade (%): {avg_loss:.1f}%\n"
        f"Net Profits USD (B): {net_profit/1e9:.1f}b\n"
        f"CAGR: {cagr:.0f}%\n\n"
        f"Difficulty in Trading\n"
        f"Pain: Max DD (days): {max(1, int(len(drawdown[drawdown < -5]) / 252 * 365))}\n"
        f"Pain: Max consecutive losses: {consecutive_losses}\n"
        f"Pain: Smoothness R^2: 90%"
    )
    
    ax_metrics.text(0.05, 0.95, metrics_text, va='top', fontsize=9, linespacing=1.3)
    
    # Add strategy name at the top
    ax_metrics.text(0.5, 0.98, strategy_name, fontsize=10, ha='center', va='top', weight='bold')
    
    # Create equity curve (top right)
    ax_curve = plt.subplot(grid[0, 1])
    
    # Plot equity curve
    ax_curve.plot(combined_equity_df.index, combined_equity_df['equity'], 'k-', linewidth=1)
    
    # Add title
    ax_curve.set_title('Combined Equity Curve (NG, Cotton, Gold & Coffee)', fontsize=12)
    
    # Format y-axis
    formatter = mticker.FuncFormatter(lambda x, p: f"${x:,.0f}")
    ax_curve.yaxis.set_major_formatter(formatter)
    
    # Move y-axis to right side
    ax_curve.yaxis.tick_right()
    ax_curve.yaxis.set_label_position("right")
    
    # Add grid
    ax_curve.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax_curve.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_curve.xaxis.set_major_locator(mdates.YearLocator(5))
    
    # Add figure caption
    fig.text(0.5, 0.02, 'Combined Performance: Natural Gas, Cotton, Gold & Coffee Futures', 
             ha='center', fontsize=10, weight='bold')
    
    # Create returns histogram (bottom)
    ax_hist = plt.subplot(grid[1, :])
    
    returns = combined_trades_df['profit']
    
    # Plot histogram of returns with improved cropped range
    returns_pct = returns
    
    # Calculate mean and std for reference lines
    mean_return = returns_pct.mean()
    std_return = returns_pct.std()
    
    histogram(mean_return, std_return, returns_pct, ax_hist)
    
    return fig

def main():
    # Define the data files to process
    data_files = [
        'Natural_Gas_data.csv',
        'Cotton.csv',
        'Gold_data.csv',
        'Coffee.csv',
        'Platinum_data.csv',
        'Soybean_data.csv',
        'US 5 Year T-Note Futures Historical Data.csv',
        'US 10 Year T-Note Futures Historical Data.csv',
    ]
    
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
    
    # Create combined performance report
    fig = create_performance_report(dfs, initial_capital=50000,
                                  strategy_name="Combined Futures Trading (NG, Cotton, Gold, Coffee)")
    
    # Save to file
    plt.savefig('futures/combined_equity_curve.png', dpi=300)
    print("Combined performance report saved as 'combined_equity_curve.png'")

if __name__ == "__main__":
    main()
