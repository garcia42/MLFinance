import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import os
from dow import calculate_equity_curve_dow_theory
from units import CONTRACT_UNITS
from risk_of_ruin import monte_carlo_risk_of_ruin
from matplotlib.backends.backend_pdf import PdfPages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import os
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Option 2: Create a multi-page PDF
def create_multipage_pdf(equity_df, results):
    """Create a multi-page PDF with each visualization on its own page"""
    with PdfPages('trading_analysis.pdf') as pdf:
        # Page 1: Candlestick chart with trade overlays
        fig1, ax1 = plt.subplots(figsize=(15, 8))
        create_trade_visualization_subplot(equity_df, ax1)
        pdf.savefig(fig1)
        plt.close(fig1)
        
        # Page 2: Equity Curve
        fig_equity, ax_equity = plt.subplots(figsize=(12, 7))
        plot_equity_curve_subplot(equity_df, ax_equity)
        pdf.savefig(fig_equity)
        plt.close(fig_equity)
        
        # Page 2: Simulation paths
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        plot_simulation_paths_subplot(results, ax2)
        pdf.savefig(fig2)
        plt.close(fig2)
        
        # Page 3: Balance distribution
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        plot_final_balance_distribution_subplot(results, ax3)
        pdf.savefig(fig3)
        plt.close(fig3)
    
    print("Multi-page visualization saved as 'trading_analysis.pdf'")

# Helper function for candlestick subplot
def create_trade_visualization_subplot(equity_df, ax):
    """Create candlestick chart with trade overlays on the provided axis"""
    # Filter to first 3 months of data (or adjust as needed)
    start_date = equity_df.index[0]
    last_date = start_date + pd.DateOffset(months=3)
    equity_df = equity_df[equity_df.index <= last_date]
    
    # We need to handle the candlestick chart separately since mpf creates its own figure
    # Instead, we'll create a temporary figure with mpf and extract the data
    
    # Create trailing swing points
    last_high = None
    last_low = None
    trailing_highs = []
    trailing_lows = []
    
    for idx in equity_df.index:
        if not pd.isna(equity_df.loc[idx, 'swing_high']):
            last_high = equity_df.loc[idx, 'swing_high']
        if not pd.isna(equity_df.loc[idx, 'swing_low']):
            last_low = equity_df.loc[idx, 'swing_low']
        trailing_highs.append(last_high)
        trailing_lows.append(last_low)
    
    equity_df['trailing_high'] = trailing_highs
    equity_df['trailing_low'] = trailing_lows
    
    # Plot candlesticks directly on the axis
    # Note: This is a simplified version as mpf doesn't work directly with existing axes
    # For a complete solution, you'd need to draw candlesticks manually or use a different approach
    
    # Instead, we'll plot OHLC data as line and bar charts
    dates = np.arange(len(equity_df))
    ax.plot(dates, equity_df['Close'], color='black', linewidth=1, label='Close')
    
    # Plot swing points
    for i, idx in enumerate(equity_df.index):
        if not pd.isna(equity_df.loc[idx, 'swing_high']):
            ax.scatter(i, equity_df.loc[idx, 'swing_high'], marker='^', s=100, color='red')
    
    for i, idx in enumerate(equity_df.index):
        if not pd.isna(equity_df.loc[idx, 'swing_low']):
            ax.scatter(i, equity_df.loc[idx, 'swing_low'], marker='v', s=100, color='green')
    
    # Plot trailing lines
    ax.plot(dates, equity_df['trailing_high'], color='red', linestyle='--', alpha=0.5)
    ax.plot(dates, equity_df['trailing_low'], color='green', linestyle='--', alpha=0.5)
    
    # Add position overlays
    long_mask = equity_df['signal'] == 1
    short_mask = equity_df['signal'] == -1
    
    # Plot position overlays
    if long_mask.any():
        ax.fill_between(dates, ax.get_ylim()[0], ax.get_ylim()[1], 
                      where=long_mask.values, color='green', alpha=0.1)
    if short_mask.any():
        ax.fill_between(dates, ax.get_ylim()[0], ax.get_ylim()[1],
                      where=short_mask.values, color='red', alpha=0.1)
    
    # Create custom legend elements
    legend_elements = [
        Line2D([0], [0], marker='^', color='red', label='Swing High', 
               markersize=10, linestyle='none'),
        Line2D([0], [0], marker='v', color='green', label='Swing Low', 
               markersize=10, linestyle='none'),
        Line2D([0], [0], color='red', label='Last Swing High', 
               linestyle='--', alpha=0.5),
        Line2D([0], [0], color='green', label='Last Swing Low', 
               linestyle='--', alpha=0.5),
        Patch(facecolor='green', alpha=0.1, label='Long Position (Buy)'),
        Patch(facecolor='red', alpha=0.1, label='Short Position (Sell)')
    ]
    
    # Add custom legend and labels
    ax.legend(handles=legend_elements, loc='upper left')
    ax.set_title('Price with Trade Overlays')
    ax.set_ylabel('Price')
    ax.set_xticks(np.arange(0, len(equity_df), len(equity_df)//10))
    ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in equity_df.index[::len(equity_df)//10]], rotation=45)
    ax.grid(True, alpha=0.3)

# Helper function for simulation paths subplot
def plot_simulation_paths_subplot(results, ax, max_paths_to_show=100):
    """Plot simulation paths on the provided axis"""
    # Get random sample of paths to avoid overcrowding
    paths_to_plot = min(max_paths_to_show, results["paths"].shape[0])
    random_indices = np.random.choice(results["paths"].shape[0], paths_to_plot, replace=False)
    
    # Plot individual paths
    for idx in random_indices:
        path = results["paths"][idx]
        if path[-1] <= 0:
            ax.plot(path, color='red', alpha=0.1)  # Failed simulations in red
        else:
            ax.plot(path, color='blue', alpha=0.1)  # Successful simulations in blue
    
    # Plot average path of successful simulations
    successful_paths = results["paths"][results["paths"][:, -1] > 0]
    if len(successful_paths) > 0:
        avg_path = np.mean(successful_paths, axis=0)
        ax.plot(avg_path, color='green', linewidth=2, label='Average (Successful)')
    
    # Add labels and title
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.axhline(y=results["parameters"]["starting_balance"], color='black', linestyle=':', alpha=0.3)
    ax.set_xlabel('Number of Trades')
    ax.set_ylabel('Account Balance')
    ax.set_title(f'Monte Carlo Simulation: Risk of Ruin = {results["risk_of_ruin_percent"]:.2f}%')
    ax.grid(True, alpha=0.3)
    ax.legend()

# Helper function for balance distribution subplot
def plot_final_balance_distribution_subplot(results, ax):
    """Plot histogram of final balances on the provided axis"""
    # Get only positive final balances for the histogram
    final_balances = results["paths"][:, -1]
    positive_balances = final_balances[final_balances > 0]
    
    ax.hist(positive_balances, bins=30, alpha=0.7, color='blue')
    ax.axvline(x=results["parameters"]["starting_balance"], color='red', 
              linestyle='--', label=f'Starting Balance (${results["parameters"]["starting_balance"]})')
    
    ax.set_xlabel('Final Account Balance')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Final Account Balances (Surviving Accounts)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add text with risk of ruin
    ax.text(0.05, 0.95, f'Risk of Ruin: {results["risk_of_ruin_percent"]:.2f}%', 
           transform=ax.transAxes, fontsize=12,
           bbox=dict(facecolor='white', alpha=0.8))

# New helper function for equity curve subplot
def plot_equity_curve_subplot(equity_df, ax):
    """Create equity curve visualization on the provided axis"""
    # Plot equity curve
    ax.plot(equity_df.index, equity_df['equity'], color='blue', linewidth=2)
    
    # Add initial capital horizontal line
    initial_capital = equity_df['equity'].iloc[0]
    ax.axhline(y=initial_capital, color='red', linestyle='--', 
               label=f'Initial Capital (${initial_capital:,.0f})')
    
    # Calculate drawdowns for shading
    equity_df['rolling_max'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['equity'] / equity_df['rolling_max'] - 1) * 100
    
    # Add drawdown as a shaded area at the bottom of the chart
    ax_dd = ax.twinx()
    ax_dd.fill_between(equity_df.index, equity_df['drawdown'], 0,
                     color='red', alpha=0.3, label='Drawdown %')
    ax_dd.set_ylim(-100, 0)  # Limit drawdown axis to -100% max
    ax_dd.set_ylabel('Drawdown %')
    
    # Calculate and annotate max drawdown
    max_dd = equity_df['drawdown'].min()
    max_dd_date = equity_df.loc[equity_df['drawdown'] == max_dd].index[0]
    ax_dd.annotate(f'Max DD: {max_dd:.1f}%', 
                 xy=(max_dd_date, max_dd),
                 xytext=(max_dd_date, max_dd/2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10, ha='center')


# Modify your main function to use the new visualization functions
def main():
    # Load data
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

    # Calculate signals using Dow Theory
    equity_df, trades_df = calculate_equity_curve_dow_theory(df, initial_capital=50000)
    
    avg_win = trades_df.loc[trades_df['profit'] > 0, 'profit'].mean() if sum(trades_df['profit'] > 0) > 0 else 0
    avg_loss = trades_df.loc[trades_df['profit'] < 0, 'profit'].mean() if sum(trades_df['profit'] < 0) > 0 else 0
    win_rate = (trades_df['profit'] > 0).mean()
    
    # Run simulation with parameters from your spreadsheet
    results = monte_carlo_risk_of_ruin(
        starting_balance=20,
        win_probability=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss * -1,
        num_trades=1000,
        num_simulations=1000
    )

    # Print results
    print(f"Risk of Ruin: {results['risk_of_ruin_percent']:.2f}%")
    print(f"Average Final Balance (for surviving accounts): ${results['avg_final_balance']:.2f}")
    
    create_multipage_pdf(equity_df, results)

if __name__ == "__main__":
    main()