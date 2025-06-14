import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

def plot_equity_curves_fixed(strategies_data, title="Portfolio Equity Curves", 
                            initial_capital=5000000, figsize=(14, 8)):
    """
    Plot equity curves with proper date sorting to show actual performance progression.
    
    Parameters:
    -----------
    strategies_data : dict
        Keys are strategy labels, values are DataFrames with columns:
        - 'Date': datetime index
        - 'Portfolio_Value': actual portfolio values
        or
        - 'Cumulative_Return': cumulative returns (will calculate portfolio value)
    title : str
        Plot title
    initial_capital : float
        Starting capital amount
    figsize : tuple
        Figure size (width, height)
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    
    print("DEBUGGING EQUITY CURVE DATA:")
    print("=" * 50)
    
    for i, (strategy_name, returns_df) in enumerate(strategies_data.items()):
        
        # Make a copy and ensure Date column is datetime
        df = returns_df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # CRITICAL FIX: Sort by date in ASCENDING order (oldest first)
        df = df.sort_values('Date', ascending=True).reset_index(drop=True)
        
        # Debug: Print date range and direction
        print(f"\n{strategy_name}:")
        print(f"  Original date range: {returns_df['Date'].iloc[0]} to {returns_df['Date'].iloc[-1]}")
        print(f"  Sorted date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
        print(f"  Data points: {len(df)}")
        
        # Get portfolio values
        if 'Portfolio_Value' in df.columns:
            portfolio_values = df['Portfolio_Value']
            dates = df['Date']
        elif 'Cumulative_Return' in df.columns:
            portfolio_values = initial_capital * (1 + df['Cumulative_Return'])
            dates = df['Date']
        else:
            raise ValueError(f"Cannot find portfolio value data for {strategy_name}")
        
        # Debug: Print portfolio value progression
        print(f"  Start portfolio value: ${portfolio_values.iloc[0]:,.0f}")
        print(f"  End portfolio value: ${portfolio_values.iloc[-1]:,.0f}")
        print(f"  Peak portfolio value: ${portfolio_values.max():,.0f}")
        
        # Calculate actual returns to verify direction
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100
        print(f"  Total return: {total_return:.1f}%")
        
        # Convert to millions for better readability
        portfolio_values_millions = portfolio_values / 1_000_000
        
        # Plot equity curve
        ax.plot(dates, portfolio_values_millions, 
               label=strategy_name, 
               color=colors[i % len(colors)],
               linestyle=linestyles[i % len(linestyles)],
               linewidth=1.8)
        
        # Add start and end markers for clarity
        ax.scatter(dates.iloc[0], portfolio_values_millions.iloc[0], 
                  color=colors[i % len(colors)], s=50, marker='o', zorder=5)
        ax.scatter(dates.iloc[-1], portfolio_values_millions.iloc[-1], 
                  color=colors[i % len(colors)], s=50, marker='s', zorder=5)
    
    # Customize the plot
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Portfolio Value ($M)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Format x-axis to show years nicely
    ax.xaxis.set_major_locator(mdates.YearLocator(2))  # Every 2 years
    ax.xaxis.set_minor_locator(mdates.YearLocator())   # Every year
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    
    # Grid and styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
    
    # Format y-axis to show millions with $ sign
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.1f}M'))
    
    # Add annotation explaining markers
    ax.text(0.02, 0.02, 'Circles = Start, Squares = End', 
            transform=ax.transAxes, fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Improve layout
    plt.tight_layout()
    
    return fig, ax


def diagnose_data_direction(strategies_data):
    """
    Diagnose if data is in reverse chronological order and needs fixing.
    """
    print("DATA DIRECTION DIAGNOSIS")
    print("=" * 40)
    
    for strategy_name, df in strategies_data.items():
        print(f"\n{strategy_name}:")
        
        # Check date order
        dates = pd.to_datetime(df['Date'])
        first_date = dates.iloc[0]
        last_date = dates.iloc[-1]
        is_ascending = dates.is_monotonic_increasing
        is_descending = dates.is_monotonic_decreasing
        
        print(f"  First date in data: {first_date}")
        print(f"  Last date in data: {last_date}")
        print(f"  Is chronological (ascending): {is_ascending}")
        print(f"  Is reverse chronological (descending): {is_descending}")
        
        # Check portfolio values
        if 'Portfolio_Value' in df.columns:
            start_value = df['Portfolio_Value'].iloc[0]
            end_value = df['Portfolio_Value'].iloc[-1]
            print(f"  Start portfolio value: ${start_value:,.0f}")
            print(f"  End portfolio value: ${end_value:,.0f}")
            print(f"  Apparent return: {((end_value/start_value - 1) * 100):.1f}%")
        
        if not is_ascending and is_descending:
            print(f"  ⚠️  WARNING: Data appears to be in REVERSE chronological order!")
            print(f"  ⚠️  This will make profitable strategies look like losses!")
