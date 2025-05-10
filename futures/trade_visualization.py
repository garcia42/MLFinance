import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import os
from dow import calculate_equity_curve_dow_theory
from units import CONTRACT_UNITS

def create_trade_visualization(df):
    """Create visualization showing price data with trade overlays"""
    # Calculate signals using Dow Theory
    equity_df, trades_df = calculate_equity_curve_dow_theory(df, initial_capital=50000)
    
    # Filter to last 4 months
    # last_date = equity_df.index[-1]
    # start_date = last_date - pd.DateOffset(months=6)
    # equity_df = equity_df[equity_df.index >= start_date]
    
    start_date = equity_df.index[0]
    last_date = start_date + pd.DateOffset(months=32)
    equity_df = equity_df[equity_df.index <= last_date]
    
    # Prepare data for mplfinance
    plot_data = equity_df[['Open', 'High', 'Low', 'Close']].copy()

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
    
    # Prepare addplots for mplfinance
    addplots = []
    
    # Add swing high points
    swing_high_data = equity_df['swing_high'].copy()
    addplots.append(mpf.make_addplot(swing_high_data, type='scatter', 
                                   marker='^', markersize=100, color='red'))
    
    # Add swing low points
    swing_low_data = equity_df['swing_low'].copy()
    addplots.append(mpf.make_addplot(swing_low_data, type='scatter',
                                   marker='v', markersize=100, color='green'))
    
    # Add trailing lines
    addplots.append(mpf.make_addplot(equity_df['trailing_high'], color='red', 
                                   linestyle='--', alpha=0.5))
    addplots.append(mpf.make_addplot(equity_df['trailing_low'], color='green',
                                   linestyle='--', alpha=0.5))
    
    # Create candlestick chart with addplots
    kwargs = dict(type='candle', 
                 style='charles',
                 title='Natural Gas Price with Trade Overlays',
                 ylabel='Price',
                 volume=False,
                 figsize=(15, 8),
                 warn_too_much_data=10000000,
                 addplot=addplots,
                 returnfig=True)
    
    fig, axlist = mpf.plot(plot_data, **kwargs)
    ax1 = axlist[0]
    
    # Add position overlays
    long_mask = equity_df['signal'] == 1
    short_mask = equity_df['signal'] == -1
    
    # Get x-axis dates in numeric format
    dates_num = [i for i in range(len(equity_df))]
    
    # Plot position overlays
    if long_mask.any():
        ax1.fill_between(dates_num, ax1.get_ylim()[0], ax1.get_ylim()[1], 
                        where=long_mask, color='green', alpha=0.1)
    if short_mask.any():
        ax1.fill_between(dates_num, ax1.get_ylim()[0], ax1.get_ylim()[1],
                        where=short_mask, color='red', alpha=0.1)
    
    # Create custom legend elements
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
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
    
    # Add custom legend
    ax1.legend(handles=legend_elements, loc='upper left')
    
    # Save plot
    plt.savefig('futures/trade_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'trade_visualization.png'")

def main():
    # Load data
    file = 'Soybean_data.csv'
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
    
    # Create visualization
    create_trade_visualization(df)

if __name__ == "__main__":
    main()
