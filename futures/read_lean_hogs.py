import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from typing import Tuple, List

class LeanHogsAnalyzer:
    def __init__(self):
        # Get the absolute path to the CSV file
        self.csv_path = os.path.join(os.path.dirname(__file__), 'individual_data', 'Lean Hogs.csv')
        self.df = pd.read_csv(self.csv_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.set_index('Date', inplace=True)

    def calculate_moving_averages(self, prices: pd.Series, short_period: int, long_period: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate short and long moving averages."""
        short_ma = prices.rolling(window=short_period).mean()
        long_ma = prices.rolling(window=long_period).mean()
        return short_ma, long_ma

    def generate_signals(self, short_ma: pd.Series, long_ma: pd.Series) -> pd.Series:
        """Generate trading signals based on moving average crossovers."""
        signals = pd.Series(0, index=short_ma.index)
        signals[short_ma > long_ma] = 1  # Buy signal
        signals[short_ma < long_ma] = -1  # Sell signal
        return signals

    def calculate_returns(self, prices: pd.Series, signals: pd.Series) -> float:
        """Calculate strategy returns."""
        daily_returns = prices.pct_change()
        strategy_returns = signals.shift(1) * daily_returns
        return strategy_returns.sum()

    def optimize_parameters(self, train_data: pd.DataFrame, short_range: List[int], long_range: List[int]) -> Tuple[int, int]:
        """Find optimal MA periods using training data."""
        best_return = float('-inf')
        best_params = (0, 0)

        for short_period in short_range:
            for long_period in long_range:
                if short_period >= long_period:
                    continue
                
                short_ma, long_ma = self.calculate_moving_averages(train_data['Close'], short_period, long_period)
                signals = self.generate_signals(short_ma, long_ma)
                returns = self.calculate_returns(train_data['Close'], signals)
                
                if returns > best_return:
                    best_return = returns
                    best_params = (short_period, long_period)
        
        return best_params

    def walk_forward_test(self, train_window: int = 252, test_window: int = 126) -> pd.DataFrame:
        """
        Perform walk-forward testing of moving average crossover strategy.
        
        Args:
            train_window: Number of days for training (default 1 year)
            test_window: Number of days for testing (default 6 months)
        
        Returns:
            DataFrame with test results including returns and parameters used
        """
        results = []
        short_range = range(5, 51, 5)  # 5 to 50 days
        long_range = range(10, 201, 10)  # 10 to 200 days
        
        # Walk forward through the data
        for start_idx in range(0, len(self.df) - train_window - test_window, test_window):
            # Define training and testing periods
            train_start = start_idx
            train_end = start_idx + train_window
            test_start = train_end
            test_end = test_start + test_window
            
            train_data = self.df.iloc[train_start:train_end]
            test_data = self.df.iloc[test_start:test_end]
            
            # Skip if not enough data
            if len(train_data) < train_window or len(test_data) < test_window:
                continue
            
            # Optimize parameters on training data
            best_short, best_long = self.optimize_parameters(train_data, short_range, long_range)
            
            # Apply strategy to test data
            short_ma, long_ma = self.calculate_moving_averages(test_data['Close'], best_short, best_long)
            signals = self.generate_signals(short_ma, long_ma)
            returns = self.calculate_returns(test_data['Close'], signals)
            
            # Calculate win rate
            daily_returns = test_data['Close'].pct_change()
            strategy_returns = signals.shift(1) * daily_returns
            win_rate = (strategy_returns > 0).mean()
            
            results.append({
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'short_ma': best_short,
                'long_ma': best_long,
                'returns': returns,
                'win_rate': win_rate
            })
        
        return pd.DataFrame(results)

    def print_daily_values(self):
        """Print each day's values."""
        for date, row in self.df.iterrows():
            print(f"Date: {date}")
            print(f"Open: {row['Open']}")
            print(f"High: {row['High']}")
            print(f"Low: {row['Low']}")
            print(f"Close: {row['Close']}")
            print(f"Volume: {row['Volume']}")
            print("-" * 50)

# Example usage
if __name__ == "__main__":
    analyzer = LeanHogsAnalyzer()
    
    # Perform walk-forward test
    results = analyzer.walk_forward_test()
    
    # Print summary statistics
    print("\nWalk-Forward Test Results:")
    print("=========================")
    print(f"Average Returns: {results['returns'].mean():.4f}")
    print(f"Average Win Rate: {results['win_rate'].mean():.2%}")
    print(f"Best Period Returns: {results['returns'].max():.4f}")
    print(f"Worst Period Returns: {results['returns'].min():.4f}")
    print("\nMoving Average Periods Used:")
    print(f"Short MA - Mean: {results['short_ma'].mean():.1f}, Median: {results['short_ma'].median()}")
    print(f"Long MA - Mean: {results['long_ma'].mean():.1f}, Median: {results['long_ma'].median()}")

    # Plot returns over time
    plt.figure(figsize=(12, 6))
    
    # Plot individual period returns
    plt.subplot(2, 1, 1)
    plt.plot(results['test_start'], results['returns'], marker='o')
    plt.title('Returns by Test Period')
    plt.ylabel('Period Returns')
    plt.grid(True)
    
    # Plot cumulative returns
    plt.subplot(2, 1, 2)
    cumulative_returns = (1 + results['returns']).cumprod() - 1
    plt.plot(results['test_start'], cumulative_returns, marker='o')
    plt.title('Cumulative Returns Over Time')
    plt.ylabel('Cumulative Returns')
    plt.grid(True)
    
    plt.tight_layout()
    # Save plots to files instead of displaying
    plt.savefig('futures/returns/moving_average_returns.png')
    plt.close()
    print("\nPlots saved to futures/returns/moving_average_returns.png")
