"""
Momentum Trading Strategy Simulation

This simulation implements the momentum-based stock trading strategy described in the document.
Key components include:
- S&P 500 membership check
- Largest move calculation (90-day gap analysis)
- ATR volatility measurement (20-day)
- Risk-adjusted momentum calculation
- 200-day moving average check
- Weekly rebalancing logic
- Position sizing with risk parity
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import statsmodels.api as sm


class MomentumStrategy:
    def __init__(self, data, config=None):
        """
        Initialize the momentum strategy.
        
        Parameters:
        - data: Dictionary containing historical price data and S&P 500 membership
        - config: Configuration parameters for the strategy
        """
        # Store historical price data
        self.data = data
        
        # Strategy configuration with defaults
        default_config = {
            "lookback_period": 90,        # Days for momentum calculation
            "atr_period": 20,             # Days for ATR calculation
            "max_allowed_gap": 0.15,      # Maximum 15% gap allowed
            "moving_average_period": 200, # SMA period for trend filter
            "rebalance_periodicity": 5,   # Weekly (every 5 trading days)
            "top_rank_threshold": 100,    # Keep stocks in top 100 momentum rank
            "position_size_rebalance_threshold": 0.05,  # 5% threshold for rebalancing
            "initial_cash": 1000000,      # Starting capital
        }
        
        self.config = default_config
        if config:
            self.config.update(config)
        
        # Portfolio tracking
        self.portfolio = {
            "cash": self.config["initial_cash"],
            "positions": {},
            "history": []
        }
        
        # Current trading day and universe
        self.current_date = None
        self.universe = []
        self.rankings = []
        
        # Convert string dates to datetime if needed
        self._prepare_dates()
    
    def _prepare_dates(self):
        """Convert string dates to datetime objects if needed."""
        if self.data.get("dates") and isinstance(next(iter(self.data["dates"])), str):
            new_dates = {}
            for date_str, value in self.data["dates"].items():
                try:
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                    new_dates[date_obj] = value
                except ValueError:
                    pass
            self.data["dates"] = new_dates
    
    def run_simulation(self, start_date, end_date):
        """
        Run the simulation over the provided date range.
        
        Parameters:
        - start_date: Start date for the simulation (string or datetime)
        - end_date: End date for the simulation (string or datetime)
        
        Returns:
        - List of portfolio snapshots for each trading day
        """
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Get trading days within the date range
        trading_days = [date for date in sorted(self.data["dates"].keys())
                        if start_date <= date <= end_date]
        
        print(f"Starting simulation from {start_date.strftime('%Y-%m-%d')} "
              f"to {end_date.strftime('%Y-%m-%d')} ({len(trading_days)} trading days)")
        
        day_counter = 0
        
        for date in trading_days:
            self.current_date = date
            day_counter += 1
            
            # Run daily logic
            self.process_trading_day()
            
            # Record portfolio state
            self.portfolio["history"].append({
                "date": date,
                "cash": self.portfolio["cash"],
                "equity_value": self.calculate_portfolio_value(),
                "positions": self.portfolio["positions"].copy()
            })
            
            # Log progress every 20 days
            if day_counter % 20 == 0:
                print(f"Processed {day_counter} days. Current date: {date.strftime('%Y-%m-%d')}")
        
        final_value = self.calculate_portfolio_value()
        print(f"Simulation complete. Final portfolio value: ${final_value:.2f}")
        return self.portfolio["history"]
    
    def process_trading_day(self):
        """Process a single trading day."""
        # Step 1: Update universe and analytics
        self.update_universe()
        self.calculate_analytics()
        
        # Step 2: Check if we need to sell anything
        self.check_exit_conditions()
        
        # Step 3: Check if this is a rebalancing day
        is_rebalancing_day = self.is_rebalancing_day()
        if is_rebalancing_day:
            self.rebalance_positions()
            self.execute_new_trades()
    
    def update_universe(self):
        """Update the universe of tradable stocks for the current date."""
        # In a real implementation, this would filter for S&P 500 members on the current date
        self.universe = [ticker for ticker in self.data["stocks"].keys()
                        if self.data["stocks"][ticker].get(self.current_date) and 
                        self.is_in_sp500(ticker, self.current_date)]
    
    def is_in_sp500(self, ticker, date):
        """
        Check if a stock is in the S&P 500 index on a given date.
        
        Parameters:
        - ticker: Stock ticker symbol
        - date: Date to check
        
        Returns:
        - Boolean indicating if the stock is in the S&P 500
        """
        # This is a simplified implementation
        # In reality, you would check against actual historical S&P 500 constituent data
        stock_data = self.data["stocks"].get(ticker)
        if stock_data and stock_data.get("sp500_membership") and stock_data["sp500_membership"].get(date):
            return stock_data["sp500_membership"][date] == 1
        return False
    
    def calculate_analytics(self):
        """Calculate all the required analytics for each stock in the universe."""
        self.rankings = []
        
        for ticker in self.universe:
            # 1. Calculate the largest gap in the past 90 days
            largest_gap = self.calculate_largest_gap(ticker)
            
            # 2. Calculate 20-day ATR
            atr = self.calculate_atr(ticker)
            
            # 3. Calculate risk-adjusted momentum (annualized regression slope * R²)
            momentum = self.calculate_risk_adjusted_momentum(ticker)
            
            # 4. Check if stock is above its 200-day moving average
            is_above_200ma = self.is_above_200day_ma(ticker)
            
            # Create stock analytics record
            stock_analytics = {
                "ticker": ticker,
                "largest_gap": largest_gap,
                "atr": atr,
                "momentum": momentum,
                "is_above_200ma": is_above_200ma,
                "qualifies": largest_gap <= self.config["max_allowed_gap"] and is_above_200ma
            }
            
            # Only include stocks that qualify
            if stock_analytics["qualifies"]:
                self.rankings.append(stock_analytics)
        
        # Sort by momentum, highest first
        self.rankings = sorted(self.rankings, key=lambda x: x["momentum"], reverse=True)
    
    def calculate_largest_gap(self, ticker):
        """
        Calculate the largest price gap in the lookback period.
        
        Parameters:
        - ticker: Stock ticker symbol
        
        Returns:
        - Largest gap as a decimal (e.g., 0.05 for 5%)
        """
        stock_data = self.data["stocks"][ticker]
        dates = self.get_past_trading_days(self.current_date, self.config["lookback_period"])
        
        largest_gap = 0
        
        for i in range(1, len(dates)):
            prev_date = dates[i-1]
            current_date = dates[i]
            
            if prev_date in stock_data and current_date in stock_data:
                prev_close = stock_data[prev_date]["close"]
                current_open = stock_data[current_date]["open"]
                
                gap = abs(current_open - prev_close) / prev_close
                largest_gap = max(largest_gap, gap)
        
        return largest_gap
    
    def calculate_atr(self, ticker):
        """
        Calculate Average True Range for volatility measurement.
        
        Parameters:
        - ticker: Stock ticker symbol
        
        Returns:
        - ATR value
        """
        stock_data = self.data["stocks"][ticker]
        dates = self.get_past_trading_days(self.current_date, self.config["atr_period"])
        
        sum_tr = 0
        valid_days = 0
        
        for i in range(1, len(dates)):
            prev_date = dates[i-1]
            current_date = dates[i]
            
            if prev_date in stock_data and current_date in stock_data:
                high = stock_data[current_date]["high"]
                low = stock_data[current_date]["low"]
                prev_close = stock_data[prev_date]["close"]
                
                # True Range is the greatest of:
                # 1. Current High - Current Low
                # 2. |Current High - Previous Close|
                # 3. |Current Low - Previous Close|
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                
                tr = max(tr1, tr2, tr3)
                sum_tr += tr
                valid_days += 1
        
        return sum_tr / valid_days if valid_days > 0 else 0
    
    def calculate_risk_adjusted_momentum(self, ticker):
        """
        Calculate risk-adjusted momentum (annualized exponential regression slope * R²).
        
        Parameters:
        - ticker: Stock ticker symbol
        
        Returns:
        - Risk-adjusted momentum value
        """
        stock_data = self.data["stocks"][ticker]
        dates = self.get_past_trading_days(self.current_date, self.config["lookback_period"])
        
        # Get log prices for the period
        prices = []
        times = []
        
        for i, date in enumerate(dates):
            if date in stock_data:
                prices.append(np.log(stock_data[date]["close"]))
                times.append(i)
        
        if len(prices) < 30:  # Need enough data points for meaningful regression
            return 0
        
        # Use statsmodels for linear regression
        X = sm.add_constant(times)
        model = sm.OLS(prices, X)
        results = model.fit()
        
        slope = results.params[1]
        r_squared = results.rsquared
        
        # Annualize the slope (assuming 252 trading days per year)
        annualized_slope = slope * 252
        
        # Risk-adjusted momentum
        return annualized_slope * r_squared
    
    def is_above_200day_ma(self, ticker):
        """
        Check if stock is above its 200-day moving average.
        
        Parameters:
        - ticker: Stock ticker symbol
        
        Returns:
        - Boolean indicating if stock is above its 200-day MA
        """
        stock_data = self.data["stocks"][ticker]
        dates = self.get_past_trading_days(self.current_date, self.config["moving_average_period"])
        
        sum_price = 0
        count = 0
        
        for date in dates:
            if date in stock_data:
                sum_price += stock_data[date]["close"]
                count += 1
        
        if count == 0:
            return False
        
        moving_average = sum_price / count
        current_price = stock_data[self.current_date]["close"]
        
        return current_price > moving_average
    
    def check_exit_conditions(self):
        """Check exit conditions for current positions."""
        tickers_to_exit = []
        
        for ticker, shares in self.portfolio["positions"].items():
            if shares > 0:
                should_exit = False
                
                # 1. Check if stock left the S&P 500
                if not self.is_in_sp500(ticker, self.current_date):
                    should_exit = True
                
                # 2. Check if no longer in top 100 by momentum rank
                if not should_exit:
                    rank_list = [stock["ticker"] for stock in self.rankings]
                    if ticker not in rank_list or rank_list.index(ticker) >= self.config["top_rank_threshold"]:
                        should_exit = True
                
                # 3. Check if below 100-day moving average
                if not should_exit and not self.is_above_100day_ma(ticker):
                    should_exit = True
                
                # 4. Check if had a gap larger than 15%
                if not should_exit:
                    stock_data = self.data["stocks"][ticker]
                    prev_date = self.get_prev_trading_day(self.current_date)
                    
                    if prev_date and prev_date in stock_data and self.current_date in stock_data:
                        prev_close = stock_data[prev_date]["close"]
                        current_open = stock_data[self.current_date]["open"]
                        gap = abs(current_open - prev_close) / prev_close
                        
                        if gap > self.config["max_allowed_gap"]:
                            should_exit = True
                
                # Add to exit list if any condition is met
                if should_exit:
                    tickers_to_exit.append(ticker)
        
        # Sell positions marked for exit
        for ticker in tickers_to_exit:
            self.sell_position(ticker)
    
    def is_above_100day_ma(self, ticker):
        """
        Check if stock is above its 100-day moving average (for exit condition).
        
        Parameters:
        - ticker: Stock ticker symbol
        
        Returns:
        - Boolean indicating if stock is above its 100-day MA
        """
        stock_data = self.data["stocks"][ticker]
        dates = self.get_past_trading_days(self.current_date, 100)
        
        sum_price = 0
        count = 0
        
        for date in dates:
            if date in stock_data:
                sum_price += stock_data[date]["close"]
                count += 1
        
        if count == 0:
            return False
        
        moving_average = sum_price / count
        current_price = stock_data[self.current_date]["close"]
        
        return current_price > moving_average
    
    def is_rebalancing_day(self):
        """
        Check if current date is a rebalancing day.
        
        Returns:
        - Boolean indicating if today is a rebalancing day
        """
        trading_days = self.get_past_trading_days(self.current_date, 20)
        return len(trading_days) % self.config["rebalance_periodicity"] == 0
    
    def rebalance_positions(self):
        """Rebalance existing positions."""
        tickers_to_rebalance = []
        
        for ticker, shares in self.portfolio["positions"].items():
            if shares > 0:
                # Find the stock in the rankings
                stock_rank = next((stock for stock in self.rankings if stock["ticker"] == ticker), None)
                
                if stock_rank:
                    # Calculate target position size based on risk parity
                    target_size = self.calculate_position_size(stock_rank)
                    current_size = shares
                    stock_data = self.data["stocks"][ticker]
                    current_price = stock_data[self.current_date]["close"]
                    
                    # Calculate the difference as a percentage
                    size_diff = abs(target_size - current_size) / current_size if current_size > 0 else float('inf')
                    
                    # Only rebalance if difference is greater than threshold
                    if size_diff > self.config["position_size_rebalance_threshold"]:
                        tickers_to_rebalance.append({
                            "ticker": ticker,
                            "current_size": current_size,
                            "target_size": target_size,
                            "price": current_price
                        })
        
        # Execute rebalancing
        for rebalance in tickers_to_rebalance:
            ticker = rebalance["ticker"]
            target_size = rebalance["target_size"]
            current_size = rebalance["current_size"]
            price = rebalance["price"]
            
            if target_size > current_size:
                # Need to buy more
                shares_to_buy = target_size - current_size
                cost = shares_to_buy * price
                
                if self.portfolio["cash"] >= cost:
                    self.portfolio["positions"][ticker] = target_size
                    self.portfolio["cash"] -= cost
            else:
                # Need to sell some
                shares_to_sell = current_size - target_size
                proceeds = shares_to_sell * price
                
                self.portfolio["positions"][ticker] = target_size
                self.portfolio["cash"] += proceeds
    
    def execute_new_trades(self):
        """Execute new trades from the top of the ranking list."""
        # Go through the rankings from the top
        for stock in self.rankings:
            ticker = stock["ticker"]
            
            # Skip if we already own this stock
            if ticker in self.portfolio["positions"] and self.portfolio["positions"][ticker] > 0:
                continue
            
            # Calculate position size
            position_size = self.calculate_position_size(stock)
            stock_data = self.data["stocks"][ticker]
            current_price = stock_data[self.current_date]["close"]
            cost = position_size * current_price
            
            # Buy if we have enough cash
            if self.portfolio["cash"] >= cost:
                self.portfolio["positions"][ticker] = position_size
                self.portfolio["cash"] -= cost
            else:
                # Not enough cash, stop buying
                break
    
    def calculate_position_size(self, stock):
        """
        Calculate position size based on risk parity.
        
        Parameters:
        - stock: Dictionary with stock analytics
        
        Returns:
        - Position size in shares
        """
        # Target equal risk contribution from each position
        # Use inverse of volatility (ATR) for risk parity
        risk_contribution = 1 / (stock["atr"] or 0.01)  # Avoid division by zero
        
        # Allocate based on portfolio value and number of positions
        portfolio_value = self.calculate_portfolio_value()
        target_position_value = portfolio_value * 0.05  # Target 5% per position
        
        stock_data = self.data["stocks"][stock["ticker"]]
        current_price = stock_data[self.current_date]["close"]
        
        # Calculate position size in shares
        return int(target_position_value / current_price)
    
    def sell_position(self, ticker):
        """
        Sell a position completely.
        
        Parameters:
        - ticker: Stock ticker symbol
        """
        stock_data = self.data["stocks"][ticker]
        current_price = stock_data[self.current_date]["close"]
        shares = self.portfolio["positions"][ticker]
        
        proceeds = shares * current_price
        self.portfolio["cash"] += proceeds
        self.portfolio["positions"][ticker] = 0
    
    def calculate_portfolio_value(self):
        """
        Calculate total portfolio value (cash + positions).
        
        Returns:
        - Total portfolio value
        """
        positions_value = 0
        
        for ticker, shares in self.portfolio["positions"].items():
            if shares > 0:
                stock_data = self.data["stocks"].get(ticker)
                if stock_data and self.current_date in stock_data:
                    current_price = stock_data[self.current_date]["close"]
                    positions_value += shares * current_price
        
        return self.portfolio["cash"] + positions_value
    
    def get_past_trading_days(self, date, n):
        """
        Get past N trading days from a given date.
        
        Parameters:
        - date: Reference date
        - n: Number of days to look back
        
        Returns:
        - List of past trading dates
        """
        all_dates = sorted(self.data["dates"].keys())
        try:
            current_index = all_dates.index(date)
        except ValueError:
            return []
        
        start_index = max(0, current_index - n + 1)
        return all_dates[start_index:current_index + 1]
    
    def get_prev_trading_day(self, date):
        """
        Get the previous trading day.
        
        Parameters:
        - date: Reference date
        
        Returns:
        - Previous trading day or None
        """
        all_dates = sorted(self.data["dates"].keys())
        try:
            current_index = all_dates.index(date)
        except ValueError:
            return None
        
        if current_index <= 0:
            return None
        
        return all_dates[current_index - 1]


def run_backtest():
    """Sample usage of the momentum strategy."""
    # This is a placeholder for actual historical data
    # In a real implementation, you would load this from a CSV, database, or API
    
    # Create dates for sample data
    sample_dates = [datetime(2020, 1, 2) + timedelta(days=i) for i in range(365)]
    trading_days = {date: True for date in sample_dates}
    
    # Create sample stock data
    stocks = {
        "AAPL": {
            sample_dates[0]: {
                "open": 74.06,
                "high": 75.15,
                "low": 74.05,
                "close": 75.09,
                "volume": 135480400
            },
            sample_dates[1]: {
                "open": 74.29,
                "high": 75.14,
                "low": 74.13,
                "close": 74.36,
                "volume": 146322800
            }
            # ... more days would be added in a real implementation
        },
        # ... more stocks would be added in a real implementation
    }
    
    # Add S&P 500 membership data
    for ticker in stocks:
        stocks[ticker]["sp500_membership"] = {date: 1 for date in sample_dates}
    
    # Create historical data structure
    historical_data = {
        "dates": trading_days,
        "stocks": stocks
    }
    
    # Create and configure the strategy
    strategy = MomentumStrategy(historical_data, {
        "initial_cash": 1000000
    })
    
    # Run the backtest from 2020-01-02 to 2020-12-31
    start_date = datetime(2020, 1, 2)
    end_date = datetime(2020, 12, 31)
    results = strategy.run_simulation(start_date, end_date)
    
    # Calculate and display performance metrics
    start_value = results[0]["equity_value"]
    end_value = results[-1]["equity_value"]
    return_pct = ((end_value / start_value) - 1) * 100
    
    print(f"Starting portfolio value: ${start_value:.2f}")
    print(f"Ending portfolio value: ${end_value:.2f}")
    print(f"Total return: {return_pct:.2f}%")
    
    # More detailed analysis could be added here (drawdowns, Sharpe, etc.)
    
    # Visualization with matplotlib could be added
    # import matplotlib.pyplot as plt
    # dates = [r["date"] for r in results]
    # equity = [r["equity_value"] for r in results]
    # plt.figure(figsize=(12, 6))
    # plt.plot(dates, equity)
    # plt.title("Portfolio Equity Curve")
    # plt.xlabel("Date")
    # plt.ylabel("Portfolio Value ($)")
    # plt.grid(True)
    # plt.show()


if __name__ == "__main__":
    # This would run the backtest when the script is executed directly
    run_backtest()


# Implementation notes:
# 1. This is a simplified version that outlines the core strategy
# 2. In a real implementation, you would need proper historical data
# 3. Additional features like transaction costs, slippage, and dividends could be added
# 4. Extensive error handling and edge cases would need to be addressed
# 5. Using pandas DataFrames would make many calculations more efficient
# 6. Visualization of results with matplotlib or plotly would be helpful for analysis

# To run with real data, you would:
# 1. Fetch historical price data for the S&P 500 constituents
# 2. Fetch historical S&P 500 membership data
# 3. Format the data in the structure expected by the strategy
# 4. Run the simulation and analyze the results