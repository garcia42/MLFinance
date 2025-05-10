import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
import logging
import time
from typing import Tuple, Optional, Dict, List
import requests
from ib_insync import *
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FUTURES = {
    # Equity Index Futures
    'ES=F': 'E-mini S&P 500',
    # 'NQ=F': 'E-mini NASDAQ 100',
    # 'RTY=F': 'E-mini Russell 2000',
    # 'YM=F': 'E-mini Dow',
    # ... other futures definitions remain unchanged
}

SPY_100_TOP_50_NASDAQ_ONLY = {
    # Technology 
    "AAPL": 'Apple Inc.',
    "MSFT": 'Microsoft Corporation',
    "NVDA": 'NVIDIA Corporation',
    'META': 'Meta Platforms Inc.',
    'AVGO': 'Broadcom Inc.',
    'ADBE': 'Adobe Inc.',
    'CSCO': 'Cisco Systems Inc.',
    'SNOW': 'Snow',  # Replacing ORCL
    'INTC': 'Intel Corporation',
    'INTU': 'Intuit Inc.',  # Replacing CRM
    # ... rest of symbols remain unchanged
}

async def fetch_stock_history(ib: IB, symbol: str, years_back: int = 5, exchange: str = 'SMART') -> Optional[pd.DataFrame]:
    """
    Fetch historical daily market data for a stock from Interactive Brokers using async methods.
    
    Parameters:
    ib (IB): Active IB connection instance
    symbol (str): Stock symbol (e.g., 'AAPL')
    years_back (int): Number of years of historical data to fetch
    exchange (str): Exchange to use, defaults to 'SMART' for best execution
    
    Returns:
    pd.DataFrame: DataFrame with columns Date, Open, High, Low, Close, Volume or None if error
    """
    try:
        # Create stock contract
        stock = Stock(symbol, exchange, 'USD')
        
        # Qualify the contract first
        qualified_contracts = await ib.qualifyContractsAsync(stock)
        if not qualified_contracts:
            logger.error(f"Failed to qualify contract for {symbol}")
            return None
            
        qualified_stock = qualified_contracts[0]
        
        # Calculate duration
        end_date = datetime.now()
        duration = f'{years_back} Y'
        
        # Request historical data using async method
        bars = await ib.reqHistoricalDataAsync(
            contract=qualified_stock,
            endDateTime=end_date,
            durationStr=duration,
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        
        if not bars:
            logger.error(f"No data received for {symbol}")
            return None
            
        # Convert to DataFrame
        df = util.df(bars)
        
        # Clean up and format DataFrame
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Rename columns to match standard format
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

async def fetch_market_indicators(ib: IB, lookback_years: int, symbols: Dict[str, str] = None) -> Tuple[pd.DataFrame, int]:
    """
    Fetch market relationship indicators from Interactive Brokers.
    Handles missing values using forward fill.
    
    Parameters:
    ib (IB): Active IB connection
    lookback_years (int): Number of years to look back for historical data
    symbols (Dict[str, str], optional): Dictionary of symbols to fetch, defaults to SPY_100_TOP_50_NASDAQ_ONLY
    
    Returns:
    Tuple[pd.DataFrame, int]: 
        - DataFrame containing market data in long format (Date, Market, Open, High, Low, Close)
        - Number of markets successfully fetched
    """    
    if symbols is None:
        symbols = SPY_100_TOP_50_NASDAQ_ONLY

    symbol_list = list(symbols.keys())
    
    # Create tasks for all symbols
    tasks = [fetch_stock_history(ib, symbol, lookback_years) for symbol in symbol_list]
    
    # Execute all tasks concurrently with proper error handling
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    symbol_dfs = {}
    all_dates = set()
    
    for symbol, result in zip(symbol_list, results):
        if isinstance(result, Exception):
            logger.error(f"Failed to fetch {symbol}: {str(result)}")
        elif result is not None:
            symbol_dfs[symbol] = result
            all_dates.update(result.index)
    
    if not symbol_dfs:
        raise ValueError("No valid data could be fetched for any symbol")
    
    # Convert to sorted list
    all_dates = sorted(list(all_dates))
    
    # Set minimum value to avoid zeros
    MIN_VALUE = 1e-60

    # Create rows with forward fill
    rows = []
    for symbol in symbol_list:
        if symbol in symbol_dfs:
            df = symbol_dfs[symbol]
            
            # Reindex to include all dates and forward fill
            df = df.reindex(all_dates).ffill()
            
            # Convert each row to the required format
            for date, row in df.iterrows():
                rows.append({
                    'Date': date,
                    'Market': symbol,
                    'Open': max(row['Open'], MIN_VALUE),
                    'High': max(row['High'], MIN_VALUE),
                    'Low': max(row['Low'], MIN_VALUE),
                    'Close': max(row['Close'], MIN_VALUE),
                })
    
    # Create DataFrame and sort to ensure consistent order
    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(['Date', 'Market']).reset_index(drop=True)
    
    n_markets = len(symbol_dfs)
    
    # Verify all markets have the same number of rows
    market_counts = result_df.groupby('Market').size()
    if len(set(market_counts)) > 1:
        logger.warning("Not all markets have the same number of rows after forward fill:")
        for market, count in market_counts.items():
            logger.warning(f"{market}: {count} rows")

    return result_df, n_markets

# The following helper functions don't need modification as they don't interact with IB API
def fill_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing data in financial time series using appropriate methods
    for different types of instruments.
    
    Parameters:
    df (pd.DataFrame): DataFrame in long format with columns: Date, Market, Open, High, Low, Close
    
    Returns:
    pd.DataFrame: Filled DataFrame
    """
    # Create a copy to avoid modifying the original
    df_filled = df.copy()
    
    # Ensure Date is datetime
    df_filled['Date'] = pd.to_datetime(df_filled['Date'])
    
    # Define instrument groups
    market_indices = ['^GSPC', '^FTSE', '^GDAXI', '^AXJO']
    currencies = ['JPY=X', 'EUR=X', 'GBPUSD=X', 'AUDUSD=X', 'CADUSD=X']
    futures = ['GC=F', 'CL=F', 'HG=F', 'SI=F', 'NG=F']  # Futures contracts
    etfs = ['SPY', 'TLT', 'LQD']
    
    # Process each market type separately
    for market in df_filled['Market'].unique():
        mask = df_filled['Market'] == market
        
        if market in futures:
            # Forward fill futures data without limit
            for col in ['Open', 'High', 'Low', 'Close']:
                df_filled.loc[mask, col] = df_filled.loc[mask, col].fillna(method='ffill')
        
        elif market in market_indices:
            # Fill market indices during their trading hours
            for col in ['Open', 'High', 'Low', 'Close']:
                df_filled.loc[mask, col] = df_filled.loc[mask, col].fillna(method='ffill', limit=8)
        
        elif market in currencies:
            # Interpolate currencies with time-weighted values
            for col in ['Open', 'High', 'Low', 'Close']:
                df_filled.loc[mask, col] = df_filled.loc[mask, col].interpolate(method='linear', limit=4)
        
        elif market in etfs:
            # Forward fill ETFs similar to their underlying indices
            for col in ['Open', 'High', 'Low', 'Close']:
                df_filled.loc[mask, col] = df_filled.loc[mask, col].fillna(method='ffill', limit=8)
        
        elif market == '^VIX':
            # Special handling for VIX - use ffill with shorter window
            for col in ['Open', 'High', 'Low', 'Close']:
                df_filled.loc[mask, col] = df_filled.loc[mask, col].fillna(method='ffill')
    
    # Final backfill pass for any remaining missing values
    df_filled = df_filled.fillna(method='bfill')
    
    return df_filled

def generate_summary_stats(df):
    """
    Generate summary statistics for the dataset.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame with columns Date, Market, Open, High, Low, Close
    
    Returns:
    dict: Dictionary of DataFrames containing summary statistics by market
    """
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    
    # Group by Market
    grouped = df.groupby('Market')
    
    stats_dict = {}
    
    # Numeric columns to analyze
    numeric_cols = ['Open', 'High', 'Low', 'Close']
    
    for market, group in grouped:
        stats = pd.DataFrame({
            'start_date': group['Date'].min(),
            'end_date': group['Date'].max(),
            'data_points': group[numeric_cols].count().iloc[0],
            'missing_pct': (group[numeric_cols].isna().sum() / len(group) * 100).round(2).mean(),
            'mean': group[numeric_cols].mean().round(4),
            'std': group[numeric_cols].std().round(4),
            'min': group[numeric_cols].min().round(4),
            'max': group[numeric_cols].max().round(4)
        }, index=['Value'])
        
        stats_dict[market] = stats

    return stats_dict

def print_summary_stats(df):
    """
    Print formatted summary statistics for each market.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    """
    stats_dict = generate_summary_stats(df)
    
    for market, stats in stats_dict.items():
        print(f"\n=== {market} Statistics ===")
        print(stats)
        print("\n")