import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests

def fetch_market_indicators(start_date: datetime, end_date=None) -> pd.DataFrame:
    """
    Fetch market relationship indicators from Yahoo Finance.
    
    Parameters:
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format, defaults to today if None
    
    Returns:
    pandas.DataFrame: DataFrame containing market indicators
    """
    # If no end date specified, use today
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    # Define the symbols to fetch
    symbols = {
        # Original symbols
        'GC=F': 'Gold',
        'JPY=X': 'USD/JPY',
        '^GSPC': 'S&P 500',
        '^VIX': 'VIX',
        
        # New additions
        '^AXJO': 'ASX 200 (Australia)',
        '^FTSE': 'FTSE 100 (UK)',
        '^GDAXI': 'DAX (Germany)',
        'EUR=X': 'EUR/USD',
        'CL=F': 'Crude Oil',
        'SPY': 'S&P 500 ETF',
        '^NYHILO': 'NYSE New High/Low Index',
        
        # Additional bond ETFs
        'TLT': '20+ Year Treasury Bond ETF',
        # 'IEF': '7-10 Year Treasury Bond ETF',
        # 'SHY': '1-3 Year Treasury Bond ETF',
        'LQD': 'Investment Grade Corporate Bond ETF',
        # 'HYG': 'High Yield Corporate Bond ETF',
        
        # Additional currency pairs
        # 'GBPUSD=X': 'GBP/USD',
        # 'AUDUSD=X': 'AUD/USD',
        # 'CADUSD=X': 'CAD/USD',
        
        # Additional commodities
        # 'SI=F': 'Silver',
        'HG=F': 'Copper',
        # 'NG=F': 'Natural Gas'
    }
    
    # Initialize an empty dictionary to store the data
    data_dict = {}
    
    # Fetch data for each symbol
    for symbol, description in symbols.items():
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if not df.empty:
                # We'll use the adjusted closing price
                data_dict[symbol] = df['Close']
                print(f"Successfully fetched data for {symbols[symbol]} ({symbol})")
            else:
                print(f"No data available for {symbols[symbol]} ({symbol})")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
    
    # Combine all series into a single DataFrame
    combined_df = pd.DataFrame(data_dict)
    
    combined_df = fill_missing_data(combined_df)
    
    combined_df.index = combined_df.index.tz_localize(None)
    
    return combined_df

def fill_missing_data(df) -> pd.DataFrame:
    """
    Fill missing data in financial time series using appropriate methods
    for different types of instruments.
    
    Parameters:
    df (pd.DataFrame): DataFrame with financial instruments and timestamp column 'Date'
    
    Returns:
    pd.DataFrame: Filled DataFrame
    """
    # Create a copy to avoid modifying the original
    df_filled = df.copy()
    
    # Convert timestamp to datetime
    df_filled['datetime'] = pd.to_datetime(df_filled.index, unit='ms')
    df_filled = df_filled.set_index('datetime')
    
    # Group columns by type
    market_indices = ['^GSPC', '^FTSE', '^GDAXI', '^AXJO']
    currencies = ['JPY=X', 'EUR=X']
    commodities = ['GC=F', 'CL=F', 'HG=F']
    etfs = ['SPY', 'TLT', 'LQD']
    
    # Fill market indices during their trading hours
    for idx in market_indices:
        df_filled[idx] = df_filled[idx].fillna(method='ffill', limit=8)
    
    # Interpolate currencies with time-weighted values
    for curr in currencies:
        df_filled[curr] = df_filled[curr].interpolate(method='time', limit=4)
    
    # Forward fill commodities but reset at day boundaries
    for comm in commodities:
        df_filled[comm] = df_filled.groupby(df_filled.index.date)[comm].fillna(method='ffill')
        df_filled[comm] = df_filled[comm].fillna(method='ffill')

    # Forward fill ETFs similar to their underlying indices
    for etf in etfs:
        df_filled[etf] = df_filled[etf].fillna(method='ffill', limit=8)
    
    # Special handling for VIX - use ffill with shorter window
    # Risks having stale VIX data into the future but that's the tradeoff
    df_filled['^VIX'] = df_filled['^VIX'].fillna(method='ffill')
    
    df_filled = df_filled.fillna(method='bfill', axis=0)
    
    # Keep the original Date column
    df_filled['Date'] = df.index
    
    return df_filled

def generate_summary_stats(df):
    """
    Generate summary statistics for the dataset.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    
    Returns:
    pandas.DataFrame: Summary statistics
    """
    stats = pd.DataFrame({
        'start_date': df.index.min(),
        'end_date': df.index.max(),
        'data_points': df.count(),
        'missing_pct': (df.isna().sum() / len(df) * 100).round(2),
        'mean': df.mean().round(4),
        'std': df.std().round(4),
        'min': df.min().round(4),
        'max': df.max().round(4)
    }).T
    
    return stats
