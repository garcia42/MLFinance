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
        '^FVX': '5-Year Treasury',
        '^GSPC': 'S&P 500',
        '^TYX': '30-Year Treasury',
        '^VIX': 'VIX',
        
        # New additions
        '^AXJO': 'ASX 200 (Australia)',
        '^FTSE': 'FTSE 100 (UK)',
        '^GDAXI': 'DAX (Germany)',
        'EUR=X': 'EUR/USD',
        'CL=F': 'Crude Oil',
        '^TNX': '10-Year Treasury Yield',
        '^IRX': '13-Week Treasury Yield',
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
    
    return combined_df

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
