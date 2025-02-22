import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
import logging
import time
from typing import Tuple, Optional
# Configure a custom user agent
import requests
from ib_insync import *
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FUTURES  = {
    # Equity Index Futures
    'ES=F': 'E-mini S&P 500',
    # 'NQ=F': 'E-mini NASDAQ 100',
    # 'RTY=F': 'E-mini Russell 2000',
    # 'YM=F': 'E-mini Dow',

    # # Currency Futures
    # '6E=F': 'Euro FX',
    # '6J=F': 'Japanese Yen',
    # '6B=F': 'British Pound',
    # '6A=F': 'Australian Dollar',
    # '6C=F': 'Canadian Dollar',
    # '6M=F': 'Mexican Peso',
    # '6N=F': 'New Zealand Dollar',
    # '6S=F': 'Swiss Franc',

    # # Metal Futures
    # 'GC=F': 'Gold',
    # 'SI=F': 'Silver',
    # 'HG=F': 'Copper',
    # 'PL=F': 'Platinum',
    # 'PA=F': 'Palladium',

    # # Energy Futures
    # 'CL=F': 'Crude Oil WTI',
    # 'BZ=F': 'Brent Crude',
    # 'NG=F': 'Natural Gas',
    # 'HO=F': 'Heating Oil',
    # 'RB=F': 'RBOB Gasoline',

    # # Interest Rate Futures
    # 'ZN=F': '10-Year T-Note',
    # 'ZT=F': '2-Year T-Note',
    # 'ZF=F': '5-Year T-Note',
    # 'ZB=F': '30-Year T-Bond',
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
    
    # Financial Services
    'PYPL': 'PayPal Holdings Inc.',
    'ADYEY': 'Adyen N.V.',  # Replacing V
    'AFRM': 'Affirm Holdings',  # Replacing MA
    'COIN': 'Coinbase Global Inc.',
    'HOOD': 'Robinhood Markets Inc.',
    'MKTX': 'MarketAxess Holdings',  # Replacing FITB
    'SBNY': 'Signature Bank',
    'CFLT': 'Confluent Inc.',
    'SOFI': 'SoFi Technologies Inc.',
    'EXFY': 'Expensify Inc.',  # Adding new fintech
    
    # Healthcare
    'REGN': 'Regeneron Pharmaceuticals',
    'GILD': 'Gilead Sciences Inc.',
    'IDXX': 'IDEXX Laboratories',  # Replacing LLY
    'MRNA': 'Moderna Inc.',
    'DXCM': 'Dexcom.',  # Replacing MRK
    'BNTX': 'BioNTech SE',
    'ILMN': 'Illumina Inc.',
    'VRTX': 'Vertex Pharmaceuticals',
    "AMGN": "Amgen",
    "BIIB": "Biogen Inc.",
    
    # Consumer/Tech
    'AMZN': 'Amazon.com Inc.',
    'ABNB': 'Airbnb Inc.',
    'DDOG': 'Datadog Inc.',
    'COST': 'Costco Wholesale Corporation',
    'LCID': 'Lucid Group Inc.',
    'MNDY': 'Monday.com',  # Replacing PEP
    'LULU': 'Lululemon Athletica',
    'ETSY': 'Etsy Inc.',
    "BKNG": "Booking Holdings",
    "TEAM": "Atlassian Corporation",
    
    # Tech/Growth
    'RIVN': 'Rivian Automotive',
    'PANW': 'Palo Alto Networks',
    'ZM': 'Zoom Video Communications',
    'CRWD': 'CrowdStrike Holdings',
    'ZS': 'Zscaler Inc.',
    'OKTA': 'Okta Inc.',
    'SNPS': 'Synopsys Inc.',
    'CDNS': 'Cadence Design Systems',
    
    # Communications/Media
    'ROKU': 'Roku Inc.',
    'SPOT': 'Spotify Technology',  # Replacing CMCSA
    'TMUS': 'T-Mobile US Inc.',
    'SIRI': 'Sirius XM Holdings',  # Replacing CHTR
    
    # Entertainment/Tech
    'NFLX': 'Netflix Inc.',
    'TTD': 'The Trade Desk',
    
    # Others
    'MELI': 'MercadoLibre Inc.',
    'ASML': 'ASML Holding NV',
    "AMD": 'Advanced Micro Devices Inc.',
    "TSLA": "Tesla, Inc.",
    "MDB": "MongoDB Inc.",
    "NET": "Cloudflare Inc.",
}



async def fetch_stock_history(ib: IB, symbol: str, years_back: int = 5, exchange: str = 'SMART') -> pd.DataFrame:
    """
    Fetch historical daily market data for a stock from Interactive Brokers.
    
    Parameters:
    ib (IB): Active IB connection instance
    symbol (str): Stock symbol (e.g., 'AAPL')
    years_back (int): Number of years of historical data to fetch
    exchange (str): Exchange to use, defaults to 'SMART' for best execution
    
    Returns:
    pd.DataFrame: DataFrame with columns Date, Open, High, Low, Close, Volume
    """
    try:
        # Create stock contract
        stock = Stock(symbol, exchange, 'USD')
        
        # Calculate start date
        end_date = datetime.now()
        duration = f'{years_back} Y'  # Add 5 days to account for holidays
        
        # Request historical data
        bars = ib.reqHistoricalData(
            contract=stock,
            endDateTime=end_date,
            durationStr=duration,
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        
        if not bars:
            raise ValueError(f"No data received for {symbol}")
            
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
        logging.error(f"Error fetching data for {symbol}: {str(e)}")
        raise

async def fetch_market_indicators(ib: IB, lookback_years: int) -> tuple[pd.DataFrame, int]:
    """
    Fetch market relationship indicators from Yahoo Finance in format ready for multi_indicators.
    Handles missing values using forward fill.
    
    Parameters:
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format, defaults to today if None
    
    Returns:
    tuple[pd.DataFrame, int]: 
        - DataFrame containing market data in long format (Date, Market, Open, High, Low, Close)
        - Number of markets
    """    
    symbols = SPY_100_TOP_50_NASDAQ_ONLY

    tasks = []
    
    # Create tasks for all symbols
    for symbol in symbols.keys():
        tasks.append(fetch_stock_history(ib, symbol, lookback_years))
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    symbol_dfs = {}
    all_dates = set()
    
    for symbol, result in zip(symbols.keys(), results):
        if isinstance(result, Exception):
            logger.error(f"Failed to fetch {symbol}: {result}")
            continue
        if result is not None:
            symbol_dfs[symbol] = result
            all_dates.update(result.index)
            
    if not symbol_dfs:
        raise ValueError("No valid data could be fetched for any symbol")
    
    # Convert to sorted list
    all_dates = sorted(list(all_dates))
    
    # In the second pass where rows are created, add a minimum threshold
    MIN_VALUE = 1e-60

    # Second pass: Create rows with forward fill
    rows = []
    for symbol in symbols.keys():
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
    
    n_markets = len(symbols)
    
    # Verify all markets have the same number of rows
    market_counts = result_df.groupby('Market').size()
    if len(set(market_counts)) > 1:
        print("Warning: Not all markets have the same number of rows after forward fill:")
        print(market_counts)

    return result_df, n_markets

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