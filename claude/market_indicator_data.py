import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests

FUTURES  = {
    # Equity Index Futures
    'ES=F': 'E-mini S&P 500',
    'NQ=F': 'E-mini NASDAQ 100',
    'RTY=F': 'E-mini Russell 2000',
    'YM=F': 'E-mini Dow',

    # Currency Futures
    '6E=F': 'Euro FX',
    '6J=F': 'Japanese Yen',
    '6B=F': 'British Pound',
    '6A=F': 'Australian Dollar',
    '6C=F': 'Canadian Dollar',
    '6M=F': 'Mexican Peso',
    '6N=F': 'New Zealand Dollar',
    '6S=F': 'Swiss Franc',

    # Metal Futures
    'GC=F': 'Gold',
    'SI=F': 'Silver',
    'HG=F': 'Copper',
    'PL=F': 'Platinum',
    'PA=F': 'Palladium',

    # Energy Futures
    'CL=F': 'Crude Oil WTI',
    'BZ=F': 'Brent Crude',
    'NG=F': 'Natural Gas',
    'HO=F': 'Heating Oil',
    'RB=F': 'RBOB Gasoline',

    # Interest Rate Futures
    'ZN=F': '10-Year T-Note',
    'ZT=F': '2-Year T-Note',
    'ZF=F': '5-Year T-Note',
    'ZB=F': '30-Year T-Bond',
}

SPY_100_TOP_50 = {
    # Technology
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'NVDA': 'NVIDIA Corporation',
    'GOOGL': 'Alphabet Inc. Class A',
    'GOOG': 'Alphabet Inc. Class C',
    'META': 'Meta Platforms Inc.',
    'AVGO': 'Broadcom Inc.',
    'ADBE': 'Adobe Inc.',
    'CRM': 'Salesforce Inc.',
    'CSCO': 'Cisco Systems Inc.',
    'ACN': 'Accenture plc',
    'ORCL': 'Oracle Corporation',
    
    # Financial Services
    'BRK.B': 'Berkshire Hathaway Inc.',
    'JPM': 'JPMorgan Chase & Co.',
    'V': 'Visa Inc.',
    'MA': 'Mastercard Incorporated',
    'BAC': 'Bank of America Corp.',
    'WFC': 'Wells Fargo & Company',
    'MS': 'Morgan Stanley',
    'GS': 'Goldman Sachs Group Inc.',
    
    # Healthcare
    'UNH': 'UnitedHealth Group Inc.',
    'JNJ': 'Johnson & Johnson',
    'LLY': 'Eli Lilly and Company',
    'ABBV': 'AbbVie Inc.',
    'MRK': 'Merck & Co. Inc.',
    'PFE': 'Pfizer Inc.',
    'TMO': 'Thermo Fisher Scientific Inc.',
    'ABT': 'Abbott Laboratories',
    
    # Consumer
    'AMZN': 'Amazon.com Inc.',
    'WMT': 'Walmart Inc.',
    'PG': 'Procter & Gamble Company',
    'COST': 'Costco Wholesale Corporation',
    'KO': 'The Coca-Cola Company',
    'PEP': 'PepsiCo Inc.',
    'MCD': "McDonald's Corporation",
    'NKE': 'Nike Inc.',
    
    # Industrial
    'XOM': 'Exxon Mobil Corporation',
    'CVX': 'Chevron Corporation',
    'UPS': 'United Parcel Service Inc.',
    'BA': 'The Boeing Company',
    'CAT': 'Caterpillar Inc.',
    'GE': 'General Electric Company',
    'HON': 'Honeywell International Inc.',
    'UNP': 'Union Pacific Corporation',
    
    # Telecommunications
    'T': 'AT&T Inc.',
    'VZ': 'Verizon Communications Inc.',
    
    # Entertainment
    'DIS': 'The Walt Disney Company',
    'NFLX': 'Netflix Inc.',
    
    # Others
    'BLK': 'BlackRock Inc.',
    'AXP': 'American Express Company',
    'AMD': 'Advanced Micro Devices Inc.'
}

def fetch_market_indicators(start_date: datetime, end_date=None) -> tuple[pd.DataFrame, int]:
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
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    symbols = SPY_100_TOP_50

    # Dictionary to store DataFrames for each symbol
    symbol_dfs = {}
    
    # First pass: Collect all unique dates
    all_dates = set()
    for symbol, description in symbols.items():
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if not df.empty:
                symbol_dfs[symbol] = df
                all_dates.update(df.index)
                print(f"Successfully fetched data for {description} ({symbol})")
            else:
                print(f"No data available for {description} ({symbol})")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
    
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