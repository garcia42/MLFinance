import pandas as pd
from datetime import timezone

def convert_data_daily(data: pd.DataFrame) -> pd.DataFrame:
    # Convert the results to a DataFrame
    df = pd.DataFrame(data)

    # Convert the timestamp to a datetime object
    df['datetime'] = pd.to_datetime(df['t'], unit='ms')

    # Group by date
    df['date'] = df['datetime'].dt.date
    grouped = df.groupby('date').agg({
        'Volume': 'sum',
        'Open': 'first',
        'Close': 'last',
        'High': 'max',
        'Low': 'min'
    }).reset_index()
    df['t'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df.set_index(pd.DatetimeIndex(df['t']), drop=False, inplace=True)
    df.index = df.index.tz_localize(timezone.utc)

    return grouped

def convert_data_weekly(data: pd.DataFrame) -> pd.DataFrame:
    # Convert the results to a DataFrame
    df = pd.DataFrame(data)

    # Convert the timestamp to a datetime object
    df['datetime'] = pd.to_datetime(df['t'], unit='ms')

    # Group by week
    df['week'] = df['datetime'].dt.to_period('W').apply(lambda r: r.start_time)

    grouped = df.groupby('week').agg({
        'Volume': 'sum',
        'Open': 'first',
        'Close': 'last',
        'High': 'max',
        'Low': 'min'
    }).reset_index()

    return grouped