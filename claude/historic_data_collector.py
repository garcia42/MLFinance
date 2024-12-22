from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytz
import logging
from dataclasses import dataclass
import praw
import fredapi
from ib_insync import Contract, BarData, IB
from fred_collector import create_fred_collector
import warnings
import time
from continuous_futures_contract import ContinuousFuturesContract
from market_indicator_data import fetch_market_indicators, generate_summary_stats
from feature_storage import FeatureStorage

warnings.simplefilter(action='ignore', category=FutureWarning)

@dataclass
class HistoricalSentiment:
    """Container for historical sentiment data"""
    date: datetime
    reddit_sentiment: float
    twitter_sentiment: float
    reddit_volume: int
    twitter_volume: int
    combined_score: float
    top_reddit_posts: List[str]
    top_tweets: List[str]

class IntegratedHistoricalCollector:
    """Collects and integrates historical market, macro, and sentiment data"""
    
    def __init__(
        self,
        ib_connection: IB,
        fred_api: fredapi.Fred,
        reddit_client: praw.Reddit,
        lookback_days: int,
        bar_size: str,
        timezone: str,
        what_to_show: str
    ):
        self.fred = fred_api
        self.ib = ib_connection
        self.reddit = reddit_client
        self.lookback_days = lookback_days
        self.logger = logging.getLogger(__name__)
        self.bar_size = bar_size
        self.timezone = timezone
        self.what_to_show = what_to_show

    def _get_trading_session(self, timestamp: pd.Timestamp) -> str:
        """
        Determine trading session for a given timestamp
        
        Args:
            timestamp: Time to check
            
        Returns:
            Trading session identifier
        """
        ts_eastern: pd.Timestamp = timestamp.astimezone(pytz.timezone('US/Eastern'))
        hour: int = ts_eastern.hour
        minute: int = ts_eastern.minute
        
        if 9 <= hour < 16:
            return 'RTH' if (hour > 9 or (hour == 9 and minute >= 30)) else 'PRE'
        elif 4 <= hour < 9:
            return 'EURO'
        elif 16 <= hour < 20:
            return 'POST'
        else:
            return 'ASIA'
            
    def _convert_bar_to_dict(self, sec_bar: BarData) -> Dict[str, Union[float, int, pd.Timestamp]]:
        """
        Convert IB bar data to dictionary
        
        Args:
            bar: Interactive Brokers bar data
            
        Returns:
            Dictionary containing bar data
        """
        return {
            'date': pd.Timestamp(sec_bar.date),
            'open': float(sec_bar.open),
            'high': float(sec_bar.high),
            'low': float(sec_bar.low),
            'close': float(sec_bar.close),
            'volume': int(sec_bar.volume),
            'average': float(sec_bar.average),
            'barCount': int(sec_bar.barCount)
        }
        
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for market data
        
        Args:
            df: Market data DataFrame
            
        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()
        
        # Price-based indicators
        df['dollar_volume'] = df['close'] * df['volume']
        df['high_low_range'] = df['high'] - df['low']
        df['bar_return'] = df['close'].pct_change()
        df['vwap'] = df['dollar_volume'].cumsum() / df['volume'].cumsum()
        
        # Moving averages
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        
        # Volatility
        df['volatility'] = df['bar_return'].rolling(window=20).std()
        
        return df

    async def collect_all_historical_data(self, contract: Any) -> pd.DataFrame:
        """
        Collect and integrate all historical data
        
        Returns:
            DataFrame with market, macro, and sentiment data aligned
        """
        try:
            # Collect market data
            continuous = ContinuousFuturesContract(self.ib)
            market_data = await continuous.get_continuous_contract(underlying="ES", exchange="CME", lookback_days=self.lookback_days)

            # Convert market data index to UTC and then remove timezone
            market_data.index = market_data.index.tz_convert('UTC').tz_localize(None)

            # Align all data to market data timestamps
            aligned_data = market_data.copy()

            fc = create_fred_collector(
                fred_api=self.fred,
                lookback_days=self.lookback_days
            )

            # Collect macro data
            macro_data: pd.DataFrame = fc.get_historical_fred_data()
            # Add macro features
            if not macro_data.empty:
                macro_features = macro_data.reindex(aligned_data.index, method='ffill')
                aligned_data = pd.concat([aligned_data, macro_features], axis=1)

            # Fetch the data
            market_indicators_data = fetch_market_indicators(datetime.now() -  timedelta(days=self.lookback_days))
            market_indicators_data = market_indicators_data.reindex(aligned_data.index, method='ffill')
            aligned_data = pd.concat([aligned_data, market_indicators_data], axis=1)
                
            return aligned_data
            
        except Exception as e:
            self.logger.error(f"Error collecting integrated historical data: {e}")
            raise

def create_data_collector(
    ib_connection: IB,
    fred_api: fredapi.Fred,
    reddit: praw.Reddit,
    contract: Contract,
    lookback_days: int = 1 * 1, #TODO start with 1 day
    bar_size: str = '5 min',
    timezone: str = 'US/Eastern'
) -> IntegratedHistoricalCollector:
    """
    Factory function to create a historical data collector
    
    Args:
        ib_connection: Interactive Brokers connection
        fred_api: FRED API connection
        lookback_days: Number of days to look back
        bar_size: Size of bars to collect
        timezone: Timezone for data collection
        
    Returns:
        Configured HistoricalDataCollector instance
    """
    
    # Initialize collector
    return IntegratedHistoricalCollector(
        ib_connection=ib_connection,
        fred_api=fred_api,
        reddit_client=reddit,
        lookback_days=lookback_days,
        bar_size=bar_size,
        timezone=timezone,
        what_to_show='TRADES'
    )
