import time
from typing import Dict, List, Optional, Tuple, Union, Any, Mapping
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from fredapi import Fred
import logging
from dataclasses import dataclass

@dataclass
class FredSeriesConfig:
    """Configuration for FRED data series"""
    series_id: str
    frequency: str  # 'd' for daily, 'w' for weekly, 'm' for monthly, 'q' for quarterly
    lookback_periods: List[int]  # Periods for rolling features
    seasonal_adjustment: bool = True

class FredDataCollector:
    """Handles collection and processing of FRED economic data"""
    
    # Dictionary mapping series names to their configurations
    FRED_SERIES: Mapping[str, FredSeriesConfig] = {
        'GDP': FredSeriesConfig(
            series_id='GDP',
            frequency='q',
            lookback_periods=[4],  # 1Q, 1Y, 2Y lookbacks
            seasonal_adjustment=True
        ),
        'UNRATE': FredSeriesConfig(
            series_id='UNRATE',
            frequency='m',
            lookback_periods=[12],  # 1M, 3M, 6M, 1Y lookbacks
            seasonal_adjustment=True
        ),
        'CPIAUCSL': FredSeriesConfig(
            series_id='CPIAUCSL',
            frequency='m',
            lookback_periods=[12],
            seasonal_adjustment=True
        ),
        'FEDFUNDS': FredSeriesConfig(
            series_id='FEDFUNDS',
            frequency='m',
            lookback_periods=[12],
            seasonal_adjustment=False
        ),
        'T10Y2Y': FredSeriesConfig(
            series_id='T10Y2Y',
            frequency='d',
            lookback_periods=[21],  # 1W, 1M, 3M, 1Y lookbacks
            seasonal_adjustment=False
        ),
        'BAMLH0A0HYM2': FredSeriesConfig( # ICE BofA US High Yield Index Option-Adjusted Spread
            series_id='BAMLH0A0HYM2',
            frequency='d',
            lookback_periods=[21],  # 1W, 1M, 3M, 1Y lookbacks
            seasonal_adjustment=False
        ),
        'WM2NS': FredSeriesConfig( # M2
            series_id='WM2NS',
            frequency='w',
            lookback_periods=[4],
            seasonal_adjustment=True
        ),
        'INDPRO': FredSeriesConfig( # Industrial Production: Total Index
            series_id='INDPRO',
            frequency='m',
            lookback_periods=[3],
            seasonal_adjustment=True
        ),
        'HOUST': FredSeriesConfig( # New Privately-Owned Housing Units Started: Total Units
            series_id='HOUST',
            frequency='m',
            lookback_periods=[4],
            seasonal_adjustment=True
        ),
        'RSAFS': FredSeriesConfig( # Advance Retail Sales: Retail Trade and Food Services
            series_id='RSAFS',
            frequency='m',
            lookback_periods=[4],
            seasonal_adjustment=True
        )
    }

    def __init__(self, fred_api: Fred, lookback_days: int) -> None:
        """
        Initialize FRED data collector
        
        Args:
            fred_api: FRED API connection
            lookback_days: Number of days to look back
        """
        self.fred = fred_api
        self.lookback_days = lookback_days
        self.logger: logging.Logger = logging.getLogger(__name__)

    def get_fred_series(self, series_config: FredSeriesConfig, start_date: datetime, end_date: datetime) -> Optional[pd.Series]:
        """Get single FRED series with proper ML dataset handling"""
        try:
            # Add extra lookback period to allow for feature calculation
            extended_start = start_date - timedelta(days=365*2)  # 2 years extra for lookback
            
            kwargs = {
                'series_id': series_config.series_id,
                'observation_start': extended_start.strftime('%Y-%m-%d'),
                'observation_end': end_date.strftime('%Y-%m-%d'),
                'frequency': series_config.frequency,
            }
            
            series = self.fred.get_series(**kwargs)
            print(series_config.series_id, series)
            return series
            
        except Exception as e:
            self.logger.error(f"Error fetching {series_config.series_id}: {e}")
            return None

    def get_historical_fred_data(self) -> pd.DataFrame:
        """
        Get FRED data processed for ML use
        
        This data will have NANs so that it can be f-filled to the market data.
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)
            
            self.logger.info(f"Requesting historical FRED data from {start_date} to {end_date} {self.lookback_days} D")
            # Collect raw data
            series_data: Dict[str, pd.Series] = {}
            for name, config in self.FRED_SERIES.items():
                series = self.get_fred_series(config, start_date, end_date)
                if series is not None:
                    series_data[name] = series
            
            if not series_data:
                return pd.DataFrame()
                
            # Create base dataframe
            df = pd.DataFrame(series_data)
            
            # Clean features
            features = df.replace([np.inf, -np.inf], np.nan)
            
            # Remove lookback period to prevent lookahead bias
            features = features.loc[start_date:]
            
            features = features.copy()  # Create explicit copy
            features = features.fillna(method='ffill', axis=0)
            features = features.interpolate(method='time', axis=0)
            features = features.fillna(method='bfill', axis=0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing ML features: {e}")
            raise

def create_fred_collector(fred_api: Fred, lookback_days: int = 30) -> FredDataCollector:
    """
    Factory function to create a FRED data collector
    
    Args:
        fred_api: FRED API connection
        lookback_days: Number of days to look back
        
    Returns:
        Configured FredDataCollector instance
    """
    return FredDataCollector(fred_api, lookback_days)