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

    def generate_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate ML features with proper time series handling"""
        features = df.copy()
        
        # Ensure index is sorted by date
        features = features.sort_index()
        
        for col in df.columns:
            config = self.FRED_SERIES.get(col)
            if not config:
                continue
                
            freq_map = {'d': 1, 'w': 5, 'm': 21, 'q': 63}
            base_period = freq_map.get(config.frequency, 1)
            
            if config.frequency in ['q', 'm']:
                # Calculate PCT_CHG before interpolation
                features[f'{col}_PCT_CHG'] = features[col].pct_change()
                
                # Then interpolate
                filled = features[col].ffill()
                features[col] = filled.interpolate(method='linear')
            
            # Rolling features using adjusted periods
            for period in config.lookback_periods:
                window = period * base_period
                
                # Use original (non-interpolated) data for longer-term features
                features[f'{col}_PCT_CHG_{period}'] = df[col].pct_change(periods=window)
                
                roll_std = features[col].pct_change().rolling(
                    window=window,
                    min_periods=max(2, window//2)
                ).std()
                features[f'{col}_VOL_{period}'] = roll_std
                
                # Z-score with error checking
                roll_mean = features[col].rolling(window=window, min_periods=2).mean()
                roll_std = features[col].rolling(window=window, min_periods=2).std()
                roll_std = roll_std.replace(0, roll_std.median())  # Replace zeros with median
                features[f'{col}_ZSCORE_{period}'] = (features[col] - roll_mean) / roll_std
                
                # Trend calculation with proper scaling
                features[f'{col}_TREND_{period}'] = (
                    (features[col] - features[col].shift(window)) / 
                    (window * base_period * features[col].shift(window))  # Normalize by level
                )
        
        return features

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
            
            print("FRED FEATURES before pre-process")
            print(df)
            
            # Clean features
            features = df.replace([np.inf, -np.inf], np.nan)
            
            # Generate features
            # TODO(See if this is even useful)
            # features = self.generate_ml_features(features)

            
            # Remove lookback period to prevent lookahead bias
            features = features.loc[start_date:]
            
            print("FRED FEATURES")
            print(features)
            
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