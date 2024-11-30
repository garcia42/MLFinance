import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from collections import deque

@dataclass
class RunBar:
    """Container for run bar data"""
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    dollar_volume: float
    tick_count: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    
class RunBarsProcessor:
    """
    Processes raw market data into various types of run bars:
    - Dollar bars
    - Volume bars
    - Tick bars
    
    Implementation based on "Advances in Financial Machine Learning" by Marcos Lopez de Prado
    """
    
    def __init__(self,
                 threshold_type: str = 'dollar',
                 threshold: float = 1_000_000,  # $1M for dollar bars
                 warm_up_bars: int = 100):
        """
        Initialize the run bars processor
        
        Args:
            threshold_type: Type of bars to create ('dollar', 'volume', 'tick')
            threshold: Threshold for bar creation
            warm_up_bars: Number of bars to use for threshold estimation
        """
        self.logger = logging.getLogger(__name__)
        
        if threshold_type not in ['dollar', 'volume', 'tick']:
            raise ValueError("threshold_type must be 'dollar', 'volume', or 'tick'")
            
        self.threshold_type = threshold_type
        self.initial_threshold = threshold
        self.threshold = threshold
        self.warm_up_bars = warm_up_bars
        
        # Initialize containers
        self.current_bar = None
        self.bars: List[RunBar] = []
        self.imbalance_window = deque(maxlen=100)  # For dynamic threshold adjustment
        
    def _initialize_bar(self, price: float, volume: float, timestamp: pd.Timestamp) -> None:
        """Initialize a new bar"""
        self.current_bar = RunBar(
            open_price=price,
            high_price=price,
            low_price=price,
            close_price=price,
            volume=volume,
            dollar_volume=price * volume,
            tick_count=1,
            start_time=timestamp,
            end_time=timestamp
        )
        
    def _update_bar(self, price: float, volume: float, timestamp: pd.Timestamp) -> None:
        """Update current bar with new tick data"""
        if self.current_bar is None:
            self._initialize_bar(price, volume, timestamp)
            return
            
        self.current_bar.high_price = max(self.current_bar.high_price, price)
        self.current_bar.low_price = min(self.current_bar.low_price, price)
        self.current_bar.close_price = price
        self.current_bar.volume += volume
        self.current_bar.dollar_volume += price * volume
        self.current_bar.tick_count += 1
        self.current_bar.end_time = timestamp
        
    def _get_cumulative_value(self) -> float:
        """Get the cumulative value based on threshold type"""
        if self.current_bar is None:
            return 0.0
            
        if self.threshold_type == 'dollar':
            return self.current_bar.dollar_volume
        elif self.threshold_type == 'volume':
            return self.current_bar.volume
        else:  # tick
            return self.current_bar.tick_count
            
    def _calculate_imbalance(self) -> float:
        """Calculate tick imbalance for threshold adjustment"""
        if len(self.bars) < 2:
            return 0.0
            
        prev_price = self.bars[-2].close_price
        curr_price = self.bars[-1].close_price
        return np.sign(curr_price - prev_price)
        
    def _adjust_threshold(self) -> None:
        """Dynamically adjust threshold based on tick imbalance"""
        if len(self.bars) < self.warm_up_bars:
            return
            
        # Calculate current imbalance
        imbalance = self._calculate_imbalance()
        self.imbalance_window.append(imbalance)
        
        # Calculate average absolute imbalance
        abs_imbalance = np.abs(np.mean(self.imbalance_window))
        
        # Adjust threshold
        self.threshold = self.initial_threshold * (1 + abs_imbalance)
        
    def process_tick(self, 
                    price: float, 
                    volume: float, 
                    timestamp: pd.Timestamp) -> Optional[RunBar]:
        """
        Process new tick data and return a completed bar if threshold is reached
        
        Args:
            price: Current tick price
            volume: Current tick volume
            timestamp: Current tick timestamp
            
        Returns:
            Completed RunBar if threshold is reached, None otherwise
        """
        try:
            # Update current bar
            self._update_bar(price, volume, timestamp)
            
            # Check if threshold is reached
            if self._get_cumulative_value() >= self.threshold:
                # Save completed bar
                completed_bar = self.current_bar
                self.bars.append(completed_bar)
                
                # Adjust threshold
                self._adjust_threshold()
                
                # Initialize new bar
                self._initialize_bar(price, volume, timestamp)
                
                return completed_bar
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing tick: {e}")
            raise
            
    def process_market_data(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert market data DataFrame to run bars
        
        Args:
            market_data: DataFrame with columns ['price', 'volume', 'timestamp']
            
        Returns:
            DataFrame of run bars with OHLCV data
        """
        try:
            bars = []
            
            # Process each tick
            for _, row in market_data.iterrows():
                completed_bar = self.process_tick(
                    price=row['price'],
                    volume=row['volume'],
                    timestamp=pd.Timestamp(row.name)
                )
                
                if completed_bar is not None:
                    bars.append(completed_bar)
                    
            # Convert bars to DataFrame
            bars_df = pd.DataFrame([
                {
                    'timestamp': bar.start_time,
                    'end_time': bar.end_time,
                    'open': bar.open_price,
                    'high': bar.high_price,
                    'low': bar.low_price,
                    'close': bar.close_price,
                    'volume': bar.volume,
                    'dollar_volume': bar.dollar_volume,
                    'tick_count': bar.tick_count
                }
                for bar in bars
            ])
            
            if not bars_df.empty:
                bars_df.set_index('timestamp', inplace=True)
                
            return bars_df
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            raise
