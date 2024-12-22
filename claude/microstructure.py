"""
This module processes market microstructure data from E-mini S&P 500 futures (ES) tick data.

Market microstructure is crucial in financial machine learning models because:
1. It captures the fine-grained dynamics of price formation and market behavior
2. Provides insights into market liquidity, trading costs, and price impact
3. Helps detect informed trading and market manipulation
4. Essential for high-frequency trading strategies and market making
5. Enables more accurate modeling of transaction costs and market friction

The ES futures tick data is particularly valuable because:
- E-mini S&P 500 is one of the most liquid and widely traded futures contracts
- High-frequency data captures microstructure effects that are lost in lower frequencies
- Institutional traders heavily use ES futures, making it representative of sophisticated market behavior
"""

import pandas as pd
import numpy as np
from pathlib import Path

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def load_es_tick_data():
    """
    Load and preprocess ES futures tick data.
    
    Returns:
        pd.DataFrame: Processed tick data with columns:
            - timestamp: Time of the tick
            - price: Transaction price
            - volume: Transaction volume
            - side: Trade direction (1 for buy, -1 for sell)
            
    Raises:
        FileNotFoundError: If the tick data file doesn't exist
    """
    file_path = Path('/Users/garciaj42/code/MLFinance/claude/ES_tick/ES_processed_final.csv')
    if not file_path.exists():
        raise FileNotFoundError(f"Tick data file not found at {file_path}")
        
    # Read the processed ES tick data
    df = pd.read_csv(file_path)
    
    # Ensure timestamp is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df

def calculate_microstructure_features(tick_data: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """
    Calculate market microstructure features from tick data.
    
    Based on week 10
    - https://github.com/QuantifiSogang/2024-02SeniorMLFinance/blob/main/Notes/Week10MicrostructuralFeatures/03SequentialTradeModels.ipynb
    - Eventually incorporate additional features
        - https://github.com/QuantifiSogang/2024-02SeniorMLFinance/blob/main/Notes/Week10MicrostructuralFeatures/04AdditionalFeatures.ipynb

    Args:
        tick_data (pd.DataFrame): Input tick data
        window (int): Rolling window size for calculations
        
    Returns:
        pd.DataFrame: Microstructure features including:
            - volume_imbalance: Buy volume - Sell volume ratio
            - tick_rule: Trade initiation (buyer vs seller initiated)
            - effective_spread: Cost of immediate execution
            - price_impact: Price change per unit of trade size
    """
    features = pd.DataFrame(index=tick_data.index)
    
    # Volume imbalance (buy volume - sell volume) / total volume
    buy_volume = tick_data['volume'] * (tick_data['side'] == 1)
    sell_volume = tick_data['volume'] * (tick_data['side'] == -1)
    features['volume_imbalance'] = (
        (buy_volume - sell_volume).rolling(window).sum() / 
        (buy_volume + sell_volume).rolling(window).sum()
    )
    
    # Tick rule (trade initiation)
    price_changes = tick_data['price'].diff()
    features['tick_rule'] = np.sign(price_changes)
    # Fill first value with 0 since we can't determine direction
    features['tick_rule'].iloc[0] = 0
    # Fill zero price changes with previous tick rule
    mask = price_changes == 0
    features.loc[mask, 'tick_rule'] = features['tick_rule'].shift(1)[mask]
    
    # Effective spread
    midpoint = (tick_data['price'].shift(1) + tick_data['price']) / 2
    features['effective_spread'] = 2 * np.abs(tick_data['price'] - midpoint)
    
    # Price impact
    features['price_impact'] = (
        (tick_data['price'] - tick_data['price'].shift(window)) / 
        tick_data['price'].shift(window) / 
        tick_data['volume']
    )
    
    return features.fillna(0)

if __name__ == '__main__':
    try:
        # Example usage
        tick_data = load_es_tick_data()
        features = calculate_microstructure_features(tick_data)
        print("Microstructure features shape:", features.shape)
        print("\nFeature descriptions:")
        print("- volume_imbalance: Measures buying vs selling pressure")
        print("- tick_rule: Indicates trade initiation (buyer vs seller initiated)")
        print("- effective_spread: Captures the true cost of trading")
        print("- price_impact: Measures price sensitivity to trade size")
    except Exception as e:
        print(f"Error processing tick data: {e}")
