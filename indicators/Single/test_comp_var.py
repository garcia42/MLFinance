import pytest
import numpy as np
import pandas as pd
from comp_var import price_intensity, macd_enhanced

def test_price_intensity():
    """Test price_intensity function with known input data"""
    # Create test data
    data = pd.DataFrame({
        'open': [10.0, 11.0, 10.5, 10.8, 11.2],
        'high': [11.0, 11.5, 11.0, 11.2, 11.8],
        'low':  [9.8,  10.8, 10.2, 10.5, 11.0],
        'close': [10.5, 11.2, 10.8, 11.0, 11.5],
        'volume': [1000, 1200, 800, 950, 1100]
    })
    
    # Test with default smoothing (n_to_smooth=1)
    result = price_intensity(data)
    assert len(result) == len(data)
    assert isinstance(result, pd.Series)
    assert -50 <= result.min() <= result.max() <= 50  # Output should be scaled to [-50, 50]
    
    # Test with increased smoothing
    result_smooth = price_intensity(data, n_to_smooth=3)
    assert len(result_smooth) == len(data)
    # Smoothed version should have less extreme values
    assert abs(result_smooth).mean() <= abs(result).mean()
    
    # Test specific case where close = open (should give zero intensity)
    zero_case = pd.DataFrame({
        'open': [10.0],
        'high': [11.0],
        'low': [9.0],
        'close': [10.0],
        'volume': [1000]
    })
    result_zero = price_intensity(zero_case)
    assert abs(result_zero[0]) < 1e-10  # Should be very close to zero
    
    # Test extreme case
    extreme_case = pd.DataFrame({
        'open': [10.0],
        'high': [15.0],
        'low': [10.0],
        'close': [15.0],
        'volume': [1000]
    })
    result_extreme = price_intensity(extreme_case)
    assert result_extreme[0] > 0  # Should be strongly positive

def test_macd_enhanced():
    """Test macd_enhanced function with known input data"""
    # Create test data with a clear trend followed by reversal
    prices = np.array([10.0, 10.5, 11.0, 11.8, 12.5, 13.0, 12.8, 12.3, 11.8, 11.0])
    data = pd.DataFrame({
        'open':  prices - 0.2,
        'high':  prices + 0.3,
        'low':   prices - 0.3,
        'close': prices
    })
    
    # Test with default parameters
    result = macd_enhanced(data, short_length=3, long_length=6)
    assert len(result) == len(data)
    assert isinstance(result, pd.Series)
    assert -50 <= result.min() <= result.max() <= 50  # Output should be scaled to [-50, 50]
    
    # Test trend detection
    # During uptrend (first half), MACD should be positive
    uptrend_signal = result[5]  # Check at peak
    assert uptrend_signal > 0, "MACD should be positive during uptrend"
    
    # During downtrend (second half), MACD should be negative
    downtrend_signal = result[-1]  # Check at end
    assert downtrend_signal < 0, "MACD should be negative during downtrend"
    
    # Test with smoothing
    result_smooth = macd_enhanced(data, short_length=3, long_length=6, n_to_smooth=3)
    assert len(result_smooth) == len(data)
    # Smoothed version should have less extreme values
    assert abs(result_smooth).mean() <= abs(result).mean()
    
    # Test with flat prices (should give near-zero signal)
    flat_data = pd.DataFrame({
        'open':  [10.0] * 10,
        'high':  [10.3] * 10,
        'low':   [9.7] * 10,
        'close': [10.0] * 10
    })
    flat_result = macd_enhanced(flat_data, short_length=3, long_length=6)
    assert abs(flat_result.iloc[-1]) < 5.0  # Should be close to zero

if __name__ == "__main__":
    pytest.main([__file__])
