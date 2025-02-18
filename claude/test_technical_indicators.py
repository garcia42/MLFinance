"""Tests for technical indicators module.

These tests verify the functionality of both Single and Mult indicator implementations.
The two implementations should be tested separately and not mixed in calculations.
"""

import pytest
import pandas as pd
import numpy as np
from .technical_indicators import add_technical_indicators, IndicatorType

@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    n = 300  # Enough data points for the longest lookback (252)
    dates = pd.date_range('2020-01-01', periods=n)
    
    # Generate sample price data with some trend and volatility
    np.random.seed(42)
    close = 100 + np.random.randn(n).cumsum()
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    open_price = close + np.random.randn(n)
    volume = 1000000 + np.random.randn(n) * 100000
    
    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

def test_add_technical_indicators_single(sample_data):
    """Test computing technical indicators using Single implementation."""
    df = add_technical_indicators(sample_data, indicator_type="single")
    
    # Check that all indicators were added
    assert 'V_MOM_10_20' in df.columns
    assert 'PR_INT_20' in df.columns
    assert 'CMMA_10_252' in df.columns
    assert 'RSI_20' in df.columns
    
    # Check that indicators have valid values (Single implementation)
    for col in ['V_MOM_10_20', 'PR_INT_20', 'CMMA_10_252', 'RSI_20']:
        values = df[col].dropna()
        assert len(values) > 0
        assert not np.any(np.isnan(values))
        assert not np.any(np.isinf(values))
    
    # RSI should be between 0 and 100
    assert df['RSI_20'].dropna().between(0, 100).all()
    
    # First values should be NaN due to lookback periods
    assert pd.isna(df['V_MOM_10_20'].iloc[0])
    assert pd.isna(df['PR_INT_20'].iloc[0])
    assert pd.isna(df['CMMA_10_252'].iloc[0])
    assert pd.isna(df['RSI_20'].iloc[0])
    
    # Later values should not be NaN
    assert not pd.isna(df['V_MOM_10_20'].iloc[-1])
    assert not pd.isna(df['PR_INT_20'].iloc[-1])
    assert not pd.isna(df['CMMA_10_252'].iloc[-1])
    assert not pd.isna(df['RSI_20'].iloc[-1])

def test_add_technical_indicators_mult(sample_data):
    """Test computing technical indicators using Mult implementation."""
    df = add_technical_indicators(sample_data, indicator_type="mult")
    
    # Check that all indicators were added
    assert 'V_MOM_10_20' in df.columns
    assert 'PR_INT_20' in df.columns
    assert 'CMMA_10_252' in df.columns
    assert 'RSI_20' in df.columns
    
    # Check that indicators have valid values (Mult implementation)
    for col in ['V_MOM_10_20', 'PR_INT_20', 'CMMA_10_252', 'RSI_20']:
        values = df[col].dropna()
        assert len(values) > 0
        assert not np.any(np.isnan(values))
        assert not np.any(np.isinf(values))
    
    # RSI should be between 0 and 100
    assert df['RSI_20'].dropna().between(0, 100).all()
    
    # First values should be NaN due to lookback periods
    assert pd.isna(df['V_MOM_10_20'].iloc[0])
    assert pd.isna(df['PR_INT_20'].iloc[0])
    assert pd.isna(df['CMMA_10_252'].iloc[0])
    assert pd.isna(df['RSI_20'].iloc[0])
    
    # Later values should not be NaN
    assert not pd.isna(df['V_MOM_10_20'].iloc[-1])
    assert not pd.isna(df['PR_INT_20'].iloc[-1])
    assert not pd.isna(df['CMMA_10_252'].iloc[-1])
    assert not pd.isna(df['RSI_20'].iloc[-1])

def test_invalid_indicator_type(sample_data):
    """Test that invalid indicator type raises ValueError."""
    with pytest.raises(ValueError, match='indicator_type must be either "single" or "mult"'):
        add_technical_indicators(sample_data, indicator_type="invalid")

