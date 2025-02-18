import pandas as pd
import numpy as np
from indicators.Single.comp_var import (
    close_minus_ma, volume_momentum, price_intensity, rsi_exact,macd_enhanced, reactivity,
    linear_trend, change_variance_ratio, price_change_oscillator,
    stochastic_exact, ppo_enhanced, normalized_obv, stochastic_rsi_exact,
    money_flow, aroon_down, aroon_difference, ma_difference
)
from indicators.Mult.comp_var import CompVar
from indicators.Mult.comp_var import (
    VAR_ABS_RATIO, VAR_ABS_SHIFT, VAR_COHERENCE,
    VAR_JANUS_INDEX_MARKET, VAR_JANUS_INDEX_DOM, 
    VAR_JANUS_RAW_RS, VAR_JANUS_FRACTILE_RS, VAR_JANUS_DELTA_FRACTILE_RS,
    VAR_JANUS_RSS, VAR_JANUS_DELTA_RSS,
    VAR_JANUS_DOM, VAR_JANUS_DOE,
    VAR_JANUS_RAW_RM, VAR_JANUS_FRACTILE_RM, VAR_JANUS_DELTA_FRACTILE_RM,
    VAR_JANUS_RS_LEADER_EQUITY, VAR_JANUS_RS_LAGGARD_EQUITY,
    VAR_JANUS_RS_LEADER_ADVANTAGE, VAR_JANUS_RS_LAGGARD_ADVANTAGE,
    VAR_JANUS_RM_LEADER_EQUITY, VAR_JANUS_RM_LAGGARD_EQUITY,
    VAR_JANUS_RM_LEADER_ADVANTAGE, VAR_JANUS_RM_LAGGARD_ADVANTAGE,
    VAR_JANUS_RS_PS, VAR_JANUS_RM_PS,
    VAR_JANUS_CMA_OOS, VAR_JANUS_LEADER_CMA_OOS, VAR_JANUS_OOS_AVG
)

def single_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["V_MOM_10_20"] = volume_momentum(df, short_length=10, mult=20)
    df["PR_INT_20"] = price_intensity(df, n_to_smooth=20)
    df["CMMA_10_252"] = close_minus_ma(df, lookback=10, atr_length=252)
    df["RSI_20"] = rsi_exact(df, lookback=20)
    df["MACD_10_100_5"] = macd_enhanced(df, short_length=10, long_length=100, n_to_smooth=5)
    df["REACT_10_8"] = reactivity(df, lookback=10, multiplier=8)
    df["LINTRND_10"] = linear_trend(df, lookback=10, atr_length=252)
    df["CVR_20_5"] = change_variance_ratio(df, short_length=20, mult=5)
    df["PCO_10_5"] = price_change_oscillator(df, 10, 5)
    df["STO_20_1"] = stochastic_exact(df, lookback=20, n_to_smooth=1)
    df["PPO_10_100_5"] = ppo_enhanced(df, short_length=10, long_length=100, n_to_smooth=5)
    df["NOBV_20"] = normalized_obv(df, 20)
    df["SR_14_14_10"] = stochastic_rsi_exact(df, rsi_lookback=14, stoch_lookback=14, smooth_periods=10)
    df["MF_21"] = money_flow(df, period=21)
    df["AROON_DN_100"] = aroon_down(df, period=100)
    df["AROON_DF_100"] = aroon_difference(df, period=100)
    df["MADIFF_10_100"] = ma_difference(df, param1=10, param2=100, param3=10)
    return df

def multi_indicators(df: pd.DataFrame, n_markets: int) -> tuple[pd.DataFrame, CompVar]:
    """
    Compute multiple-market indicators.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing OHLC data for multiple markets
    n_markets : int
        Number of markets
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added multi-market indicators
    """
    # Get unique dates in correct order
    unique_dates = df['Date'].unique()
    
    # Prepare the arrays
    open_prices, high, low, close, market_names = prepare_market_arrays(df, n_markets)

    # Initialize CompVar
    n_timestamps = len(df) // n_markets
    cv = CompVar(n_timestamps, n_markets)
    
    out_df = pd.DataFrame(index=unique_dates)  # Initialize with dates as index

    # Compute absorption ratio and related indicators
    output, _, _, _ = cv.compute(
        var_num=VAR_ABS_RATIO,
        param1=252,  # 1 year lookback
        param2=0.2,  # Use top 20% of eigenvalues
        param3=0,
        param4=0,
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['ABS_RAT_252'] = output
    
    output, _, _, _ = cv.compute(
        var_num=VAR_ABS_SHIFT,
        param1=252,  # 1 year lookback
        param2=0.2,  # Use top 20% of eigenvalues
        param3=60,   # Long MA window
        param4=20,   # Short MA window
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['ABS_SHIFT'] = output
    
    output, _, _, _ = cv.compute(
        var_num=VAR_COHERENCE,
        param1=252,
        param2=0,
        param3=0,
        param4=0,
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['COHERENCE'] = output
    
    # Janus Index indicators
    output, _, _, _ = cv.compute(
        var_num=VAR_JANUS_INDEX_MARKET,
        param1=252,
        param2=0,
        param3=0,
        param4=0,
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['J_INDEX_MKT'] = output
    
    output, _, _, _ = cv.compute(
        var_num=VAR_JANUS_INDEX_DOM,
        param1=252,
        param2=0,
        param3=0,
        param4=0,
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['J_INDEX_DOM_252'] = output
    
    # Janus RS indicators
    output, _, _, _ = cv.compute(
        var_num=VAR_JANUS_RAW_RS,
        param1=252,
        param2=15,  # Market index
        param3=0,
        param4=0,
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['J_RS_OEX_15'] = output
    
    output, _, _, _ = cv.compute(
        var_num=VAR_JANUS_FRACTILE_RS,
        param1=252,
        param2=15,
        param3=0,
        param4=0,
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['J_FR_RS_OEX_100'] = output
    
    output, _, _, _ = cv.compute(
        var_num=VAR_JANUS_DELTA_FRACTILE_RS,
        param1=252,
        param2=24,
        param3=5,
        param4=0,
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['J_DRS_IBM252_5'] = output
    
    # Janus RSS indicators
    output, _, _, _ = cv.compute(
        var_num=VAR_JANUS_RSS,
        param1=252,
        param2=100,
        param3=0,
        param4=0,
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['J_RSS_100'] = output
    
    output, _, _, _ = cv.compute(
        var_num=VAR_JANUS_DELTA_RSS,
        param1=100,
        param2=0,
        param3=0,
        param4=0,
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['J_D_RSS_100'] = output
    
    # Janus DOM/DOE indicators
    output, _, _, _ = cv.compute(
        var_num=VAR_JANUS_DOM,
        param1=252,
        param2=24,
        param3=0,
        param4=0,
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['J_DOM_IBM_252'] = output
    
    output, _, _, _ = cv.compute(
        var_num=VAR_JANUS_DOE,
        param1=252,
        param2=24,
        param3=0,
        param4=0,
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['J_DOE_IBM_252'] = output
    
    # Janus RM indicators
    output, _, _, _ = cv.compute(
        var_num=VAR_JANUS_RAW_RM,
        param1=252,
        param2=24,
        param3=0,
        param4=0,
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['J_RM_IBM_252'] = output
    
    for i in range(n_markets):
        output, _, _, _ = cv.compute(
            var_num=VAR_JANUS_FRACTILE_RM,
            param1=252,
            param2=i,
            param3=0,
            param4=0,
            open_prices=open_prices,
            high=high,
            low=low,
            close=close
        )
        out_df[f'J_FR_RM_{i}_252'] = output
    
    output, _, _, _ = cv.compute(
        var_num=VAR_JANUS_DELTA_FRACTILE_RM,
        param1=252,
        param2=24,
        param3=5,
        param4=0,
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['J_DRM_IBM252_5'] = output
    
    # Janus RS Leader/Laggard indicators
    output, _, _, _ = cv.compute(
        var_num=VAR_JANUS_RS_LEADER_EQUITY,
        param1=252,
        param2=0,
        param3=0,
        param4=0,
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['J_RS_LEADEQ_252'] = output
    
    output, _, _, _ = cv.compute(
        var_num=VAR_JANUS_RS_LAGGARD_EQUITY,
        param1=252,
        param2=0,
        param3=0,
        param4=0,
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['J_RS_LAGEQ_252'] = output
    
    output, _, _, _ = cv.compute(
        var_num=VAR_JANUS_RS_LEADER_ADVANTAGE,
        param1=252,
        param2=0,
        param3=0,
        param4=0,
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['J_RS_LEADAD_252'] = output
    
    output, _, _, _ = cv.compute(
        var_num=VAR_JANUS_RS_LAGGARD_ADVANTAGE,
        param1=252,
        param2=0,
        param3=0,
        param4=0,
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['J_RS_LAGAD_252'] = output
    
    # Janus RS/RM Performance Spread
    output, _, _, _ = cv.compute(
        var_num=VAR_JANUS_RS_PS,
        param1=252,
        param2=0,
        param3=0,
        param4=0,
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['J_RS_PS_252'] = output
    
    # Janus RM Leader/Laggard indicators
    output, _, _, _ = cv.compute(
        var_num=VAR_JANUS_RM_LEADER_EQUITY,
        param1=252,
        param2=0,
        param3=0,
        param4=0,
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['J_RM_LEADEQ_252'] = output
    
    output, _, _, _ = cv.compute(
        var_num=VAR_JANUS_RM_LAGGARD_EQUITY,
        param1=252,
        param2=0,
        param3=0,
        param4=0,
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['J_RM_LAGEQ_252'] = output
    
    output, _, _, _ = cv.compute(
        var_num=VAR_JANUS_RM_LEADER_ADVANTAGE,
        param1=252,
        param2=0,
        param3=0,
        param4=0,
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['J_RM_LEADAD_252'] = output
    
    output, _, _, _ = cv.compute(
        var_num=VAR_JANUS_RM_LAGGARD_ADVANTAGE,
        param1=252,
        param2=0,
        param3=0,
        param4=0,
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['J_RM_LAGAD_252'] = output
    
    output, _, _, _ = cv.compute(
        var_num=VAR_JANUS_RM_PS,
        param1=252,
        param2=0,
        param3=0,
        param4=0,
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['J_RM_PS_252'] = output
    
    # Janus CMA indicators
    output, _, _, _ = cv.compute(
        var_num=VAR_JANUS_CMA_OOS,
        param1=252,
        param2=0,
        param3=0,
        param4=0,
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['J_CMA_OOS'] = output
    
    output, _, _, _ = cv.compute(
        var_num=VAR_JANUS_LEADER_CMA_OOS,
        param1=252,
        param2=0,
        param3=0,
        param4=0,
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['J_LCMA_OOS'] = output
    
    output, _, _, _ = cv.compute(
        var_num=VAR_JANUS_OOS_AVG,
        param1=252,
        param2=0,
        param3=0,
        param4=0,
        open_prices=open_prices,
        high=high,
        low=low,
        close=close
    )
    out_df['J_OOS_AVG'] = output
    
    return out_df, cv, market_names

def prepare_market_arrays(df: pd.DataFrame, n_markets: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Prepare market data arrays for CompVar computation while preserving market names.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: Date, Market, Open, High, Low, Close
    n_markets : int
        Number of markets
        
    Returns:
    --------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]
        Tuple of (open_prices, high_prices, low_prices, close_prices, market_names) arrays,
        where OHLC arrays have shape (n_markets, n_timestamps) and market_names is a list
        of market identifiers
    """
    # Create pivot tables for OHLC data
    open_matrix = df.pivot(index='Market', columns='Date', values='Open')
    high_matrix = df.pivot(index='Market', columns='Date', values='High')
    low_matrix = df.pivot(index='Market', columns='Date', values='Low')
    close_matrix = df.pivot(index='Market', columns='Date', values='Close')
    
    # Store market names before converting to numpy arrays
    market_names = open_matrix.index.tolist()
    
    # Convert to numpy arrays
    open_array = open_matrix.values
    high_array = high_matrix.values
    low_array = low_matrix.values
    close_array = close_matrix.values
    
    return open_array, high_array, low_array, close_array, market_names
