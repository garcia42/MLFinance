import math
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List, Union
import calc_stats

MIN_LIQUIDITY = 1250000 # 1.25mil

def calculate_position_size(
    capital: float,
    capped_forecast: float,
    multiplier: float,
    price: float,
    target_risk_pct: float = 20.0,
    price_series: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
    idm: float = 1.0,
    weight: float = 1.0,
    debug: bool = True
) -> dict:
    """
    Fixed position sizing calculation with debugging
    """

    if debug:
        print(f"\n=== DEBUG: Position Size Calculation ===")
        print(f"Input parameters:")
        print(f"  capital: ${capital:,.2f}")
        print(f"  capped_forecast: {capped_forecast}")
        print(f"  multiplier: {multiplier}")
        print(f"  price: {price}")
        print(f"  target_risk_pct: {target_risk_pct}%")
        print(f"  idm: {idm}")
        print(f"  weight: {weight}")

    # Validate inputs
    if capital <= 0:
        raise ValueError(f"Capital must be positive, got {capital}")
    if multiplier <= 0:
        raise ValueError(f"Multiplier must be positive, got {multiplier}")
    if price <= 0:
        raise ValueError(f"Price must be positive, got {price}")
    if abs(capped_forecast) > 50:  # Sanity check
        print(f"WARNING: Very large forecast value: {capped_forecast}")

    # Calculate volatility properly
    if price_series is not None:
        price_series = np.array(price_series)
        # Remove any NaN or zero values
        valid_prices = price_series[~np.isnan(price_series)]
        valid_prices = valid_prices[valid_prices > 0]

        if len(valid_prices) < 2:
            raise ValueError(f"Insufficient price data: {len(valid_prices)} valid prices")

        # Calculate log returns
        returns = np.diff(np.log(valid_prices))
        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            raise ValueError("No valid returns calculated")

        daily_vol = np.std(returns)
        annualized_vol = daily_vol * np.sqrt(252)

        if debug:
            print(f"  price_series length: {len(price_series)}")
            print(f"  valid prices: {len(valid_prices)}")
            print(f"  returns calculated: {len(returns)}")
            print(f"  daily volatility: {daily_vol:.6f}")
            print(f"  annualized volatility: {annualized_vol:.6f} ({annualized_vol*100:.2f}%)")
    else:
        raise ValueError("Must provide price_series")

    # Convert target risk to decimal
    tau = target_risk_pct / 100.0

    if debug:
        print(f"  tau (target risk): {tau:.3f}")

    # Fixed formula - THE ISSUE IS HERE!
    # The original formula in systematic trading uses forecast/10 scaling
    # N = (forecast × capital × IDM × weight × τ) ÷ (10 × multiplier × price × σ)

    numerator = capped_forecast * capital * idm * weight * tau
    denominator = 10 * multiplier * price * annualized_vol  # ← RESTORE THE 10!

    if debug:
        print(f"\nFormula components:")
        print(f"  numerator = |{capped_forecast}| × {capital} × {idm} × {weight} × {tau}")
        print(f"  numerator = {numerator:,.2f}")
        print(f"  denominator = 10 × {multiplier} × {price} × {annualized_vol}")
        print(f"  denominator = {denominator:,.2f}")

    if denominator == 0:
        raise ValueError("Denominator is zero - check multiplier, price, and volatility")

    contracts_needed = numerator / denominator

    if debug:
        print(f"  contracts_needed (raw): {contracts_needed:.6f}")

    # Apply forecast direction
    if capped_forecast < 0:
        contracts_needed = -contracts_needed

    contracts_rounded = round(contracts_needed)

    if debug:
        print(f"  contracts_needed (with direction): {contracts_needed:.6f}")
        print(f"  contracts_rounded: {contracts_rounded}")

    # Calculate metrics
    notional_exposure = abs(contracts_rounded) * multiplier * price
    leverage_ratio = notional_exposure / capital if capital > 0 else 0

    if debug:
        print(f"  notional_exposure: ${notional_exposure:,.2f}")
        print(f"  leverage_ratio: {leverage_ratio:.4f}")
        print(f"=== END DEBUG ===\n")

    return {
        'contracts_needed': contracts_needed,
        'contracts_rounded': contracts_rounded,
        'capped_forecast': capped_forecast,
        'leverage_ratio': leverage_ratio,
        'notional_exposure': notional_exposure,
        'annualized_volatility_pct': annualized_vol * 100,
        'position_direction': 'LONG' if contracts_rounded > 0 else 'SHORT' if contracts_rounded < 0 else 'FLAT',
        # Add debug info
        'debug_info': {
            'numerator': numerator,
            'denominator': denominator,
            'daily_vol': daily_vol,
            'tau': tau,
            'valid_price_count': len(valid_prices) if price_series is not None else 0
        }
    }

def calculate_daily_volume_usd_risk(fx_rate: float,
                                    avg_daily_volume_contracts: float,
                                    volatility: float,
                                    price: float,
                                    multiplier: float) -> float:
    """
    Calculate average daily volume in USD risk.

    Formula: FX rate x Average daily volume x σₛ x Price x Multiplier

    Args:
        fx_rate: Foreign exchange rate to USD
        avg_daily_volume_contracts: Average daily volume in number of contracts
        volatility: Volatility (σₛ) of the underlying
        price: Current price of the contract
        multiplier: Contract multiplier

    Returns:
        Average daily volume in USD risk
    """
    return fx_rate * avg_daily_volume_contracts * volatility * price * multiplier


def calculate_portfolio_std(weights, std_devs, correlation_matrix):
    """
    Calculate portfolio standard deviation.

    Parameters:
    weights (list): Portfolio weights (must sum to 1.0)
    std_devs (list): Standard deviations for each asset
    correlation_matrix (2D list or array): Correlation matrix between assets

    Returns:
    float: Portfolio standard deviation

    Example:
    >>> weights = [0.6, 0.4]
    >>> std_devs = [0.15, 0.08]
    >>> corr_matrix = [[1.0, 0.3], [0.3, 1.0]]
    >>> calculate_portfolio_std(weights, std_devs, corr_matrix)
    0.1158
    """
    weights = np.array(weights)
    std_devs = np.array(std_devs)
    correlation_matrix = np.array(correlation_matrix)

    # Create covariance matrix
    cov_matrix = np.outer(std_devs, std_devs) * correlation_matrix

    # Calculate portfolio variance: w^T * Cov * w
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))

    # Return standard deviation
    return np.sqrt(portfolio_variance)

def get_idm(num_instruments):
    """
    Calculate the Instrument Diversification Multiplier (IDM) based on number of instruments.

    This function provides an approximation of IDM values as shown in Table 16.
    For values not in the lookup table, it uses interpolation or approximation.

    Args:
        num_instruments (int): Number of instruments in the portfolio

    Returns:
        float: IDM value

    Note:
        - Table assumes instrument set is relatively diversified
        - Won't be valid if you have 30 instruments, all of which are equity futures markets
    """

    # Direct lookup table based on Table 16
    idm_lookup = {
        1: 1.00,
        2: 1.20,
        3: 1.48,
        4: 1.56,
        5: 1.70,
        6: 1.90,
        7: 2.10
    }

    # Handle exact matches from the table
    if num_instruments in idm_lookup:
        return idm_lookup[num_instruments]

    # Handle ranges from the table
    if 8 <= num_instruments <= 14:
        return 2.20
    elif 15 <= num_instruments <= 24:
        return 2.30
    elif 25 <= num_instruments <= 29:
        return 2.40
    elif num_instruments >= 30:
        return 2.50

    # For edge cases (should not happen with positive integers)
    if num_instruments < 1:
        raise ValueError("Number of instruments must be at least 1")

    return 1.00  # Default fallback