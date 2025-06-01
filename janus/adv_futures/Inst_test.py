from gold_eval import InstrumentSelectionAlgorithm, get_contract_price_in_dollars
import position_size
import calc_table
import costs_stats
import commissions
import calc_stats
import strategy
import equity_curve

import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from janus.units import CONTRACT_UNITS, CONTRACTS, ContractType
import os
import math
from typing import List, Union, Optional, Dict, Tuple, Callable
from datetime import datetime, timedelta

class Extend(InstrumentSelectionAlgorithm):
    
    def debug_position_sizing_detailed(self, instrument: str, date_index: int = 100) -> Dict:
        """
        FIXED: Debug position sizing for a specific instrument on a specific date
        """
        if not self.current_portfolio or instrument not in self.current_portfolio:
            return {"error": "Instrument not in portfolio"}
        
        # Get the data for debugging - FIXED: Keep separate
        returns_data = {}
        price_data = {}
        forecast_data = {}
        
        for inst in self.current_portfolio:
            inst_data = self.instruments_data[inst]
            returns_data[inst] = inst_data['Returns'].dropna()
            price_data[inst] = inst_data['Close']
            forecast_data[inst] = inst_data['Capped_Forecast']
        
        # Find common dates
        all_dates = None
        for inst in self.current_portfolio:
            inst_dates = (returns_data[inst].index
                        .intersection(price_data[inst].index)
                        .intersection(forecast_data[inst].index))
            if all_dates is None:
                all_dates = inst_dates
            else:
                all_dates = all_dates.intersection(inst_dates)
        
        if len(all_dates) <= date_index:
            date_index = len(all_dates) - 1
        
        date = all_dates[date_index]
        
        print(f"\n=== DEBUGGING POSITION SIZE FOR {instrument} on {date} ===")
        
        # Get all the inputs - FIXED: Direct access to individual series
        raw_price = price_data[instrument].loc[date]
        price_per_unit = get_contract_price_in_dollars(instrument, raw_price)
        current_forecast = forecast_data[instrument].loc[date]  # ← FIXED: Single value
        
        contract = CONTRACTS[instrument]
        multiplier = CONTRACT_UNITS[instrument]
        weight = self.portfolio_weights[instrument]
        
        # Calculate volatility using window
        vol_window = 60
        min_periods = 30
        inst_returns = returns_data[instrument]
        
        if date_index >= min_periods:
            vol_data = inst_returns.iloc[max(0, date_index-vol_window):date_index]
            annual_vol = vol_data.std() * np.sqrt(252)
        else:
            annual_vol = inst_returns.iloc[:date_index+1].std() * np.sqrt(252)
        
        # Get portfolio metrics
        portfolio_metrics = self.calculate_portfolio_metrics(self.current_portfolio, self.portfolio_weights)
        idm = portfolio_metrics['idm']
        target_vol = self.target_risk_pct
        
        print(f"Raw inputs:")
        print(f"  raw_price: {raw_price}")
        print(f"  price_per_unit: {price_per_unit}")
        print(f"  current_forecast: {current_forecast}")  # ← Now single value
        print(f"  multiplier: {multiplier}")
        print(f"  weight: {weight}")
        print(f"  annual_vol: {annual_vol}")
        print(f"  idm: {idm}")
        print(f"  target_vol: {target_vol}")
        print(f"  capital: {self.capital}")
        
        # Check for issues - FIXED: Single value comparisons
        issues = []
        if pd.isna(current_forecast) or current_forecast == 0:
            issues.append(f"Forecast is {current_forecast}")
        if pd.isna(annual_vol) or annual_vol <= 0:
            issues.append(f"Volatility is {annual_vol}")
        if price_per_unit <= 0:
            issues.append(f"Price is {price_per_unit}")
        if multiplier <= 0:
            issues.append(f"Multiplier is {multiplier}")
        
        if issues:
            print(f"ISSUES FOUND: {issues}")
            return {"issues": issues}
        
        # Calculate position size using the formula
        numerator = current_forecast * self.capital * idm * weight * target_vol
        denominator = 10 * multiplier * price_per_unit * 1.0 * annual_vol
        contracts = numerator / denominator
        
        print(f"\nPosition calculation:")
        print(f"  numerator = {current_forecast} × {self.capital} × {idm} × {weight} × {target_vol}")
        print(f"  numerator = {numerator:,.2f}")
        print(f"  denominator = 10 × {multiplier} × {price_per_unit} × 1.0 × {annual_vol}")
        print(f"  denominator = {denominator:,.2f}")
        print(f"  contracts = {contracts:.6f}")
        
        # Calculate notional
        notional_exposure = abs(contracts * price_per_unit * multiplier)
        print(f"  notional_exposure = {abs(contracts)} × {price_per_unit} × {multiplier} = ${notional_exposure:,.2f}")
        
        return {
            "date": date,
            "raw_price": raw_price,
            "price_per_unit": price_per_unit,
            "forecast": current_forecast,
            "annual_vol": annual_vol,
            "contracts": contracts,
            "notional": notional_exposure,
            "issues": issues
        }
    
    # Add this method to generate_strategy_returns to debug the first few days
    def debug_first_few_days(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Debug version of generate_strategy_returns that shows what's happening
        """
        
        if not self.current_portfolio:
            raise ValueError("No portfolio selected. Run run_selection_algorithm() first.")
        
        print(f"=== DEBUGGING STRATEGY RETURNS ===")
        print(f"Portfolio: {self.current_portfolio[:3]}...")  # Show first 3
        
        # Get aligned returns data for all instruments in portfolio
        returns_matrix = pd.DataFrame()
        price_matrix = pd.DataFrame()
        forecast_matrix = pd.DataFrame()
        
        for instrument in self.current_portfolio:
            instrument_data = self.instruments_data[instrument]
            
            returns_series = instrument_data['Returns'].dropna()
            price_series = instrument_data['Close']
            forecast_series = instrument_data['Capped_Forecast']
            
            returns_matrix[instrument] = returns_series
            price_matrix[instrument] = price_series
            forecast_matrix[instrument] = forecast_series
        
        # Align all data
        common_dates = returns_matrix.index.intersection(price_matrix.index).intersection(forecast_matrix.index)
        returns_matrix = returns_matrix.loc[common_dates]
        price_matrix = price_matrix.loc[common_dates]
        forecast_matrix = forecast_matrix.loc[common_dates]
        
        # Drop NaN
        complete_data = pd.concat([returns_matrix, price_matrix, forecast_matrix], axis=1).dropna()
        returns_matrix = complete_data[returns_matrix.columns]
        price_matrix = complete_data[price_matrix.columns]
        forecast_matrix = complete_data[forecast_matrix.columns]
        
        print(f"Data after alignment: {len(returns_matrix)} days")
        print(f"Date range: {returns_matrix.index[0]} to {returns_matrix.index[-1]}")
        
        # Check first few forecasts
        print(f"\nFirst 10 forecasts for first 3 instruments:")
        for i, inst in enumerate(self.current_portfolio[:3]):
            forecasts = forecast_matrix[inst].iloc[:10]
            print(f"  {inst}: {forecasts.values}")
        
        # Check if all forecasts are zero
        all_forecasts = forecast_matrix.values.flatten()
        zero_forecasts = (all_forecasts == 0).sum()
        total_forecasts = len(all_forecasts[~np.isnan(all_forecasts)])
        print(f"\nForecast stats:")
        print(f"  Total forecasts: {total_forecasts}")
        print(f"  Zero forecasts: {zero_forecasts}")
        print(f"  Non-zero forecasts: {total_forecasts - zero_forecasts}")
        print(f"  Forecast range: {np.nanmin(all_forecasts):.3f} to {np.nanmax(all_forecasts):.3f}")
        
        # Debug specific position sizing for first instrument on day 60
        if len(returns_matrix) > 60:
            debug_result = self.debug_position_sizing_detailed(self.current_portfolio[0], 60)
            print(f"\nDetailed debug for {self.current_portfolio[0]}:")
            print(debug_result)
        
        return returns_matrix.head(10)  # Return first 10 days for inspection

# Fix for the main() function error
def main_fixed():
    """Fixed main execution function"""
    
    strategies = {}
    
    # Buy and Hold Strategy
    print("Running Buy and Hold Strategy...")
    selector_bh = Extend(capital=10000000, target_risk_pct=0.20)
    results_bh = selector_bh.run_selection_algorithm(strategy_type=strategy.StrategyType.BUY_AND_HOLD)
    
    # DEBUG THE POSITION SIZING ISSUE
    print("\n" + "="*60)
    print("DEBUGGING POSITION SIZING")
    print("="*60)
    debug_data = selector_bh.debug_first_few_days(start_date='2007-01-01')
    
    # If debugging shows the issue, try generating returns
    try:
        returns_bh = selector_bh.generate_strategy_returns(start_date='2007-01-01')
        strategies['buy_and_hold_returns'] = returns_bh.copy()
    except Exception as e:
        print(f"Error generating returns: {e}")
        # Create dummy data to prevent crashes
        strategies['buy_and_hold_returns'] = pd.DataFrame({
            'Date': pd.date_range('2007-01-01', periods=100),
            'Daily_Return': np.zeros(100),
            'Cumulative_Return': np.zeros(100),
            'Portfolio_Value': np.full(100, 10000000)
        })
    
    # Comment out the other strategies for now to focus on the bug
    # # EWMAC Strategy  
    # selector_ewmac = InstrumentSelectionAlgorithm(capital=10000000, target_risk_pct=0.20)
    # results_ewmac = selector_ewmac.run_selection_algorithm(strategy_type=strategy.StrategyType.EWMAC)
    # returns_ewmac = selector_ewmac.generate_strategy_returns(start_date='2007-01-01')
    # strategies['trend_up_returns'] = returns_ewmac.copy()
    
    # Create dummy strategies to prevent KeyError
    strategies['trend_up_returns'] = strategies['buy_and_hold_returns'].copy()
    strategies['trend_up_down_returns'] = strategies['buy_and_hold_returns'].copy()
    strategies['forecast_ls'] = strategies['buy_and_hold_returns'].copy()

    # Now try the equity curve
    try:
        equity_curve.plot_equity_curves_fixed(strategies)
        plt.show()
    except Exception as e:
        print(f"Error plotting equity curves: {e}")

    return strategies

# Call this instead of main()
if __name__ == "__main__":
    strategies = main_fixed()