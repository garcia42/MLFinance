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

CSV_FILES = [
    'Brent_Crude_Oil_data.csv',
    'Coffee.csv',
    'Copper_data.csv',
    'Corn_data.csv',
    'Cotton.csv',
    'Crude_Oil_data.csv',
    'Feeder Cattle.csv',
    'Gold_data.csv',
    'Heating_Oil_data.csv',
    'KC_HRW_Wheat_data.csv',
    'Lean Hogs.csv',
    'Live Cattle.csv',
    'Natural_Gas_data.csv',
    'Oat_data.csv',
    'Palladium_data.csv',
    'Platinum_data.csv',
    'RBOB_Gasoline_data.csv',
    'Rough_Rice_data.csv',
    'Silver_data.csv',
    'Soybean_data.csv',
    'Soybean_Oil_data.csv',
    'Sugar.csv',
    'US 5 Year T-Note Futures Historical Data.csv',
    'US 10 Year T-Note Futures Historical Data.csv',
    'US 30 Year T-Bond Futures Historical Data.csv',
]


def get_contract_price_in_dollars(file: str, raw_price: float) -> float:
    """
    Convert raw price data to dollars per unit based on contract type.
    
    Args:
        file: CSV filename to identify contract type
        raw_price: Raw price from CSV data
        
    Returns:
        Price in dollars per unit
    """
    contract = CONTRACTS[file]
    
    if contract.type == ContractType.AGRICULTURE:
        # ALL agricultural commodities are quoted in cents
        # Grains: cents/bushel, Livestock: cents/pound, Softs: cents/pound
        return raw_price / 100.0
            
    elif contract.type == ContractType.ENERGY:
        # Energy contracts typically in dollars
        return raw_price
        
    elif contract.type == ContractType.METALS:
        # Metals typically in dollars per troy oz or per pound
        return raw_price
        
    elif contract.type == ContractType.FINANCIAL:
        # Financial instruments quoted as percentage of par
        if contract.symbol in ['ZF', 'ZN', 'ZB']:  # Treasury futures
            # Convert percentage quote to price per dollar of face value
            # e.g., 120.5 quote = 120.5% of par = 1.205 per dollar of face value
            return raw_price / 100.0
            
    # Default: assume price is already in correct units
    return raw_price


class InstrumentSelectionAlgorithm:
    """
    Implements the systematic instrument selection algorithm based on risk-adjusted costs.
    
    The algorithm follows these steps:
    1. Decide on a set of possible instruments
    2. Choose the first instrument for the portfolio
    3. Form the initial current portfolio and measure expected SR
    4. Iterate over all instruments to find the best trade-off between diversification and costs
    5. Choose the instrument with highest trial portfolio SR from step 4
    """
    
    def __init__(self, capital: float = 1000000, target_risk_pct: float = 0.20):
        self.capital = capital
        self.target_risk_pct = target_risk_pct
        self.instruments_data = {}
        self.instrument_metrics = {}
        self.current_portfolio = []
        self.portfolio_weights = {}
        self.returns_history = None
        self.portfolio_history = None
        
        # Forecasting parameters
        self.forecast_fast_span = 16
        self.forecast_slow_span = 64
        self.forecast_scalar = 1.9
        self.max_forecast = 20
        
    def load_instrument_data(self, csv_files: List[str], strategy_type: strategy.StrategyType) -> Dict:
        """Step 1: Load and prepare all instrument data with forecasts pre-calculated"""
        print("Step 1: Loading instrument data...")
        
        for file in tqdm(csv_files, desc="Loading instruments"):
            try:
                csv_path = os.path.join(os.path.dirname(__file__), '../individual_data', file)
                df = pd.read_csv(csv_path)
                
                # Prepare data
                df.columns = [col.capitalize() for col in df.columns]
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)
                
                # Calculate returns and forecasts based on strategy type
                if strategy_type == strategy.StrategyType.BUY_AND_HOLD:
                    df['Returns'] = df['Close'].pct_change().dropna()
                    # Set constant forecast for basic method
                    df['Capped_Forecast'] = 20.0
                    
                elif strategy_type == strategy.StrategyType.EWMAC:
                    df['Returns'] = strategy.ewmac_strategy(df)['Strategy_Returns']
                    # Set constant forecast for basic method
                    df['Capped_Forecast'] = 20.0
                    
                elif strategy_type == strategy.StrategyType.EWMAC_LONG_SHORT:
                    ewmac_results = strategy.ewmac_strategy(df, can_short=True, return_signals=True)
                    df['Returns'] = ewmac_results['Strategy_Returns']
                    
                    # Convert EWMAC signal to forecast equivalent
                    # EWMAC signal: +1 = long, -1 = short, 0 = flat
                    # Convert to forecast: +10 = long, -10 = short, 0 = flat
                    ewmac_signal = ewmac_results['Position'].fillna(0)
                    df['Capped_Forecast'] = ewmac_signal * 20.0
                    
                elif strategy_type == strategy.StrategyType.TREND_FORECAST:
                    # Calculate trend forecast results once here
                    trend_results = strategy.trend_forecast_strategy(
                        price_data=df,
                        fast_span=self.forecast_fast_span,
                        slow_span=self.forecast_slow_span,
                        forecast_scalar=self.forecast_scalar,
                        max_forecast=self.max_forecast,
                        return_signals=True
                    )
                    df['Returns'] = trend_results['Strategy_Returns']
                    df['Capped_Forecast'] = trend_results['Capped_Forecast']  # ← Store forecasts here!
                    
                    # Optionally store other trend signals for analysis
                    df['EWMA_Fast'] = trend_results['EWMA_Fast']
                    df['EWMA_Slow'] = trend_results['EWMA_Slow']
                    df['Raw_Forecast'] = trend_results['Raw_Forecast']
                
                self.instruments_data[file] = df
                
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
                
        print(f"Loaded {len(self.instruments_data)} instruments successfully")
        return self.instruments_data
    
    def calculate_instrument_metrics(self) -> Dict:
        """Calculate key metrics for each instrument using pre-calculated forecasts"""
        print(f"Calculating instrument metrics using stored forecasts...")
        
        for file, df in tqdm(self.instruments_data.items(), desc="Calculating metrics"):
            try:
                # Get contract details
                contract = CONTRACTS[file]
                info = commissions.get_futures_details(contract.symbol)
                
                if info is None:
                    print(f"Warning: No futures details found for {contract.symbol} ({file})")
                    continue
                
                # Convert price to proper units
                raw_price = df['Close'].iloc[-1]
                price_per_unit = get_contract_price_in_dollars(file, raw_price)
                
                # Get the latest forecast value (already calculated and stored)
                latest_forecast = df['Capped_Forecast'].dropna().iloc[-1]
                
                # Always use forecast-based position sizing (unified approach)
                pos_size = position_size.calculate_position_size(
                    capital=self.capital,
                    capped_forecast=latest_forecast,
                    target_risk_pct=self.target_risk_pct * 100,  # Convert to percentage
                    multiplier=CONTRACT_UNITS[file],
                    price=price_per_unit,
                    price_series=df['Close'].values
                )
                
                # Store forecast metrics (always the same structure)
                forecast_metrics = {
                    'latest_forecast': latest_forecast,
                    'forecast_strength': abs(latest_forecast) / self.max_forecast,
                    'position_direction': 'LONG' if latest_forecast > 0 else 'SHORT' if latest_forecast < 0 else 'FLAT',
                    'trend_results': None  # Already stored in df if needed
                }
                # Calculate risk-adjusted costs (same for both methods)
                std_dev = calc_stats.annual_std_dev(df['Returns'].dropna())
                comm_pct = info['commission'] / (price_per_unit * contract.size)
                risk_adjusted_cost = costs_stats.calculate_risk_adjusted_cost(
                    tick_size=info['tick_size'],
                    current_price=price_per_unit * contract.size,
                    annualized_volatility=std_dev,
                    rolls_per_year=info['months_per_year'],
                    multiplier=contract.size,
                    commission_pct=comm_pct,
                    print_results=False
                )
                
                # Calculate Sharpe ratio for individual instrument
                annual_return = df['Returns'].dropna().mean() * 252
                sharpe_ratio = annual_return / std_dev if std_dev > 0 else 0
                
                # Combine all metrics (unified structure)
                self.instrument_metrics[file] = {
                    'annual_volatility': std_dev,
                    'annual_return': annual_return,
                    'sharpe_ratio': sharpe_ratio,
                    'risk_adjusted_cost': risk_adjusted_cost['annual_risk_adjusted_cost'],
                    'position_size': pos_size,
                    'current_price': price_per_unit,
                    'raw_price': raw_price,
                    'returns_series': df['Returns'].dropna(),
                    **forecast_metrics  # Add forecast metrics (always same structure)
                }
                
            except Exception as e:
                print(f"Error calculating metrics for {file}: {e}")
                continue
                
        return self.instrument_metrics
    
    def select_first_instrument(self) -> str:
        """Step 2: Choose the first instrument with lowest risk-adjusted cost (no capital filtering)"""
        print("Step 2: Selecting first instrument (lowest cost)...")
        
        # Sort ALL instruments by risk-adjusted cost (no capital filtering)
        sorted_instruments = sorted(
            self.instrument_metrics.items(),
            key=lambda x: x[1]['risk_adjusted_cost']
        )
        
        first_instrument = sorted_instruments[0][0]
        first_cost = sorted_instruments[0][1]['risk_adjusted_cost']
        
        print(f"Selected first instrument: {first_instrument}")
        print(f"Risk-adjusted cost: {first_cost:.6f}")
        
        # Show top 5 candidates for reference
        print(f"\nTop 5 candidates by risk-adjusted cost:")
        for i, (inst, metrics) in enumerate(sorted_instruments[:5], 1):
            cost = metrics['risk_adjusted_cost']
            sharpe = metrics['sharpe_ratio']
            print(f"{i}. {inst:<40} Cost: {cost:.6f}, Sharpe: {sharpe:.4f}")
        
        return first_instrument
    
    def form_initial_portfolio(self, first_instrument: str) -> Dict:
        """Step 3: Form initial portfolio with first instrument"""
        print("Step 3: Forming initial portfolio...")
        
        self.current_portfolio = [first_instrument]
        
        # For single instrument, weight is 1.0
        self.portfolio_weights = {first_instrument: 1.0}
        
        # Calculate initial portfolio metrics
        portfolio_metrics = self.calculate_portfolio_metrics(self.current_portfolio, self.portfolio_weights)
        
        print(f"Initial portfolio Sharpe ratio: {portfolio_metrics['sharpe_ratio']:.4f}")
        
        return portfolio_metrics
    
    def check_minimum_capital_for_trial_portfolio(self, trial_instruments: List[str], 
                                                    trial_weights: Dict[str, float], 
                                                    trial_idm: float) -> bool:
        """
        Step 4.iii: Check if ALL instruments in trial portfolio meet minimum capital requirements
        using the ACTUAL trial portfolio weights and IDM, not fixed assumptions.
        """
        
        for instrument in trial_instruments:
            contract = CONTRACTS[instrument]
            multiplier = CONTRACT_UNITS[instrument]
            
            # Get current price and volatility for this instrument
            raw_price = self.instrument_metrics[instrument]['raw_price']
            annual_vol = self.instrument_metrics[instrument]['annual_volatility']
            
            # Convert price to proper units
            price_per_unit = get_contract_price_in_dollars(instrument, raw_price)
            
            # Calculate minimum capital using ACTUAL trial portfolio parameters
            weight = trial_weights[instrument]
            fx_rate = 1.0
            
            numerator = 4 * multiplier * price_per_unit * fx_rate * annual_vol
            denominator = trial_idm * weight * self.target_risk_pct
            min_capital_required = numerator / denominator
            
            # Check if we have sufficient capital
            if min_capital_required > self.capital:
                print(f"  ✗ {instrument:<30} requires ${min_capital_required:,.0f} (exceeds ${self.capital:,.0f})")
                return False
            else:
                print(f"  ✓ {instrument:<30} requires ${min_capital_required:,.0f}")
        
        return True
    
    def calculate_portfolio_metrics(self, instruments: List[str], weights: Dict[str, float]) -> Dict:
        """Calculate portfolio-level metrics using IDM formulas from the algorithm"""
        
        # Get returns series for all instruments
        returns_matrix = pd.DataFrame()
        for instrument in instruments:
            returns_matrix[instrument] = self.instrument_metrics[instrument]['returns_series']
        
        # Align dates and drop NaN
        returns_matrix = returns_matrix.dropna()
        
        # Calculate correlation matrix (Σ)
        correlation_matrix = returns_matrix.corr()
        
        # Calculate IDM (Instrument Diversification Multiplier)
        # IDM = 1 / sqrt(w' * Σ * w) where w is weights vector and Σ is correlation matrix
        weights_array = np.array([weights[inst] for inst in instruments])
        correlation_array = correlation_matrix.values
        
        # Matrix multiplication: w' * Σ * w
        portfolio_variance_from_correlations = np.dot(weights_array, np.dot(correlation_array, weights_array))
        idm = 1.0 / np.sqrt(portfolio_variance_from_correlations)
        
        # Calculate individual instrument Sharpe ratios (net of costs)
        instrument_sharpes = {}
        for instrument in instruments:
            gross_sharpe = self.instrument_metrics[instrument]['sharpe_ratio']
            risk_adjusted_cost = self.instrument_metrics[instrument]['risk_adjusted_cost']
            # Net Sharpe = Gross Sharpe - (τ × c_i) where τ is target risk and c_i is cost
            net_sharpe = gross_sharpe - (self.target_risk_pct * risk_adjusted_cost)
            instrument_sharpes[instrument] = net_sharpe
        
        # Portfolio mean using IDM formula:
        # Portfolio mean = Sum(Weight_i × IDM × τ × [SR'_i - (τ × c_i)])
        portfolio_mean = 0
        for instrument in instruments:
            weight = weights[instrument]
            net_sharpe = instrument_sharpes[instrument]
            # Instrument contribution to portfolio mean
            instrument_contribution = weight * idm * self.target_risk_pct * net_sharpe
            portfolio_mean += instrument_contribution
        
        # Portfolio standard deviation using IDM formula:
        # Portfolio σ = IDM × τ × sqrt(w' * Σ * w')
        # But since IDM = 1/sqrt(w' * Σ * w'), this simplifies to:
        # Portfolio σ = τ (target risk remains constant by design)
        portfolio_volatility = self.target_risk_pct
        
        # Portfolio Sharpe Ratio:
        # Portfolio SR = Sum(Weight_i × [SR'_i - (τ × c_i)]) ÷ sqrt(w' * Σ * w')
        weighted_net_sharpe_sum = sum(weights[inst] * instrument_sharpes[inst] for inst in instruments)
        portfolio_sharpe = weighted_net_sharpe_sum / np.sqrt(portfolio_variance_from_correlations)
        
        # Calculate actual portfolio returns for validation
        portfolio_returns = pd.Series(0, index=returns_matrix.index)
        for instrument in instruments:
            portfolio_returns += returns_matrix[instrument] * weights[instrument]
        
        empirical_annual_return = portfolio_returns.mean() * 252
        empirical_annual_vol = portfolio_returns.std() * np.sqrt(252)
        empirical_sharpe = empirical_annual_return / empirical_annual_vol if empirical_annual_vol > 0 else 0
        
        return {
            'annual_return': portfolio_mean,
            'annual_volatility': portfolio_volatility,
            'sharpe_ratio': portfolio_sharpe,
            'idm': idm,
            'correlation_matrix': correlation_matrix,
            'instrument_net_sharpes': instrument_sharpes,
            'portfolio_variance_from_correlations': portfolio_variance_from_correlations,
            'returns_series': portfolio_returns,
            # Include empirical metrics for comparison
            'empirical_annual_return': empirical_annual_return,
            'empirical_annual_volatility': empirical_annual_vol,
            'empirical_sharpe': empirical_sharpe
        }
    
    def calculate_optimal_weights(self, instruments: List[str]) -> Dict[str, float]:
        """Calculate optimal weights for given instruments (simplified equal weighting)"""
        # For simplicity, using equal weights
        # In practice, you might want to use mean-variance optimization
        weight = 1.0 / len(instruments)
        return {instrument: weight for instrument in instruments}
    
    def iterate_and_select_next_instrument(self) -> Tuple[Optional[str], float]:
        """Step 4: Iterate over remaining instruments with proper capital checks"""
        print("Step 4: Evaluating potential additions to portfolio...")
        
        best_instrument = None
        best_sharpe = -float('inf')
        current_portfolio_sharpe = self.calculate_portfolio_metrics(
            self.current_portfolio, self.portfolio_weights
        )['sharpe_ratio']
        
        # Consider all instruments not currently in portfolio
        remaining_instruments = [
            inst for inst in self.instrument_metrics.keys() 
            if inst not in self.current_portfolio
        ]
        
        print(f"Evaluating {len(remaining_instruments)} potential additions...")
        
        for candidate_instrument in tqdm(remaining_instruments, desc="Testing additions"):
            try:
                # Create trial portfolio
                trial_portfolio = self.current_portfolio + [candidate_instrument]
                trial_weights = self.calculate_optimal_weights(trial_portfolio)
                
                # Calculate trial portfolio metrics (including IDM)
                trial_metrics = self.calculate_portfolio_metrics(trial_portfolio, trial_weights)
                trial_idm = trial_metrics['idm']
                
                # Step 4.iii: Check minimum capital requirement for trial portfolio
                print(f"\nChecking capital requirements for trial with {candidate_instrument}:")
                capital_ok = self.check_minimum_capital_for_trial_portfolio(
                    trial_portfolio, trial_weights, trial_idm
                )
                
                if not capital_ok:
                    print(f"Skipping {candidate_instrument} - insufficient capital")
                    continue
                
                # If capital check passes, evaluate Sharpe improvement
                improvement = trial_metrics['sharpe_ratio'] - current_portfolio_sharpe

                if trial_metrics['sharpe_ratio'] > best_sharpe:
                    best_sharpe = trial_metrics['sharpe_ratio']
                    best_instrument = candidate_instrument
                    print(f"New best candidate: {candidate_instrument} (SR: {best_sharpe:.4f}, improvement: {improvement:.4f})")
                    
            except Exception as e:
                print(f"Error evaluating {candidate_instrument}: {e}")
                continue
        
        return best_instrument, best_sharpe
    
    def run_selection_algorithm(self, strategy_type: strategy.StrategyType) -> Dict:
        """Run the complete instrument selection algorithm"""
        
        print("=" * 60)
        print("RUNNING INSTRUMENT SELECTION ALGORITHM")
        print("=" * 60)
        
        # Step 1: Load data
        self.load_instrument_data(CSV_FILES, strategy_type=strategy_type)
        
        # Calculate metrics for all instruments
        self.calculate_instrument_metrics()
        
        # Step 2: Select first instrument (no capital filtering)
        first_instrument = self.select_first_instrument()
        
        # Step 3: Form initial portfolio
        current_metrics = self.form_initial_portfolio(first_instrument)
        
        # Step 4 & 5: Iteratively add instruments
        print(f"\nStep 4-5: Iteratively adding instruments...")
        
        iteration = 1
        highest_sharpe_so_far = current_metrics['sharpe_ratio']
        while True:
            print(f"\n--- Iteration {iteration} ---")
            print(f"Current portfolio: {self.current_portfolio}")
            print(f"Current Sharpe ratio: {current_metrics['sharpe_ratio']:.4f}")
            
            # Find best addition
            best_addition, best_sharpe = self.iterate_and_select_next_instrument()
            
            if best_addition is None:
                print("No suitable instrument found for addition. Stopping.")
                break
            
            # Add the instrument to portfolio
            self.current_portfolio.append(best_addition)
            self.portfolio_weights = self.calculate_optimal_weights(self.current_portfolio)
            current_metrics = self.calculate_portfolio_metrics(self.current_portfolio, self.portfolio_weights)
            
            print(f"Added: {best_addition}")
            print(f"New Sharpe ratio: {current_metrics['sharpe_ratio']:.4f}")
            
            # Check if improvement is significant enough
            improvement = best_sharpe - current_metrics['sharpe_ratio']
            if best_sharpe < highest_sharpe_so_far * .9:
                print(f"Current iteration sharpe ({best_sharpe:.4f}) below threshold ({highest_sharpe_so_far * .8}). Stopping.")
                break
            highest_sharpe_so_far = max(highest_sharpe_so_far, current_metrics['sharpe_ratio'])

            print(f"Improvement: {improvement:.4f}")
            
            iteration += 1
        
        # Final results
        final_results = {
            'selected_instruments': self.current_portfolio,
            'portfolio_weights': self.portfolio_weights,
            'final_metrics': current_metrics,
            'instrument_details': {
                inst: self.instrument_metrics[inst] 
                for inst in self.current_portfolio
            }
        }
        
        self.print_final_results(final_results)
        
        return final_results
    
    def generate_strategy_returns(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        FIXED: Generate returns time series using stored forecasts for position sizing.
        """
        
        if not self.current_portfolio:
            raise ValueError("No portfolio selected. Run run_selection_algorithm() first.")
        
        print(f"Generating strategy returns using stored forecasts for portfolio: {self.current_portfolio}")
        
        # Get aligned returns data for all instruments in portfolio
        returns_data = {}
        price_data = {}
        forecast_data = {}
        
        # FIXED: Keep each instrument's data separate to avoid column confusion
        for instrument in self.current_portfolio:
            instrument_data = self.instruments_data[instrument]
            
            returns_series = instrument_data['Returns'].dropna()
            price_series = instrument_data['Close']
            forecast_series = instrument_data['Capped_Forecast']
            
            # Store each instrument separately
            returns_data[instrument] = returns_series
            price_data[instrument] = price_series
            forecast_data[instrument] = forecast_series
        
        # Find common dates across ALL data types for ALL instruments
        all_dates = None
        for instrument in self.current_portfolio:
            inst_dates = (returns_data[instrument].index
                        .intersection(price_data[instrument].index)
                        .intersection(forecast_data[instrument].index))
            
            if all_dates is None:
                all_dates = inst_dates
            else:
                all_dates = all_dates.intersection(inst_dates)
        
        # Now create aligned matrices using only common dates
        returns_matrix = pd.DataFrame(index=all_dates)
        price_matrix = pd.DataFrame(index=all_dates)
        forecast_matrix = pd.DataFrame(index=all_dates)
        
        for instrument in self.current_portfolio:
            returns_matrix[instrument] = returns_data[instrument].loc[all_dates]
            price_matrix[instrument] = price_data[instrument].loc[all_dates]
            forecast_matrix[instrument] = forecast_data[instrument].loc[all_dates]
        
        # Drop any remaining NaN values
        combined_check = pd.concat([returns_matrix, price_matrix, forecast_matrix], axis=1)
        clean_dates = combined_check.dropna().index
        
        returns_matrix = returns_matrix.loc[clean_dates]
        price_matrix = price_matrix.loc[clean_dates]
        forecast_matrix = forecast_matrix.loc[clean_dates]
        
        # Apply date filters if provided
        if start_date:
            start_date = pd.to_datetime(start_date)
            mask = returns_matrix.index >= start_date
            returns_matrix = returns_matrix[mask]
            price_matrix = price_matrix[mask]
            forecast_matrix = forecast_matrix[mask]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            mask = returns_matrix.index <= end_date
            returns_matrix = returns_matrix[mask]
            price_matrix = price_matrix[mask]
            forecast_matrix = forecast_matrix[mask]
        
        if returns_matrix.empty:
            raise ValueError("No data available for the specified date range")
        
        print(f"Calculating returns from {returns_matrix.index[0].strftime('%Y-%m-%d')} to {returns_matrix.index[-1].strftime('%Y-%m-%d')}")
        
        # DEBUGGING: Check first few forecasts
        print(f"\nDEBUG: First 5 forecasts for first 3 instruments:")
        for i, inst in enumerate(self.current_portfolio[:3]):
            first_forecasts = forecast_matrix[inst].iloc[:5]
            print(f"  {inst}: {first_forecasts.values}")
            if (first_forecasts == 0).all():
                print(f"  ⚠️  WARNING: All forecasts are ZERO for {inst}")
        
        # Check forecast statistics
        all_forecasts = forecast_matrix.values.flatten()
        zero_forecasts = (all_forecasts == 0).sum()
        total_forecasts = len(all_forecasts[~np.isnan(all_forecasts)])
        print(f"\nDEBUG: Forecast statistics:")
        print(f"  Total valid forecasts: {total_forecasts}")
        print(f"  Zero forecasts: {zero_forecasts}")
        print(f"  Non-zero forecasts: {total_forecasts - zero_forecasts}")
        print(f"  Forecast range: {np.nanmin(all_forecasts):.3f} to {np.nanmax(all_forecasts):.3f}")
        
        # Get portfolio metrics for IDM
        portfolio_metrics = self.calculate_portfolio_metrics(
            self.current_portfolio, self.portfolio_weights
        )
        idm = portfolio_metrics['idm']
        target_vol = self.target_risk_pct
        
        print(f"Using IDM: {idm:.4f} for position sizing")
        print(f"Target portfolio volatility: {target_vol:.1%}")
        
        # Initialize result series
        portfolio_returns = pd.Series(0.0, index=returns_matrix.index)
        portfolio_notional = pd.Series(0.0, index=returns_matrix.index)
        leverage_series = pd.Series(0.0, index=returns_matrix.index)
        
        # Calculate rolling volatilities for better position sizing
        instrument_rolling_vols = {}
        vol_window = 60
        min_periods = 30
        
        for instrument in self.current_portfolio:
            inst_returns = returns_matrix[instrument]
            rolling_vol = inst_returns.rolling(window=vol_window, min_periods=min_periods).std() * np.sqrt(252)
            expanding_vol = inst_returns.expanding(min_periods=min_periods).std() * np.sqrt(252)
            rolling_vol = rolling_vol.fillna(expanding_vol)
            instrument_rolling_vols[instrument] = rolling_vol
        
        # DEBUGGING: Test position sizing calculation for first few days
        print(f"\nDEBUG: Testing position sizing for first valid day...")
        test_date_idx = min_periods  # Use a date where we have volatility
        test_date = returns_matrix.index[test_date_idx]
        test_instrument = self.current_portfolio[0]
        
        print(f"Test date: {test_date}")
        print(f"Test instrument: {test_instrument}")
        
        # Get test values
        test_raw_price = price_matrix.loc[test_date, test_instrument]
        test_price_per_unit = get_contract_price_in_dollars(test_instrument, test_raw_price)
        test_forecast = forecast_matrix.loc[test_date, test_instrument]
        test_vol = instrument_rolling_vols[test_instrument].loc[test_date]
        
        contract = CONTRACTS[test_instrument]
        multiplier = CONTRACT_UNITS[test_instrument]
        weight = self.portfolio_weights[test_instrument]
        
        print(f"DEBUG: Position sizing inputs:")
        print(f"  raw_price: {test_raw_price}")
        print(f"  price_per_unit: {test_price_per_unit}")
        print(f"  forecast: {test_forecast}")
        print(f"  volatility: {test_vol}")
        print(f"  multiplier: {multiplier}")
        print(f"  weight: {weight}")
        
        # Test the position sizing formula
        if (not pd.isna(test_forecast) and test_forecast != 0 and 
            not pd.isna(test_vol) and test_vol > 0 and test_price_per_unit > 0):
            
            numerator = test_forecast * self.capital * idm * weight * target_vol
            denominator = 10 * multiplier * test_price_per_unit * 1.0 * test_vol
            test_contracts = numerator / denominator
            
            print(f"  numerator: {numerator:,.2f}")
            print(f"  denominator: {denominator:,.2f}")
            print(f"  contracts: {test_contracts:.6f}")
            
            if abs(test_contracts) < 0.01:
                print(f"  ⚠️  WARNING: Calculated contracts are very small!")
            else:
                print(f"  ✅ Position sizing looks reasonable")
        else:
            print(f"  ❌ ERROR: Invalid inputs for position sizing")
            print(f"    forecast valid: {not pd.isna(test_forecast) and test_forecast != 0}")
            print(f"    vol valid: {not pd.isna(test_vol) and test_vol > 0}")
            print(f"    price valid: {test_price_per_unit > 0}")
        
        # Calculate positions for each day
        position_debug_count = 0
        for i, date in enumerate(returns_matrix.index):
            daily_portfolio_return = 0.0
            daily_notional = 0.0
            
            # Skip first few days if we don't have enough volatility data
            if i < min_periods:
                portfolio_returns.loc[date] = 0.0
                portfolio_notional.loc[date] = 0.0
                leverage_series.loc[date] = 0.0
                continue
            
            # Calculate individual instrument positions
            for instrument in self.current_portfolio:
                try:
                    # Get current price, forecast, and return - FIXED: Direct access
                    raw_price = price_matrix.loc[date, instrument]
                    price_per_unit = get_contract_price_in_dollars(instrument, raw_price)
                    current_forecast = forecast_matrix.loc[date, instrument]  # ← FIXED: Single value
                    
                    if pd.isna(raw_price) or pd.isna(current_forecast):
                        continue
                    
                    if price_per_unit <= 0:
                        continue
                    
                    # Get contract details
                    contract = CONTRACTS[instrument]
                    multiplier = CONTRACT_UNITS[instrument]
                    weight = self.portfolio_weights[instrument]
                    annual_vol = instrument_rolling_vols[instrument].loc[date]
                    
                    if pd.isna(annual_vol) or annual_vol <= 0:
                        continue
                    
                    # Position sizing formula - FIXED: Using single forecast value
                    numerator = current_forecast * self.capital * idm * weight * target_vol
                    denominator = 10 * multiplier * price_per_unit * 1.0 * annual_vol
                    contracts = numerator / denominator
                    
                    # Debug first few calculations
                    if position_debug_count < 3 and abs(contracts) > 0:
                        print(f"\nDEBUG position {position_debug_count + 1} on {date}:")
                        print(f"  {instrument}: forecast={current_forecast:.3f}, contracts={contracts:.3f}")
                        position_debug_count += 1
                    
                    # Calculate notional exposure
                    notional_exposure = abs(contracts * price_per_unit * multiplier)
                    daily_notional += notional_exposure
                    
                    # Calculate contribution to portfolio return
                    instrument_return = returns_matrix.loc[date, instrument]
                    
                    # Dollar P&L from this position
                    contract_value = contracts * price_per_unit * multiplier
                    dollar_pnl = contract_value * instrument_return
                    
                    # Contribution to portfolio return (as percentage)
                    portfolio_contribution = dollar_pnl / self.capital
                    daily_portfolio_return += portfolio_contribution
                    
                except Exception as e:
                    # Skip this instrument on this day if there's an error
                    if position_debug_count < 5:  # Only show first few errors
                        print(f"Error processing {instrument} on {date}: {e}")
                        position_debug_count += 1
                    continue
            
            # Store results
            portfolio_returns.loc[date] = daily_portfolio_return
            portfolio_notional.loc[date] = daily_notional
            leverage_series.loc[date] = daily_notional / self.capital if self.capital > 0 else 0.0
        
        # Calculate cumulative returns and portfolio value
        portfolio_returns = portfolio_returns.fillna(0.0)
        cumulative_returns = (1 + portfolio_returns).cumprod() - 1
        portfolio_value = self.capital * (1 + cumulative_returns)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Date': returns_matrix.index,
            'Daily_Return': portfolio_returns.values,
            'Cumulative_Return': cumulative_returns.values,
            'Portfolio_Value': portfolio_value.values,
            'Total_Notional': portfolio_notional.values,
            'Leverage': leverage_series.values
        })
        
        # Remove any remaining NaN or inf values
        results_df = results_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        
        # Store for later access
        self.returns_history = results_df.copy()
        
        # Calculate and display summary statistics
        self._display_returns_summary_corrected(results_df, portfolio_metrics)
        
        return results_df
        
    def _display_returns_summary_corrected(self, returns_df: pd.DataFrame, portfolio_metrics: Dict):
        """Enhanced summary display with forecast-based position sizing"""
        
        daily_returns = returns_df['Daily_Return']
        total_return = returns_df['Cumulative_Return'].iloc[-1]
        
        # Annualized metrics
        trading_days = len(daily_returns)
        years = trading_days / 252
        annualized_return = (1 + total_return) ** (1/years) - 1
        annualized_vol = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(returns_df['Cumulative_Return'])
        
        # Win/Loss metrics
        winning_days = (daily_returns > 0).sum()
        losing_days = (daily_returns < 0).sum()
        win_rate = winning_days / (winning_days + losing_days) if (winning_days + losing_days) > 0 else 0
        
        # Leverage metrics
        avg_leverage = returns_df['Leverage'].mean()
        max_leverage = returns_df['Leverage'].max()
        
        print(f"\n" + "=" * 70)
        print(f"STRATEGY RETURNS SUMMARY - FORECAST-BASED POSITION SIZING")
        print("=" * 70)
        print(f"Period: {returns_df['Date'].iloc[0].strftime('%Y-%m-%d')} to {returns_df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
        print(f"Trading Days: {trading_days:,}")
        print(f"Years: {years:.2f}")
        
        print(f"Forecast Parameters: EWMAC({self.forecast_fast_span},{self.forecast_slow_span}), Scalar={self.forecast_scalar}, Max=±{self.max_forecast}")
        
        print(f"\n✅ Actual Performance Metrics:")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Annualized Volatility: {annualized_vol:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        
        print(f"\n🎯 Target Performance (from IDM formulas):")
        print(f"Expected Annual Return: {portfolio_metrics['annual_return']:.2%}")
        print(f"Target Annual Volatility: {portfolio_metrics['annual_volatility']:.2%}")
        print(f"Expected Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.4f}")
        
        print(f"\n📊 Performance vs Target:")
        return_diff = annualized_return - portfolio_metrics['annual_return']
        vol_diff = annualized_vol - portfolio_metrics['annual_volatility']
        sharpe_diff = sharpe_ratio - portfolio_metrics['sharpe_ratio']
        
        return_match = "✅ GOOD" if abs(return_diff) < 0.02 else "❌ OFF"
        vol_match = "✅ GOOD" if abs(vol_diff) < 0.02 else "❌ OFF"
        sharpe_match = "✅ GOOD" if abs(sharpe_diff) < 0.1 else "❌ OFF"
        
        print(f"Return Difference: {return_diff:+.2%} {return_match}")
        print(f"Volatility Difference: {vol_diff:+.2%} {vol_match}")
        print(f"Sharpe Difference: {sharpe_diff:+.4f} {sharpe_match}")
        
        print(f"\n🔧 Position Sizing Metrics:")
        print(f"Method: FORECAST-BASED POSITION SIZING")
        print(f"Average Leverage: {avg_leverage:.2f}x (target: ~{portfolio_metrics['idm']:.2f}x)")
        print(f"Maximum Leverage: {max_leverage:.2f}x")
        print(f"Average Total Notional: ${returns_df['Total_Notional'].mean():,.0f}")
        print(f"IDM Used: {portfolio_metrics['idm']:.4f}")
        
        print(f"\n📈 Risk Metrics:")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Winning Days: {winning_days:,}")
        print(f"Losing Days: {losing_days:,}")
        
        print(f"\n💰 Portfolio Value:")
        print(f"Starting Value: ${self.capital:,.2f}")
        print(f"Ending Value: ${returns_df['Portfolio_Value'].iloc[-1]:,.2f}")
        print(f"Peak Value: ${returns_df['Portfolio_Value'].max():,.2f}")
        
        # Diagnosis
        if avg_leverage < portfolio_metrics['idm'] * 0.8:
            print(f"\n⚠️  DIAGNOSIS: Forecast-based leverage too low")
            print(f"   This could indicate forecasts are too conservative or volatility estimates too high")
        else:
            print(f"\n✅ DIAGNOSIS: Forecast-based position sizing working correctly!")
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown from cumulative returns"""
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / (1 + peak)
        return drawdown.min()

    def plot_strategy_performance(self, save_path: str = None) -> None:
        """
        Plot strategy performance charts.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot. If None, displays the plot.
        """
        
        if self.returns_history is None:
            print("No returns history available. Run generate_strategy_returns() first.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Cumulative Returns
        ax1.plot(self.returns_history['Date'], self.returns_history['Cumulative_Return'] * 100)
        ax1.set_title('Cumulative Returns (%)')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Portfolio Value
        ax2.plot(self.returns_history['Date'], self.returns_history['Portfolio_Value'] / 1000000)
        ax2.set_title('Portfolio Value ($M)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Portfolio Value ($M)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Daily Returns Distribution
        ax3.hist(self.returns_history['Daily_Return'] * 100, bins=50, alpha=0.7, edgecolor='black')
        ax3.set_title('Daily Returns Distribution')
        ax3.set_xlabel('Daily Return (%)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Rolling Sharpe Ratio (252-day window)
        rolling_sharpe = self.returns_history['Daily_Return'].rolling(252).apply(
            lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
        )
        ax4.plot(self.returns_history['Date'], rolling_sharpe)
        ax4.set_title('Rolling 252-Day Sharpe Ratio')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
    
    def export_returns_to_csv(self, filename: str = None) -> str:
        """
        Export returns history to CSV file.
        
        Parameters:
        -----------
        filename : str, optional
            Output filename. If None, uses default naming.
            
        Returns:
        --------
        str: Path to the saved file
        """
        
        if self.returns_history is None:
            raise ValueError("No returns history available. Run generate_strategy_returns() first.")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"strategy_returns_{timestamp}.csv"
        
        # Add additional columns for analysis
        export_df = self.returns_history.copy()
        export_df['Year'] = export_df['Date'].dt.year
        export_df['Month'] = export_df['Date'].dt.month
        export_df['Weekday'] = export_df['Date'].dt.day_name()
        
        # Add rolling metrics
        export_df['Rolling_30d_Return'] = export_df['Daily_Return'].rolling(30).sum()
        export_df['Rolling_252d_Vol'] = export_df['Daily_Return'].rolling(252).std() * np.sqrt(252)
        
        export_df.to_csv(filename, index=False)
        print(f"Returns exported to: {filename}")
        
        return filename
    
    def print_final_results(self, results: Dict):
        """Print formatted final results"""
        print("\n" + "=" * 80)
        print("FINAL PORTFOLIO SELECTION RESULTS")
        print("=" * 80)
        
        print(f"\nSelected Instruments ({len(results['selected_instruments'])}):")
        print(f"{'#':<3} {'Instrument':<40} {'Weight':<8} {'Cost':<10} {'Gross SR':<8} {'Net SR':<8}")
        print("-" * 80)
        
        for i, instrument in enumerate(results['selected_instruments'], 1):
            weight = results['portfolio_weights'][instrument]
            cost = results['instrument_details'][instrument]['risk_adjusted_cost']
            gross_sharpe = results['instrument_details'][instrument]['sharpe_ratio']
            net_sharpe = results['final_metrics']['instrument_net_sharpes'][instrument]
            
            print(f"{i:2d}. {instrument:<40} {weight:6.3f}   {cost:8.6f}  {gross_sharpe:6.3f}   {net_sharpe:6.3f}")
        
        metrics = results['final_metrics']
        print(f"\nPortfolio Metrics (IDM-based):")
        print("-" * 40)
        print(f"IDM (Diversification Multiplier): {metrics['idm']:.4f}")
        print(f"Portfolio Variance (from correlations): {metrics['portfolio_variance_from_correlations']:.6f}")
        print(f"Annual Return (theoretical): {metrics['annual_return']:.4f}")
        print(f"Annual Volatility (target): {metrics['annual_volatility']:.4f}")
        print(f"Sharpe Ratio (IDM formula): {metrics['sharpe_ratio']:.4f}")
        
        print(f"\nEmpirical Portfolio Metrics (for comparison):")
        print("-" * 40)
        print(f"Annual Return (empirical): {metrics['empirical_annual_return']:.4f}")
        print(f"Annual Volatility (empirical): {metrics['empirical_annual_volatility']:.4f}")
        print(f"Sharpe Ratio (empirical): {metrics['empirical_sharpe']:.4f}")
        
        print(f"\nCorrelation Matrix:")
        print("-" * 40)
        correlation_df = metrics['correlation_matrix']
        # Show abbreviated instrument names for readability
        short_names = {inst: f"Inst{i+1}" for i, inst in enumerate(results['selected_instruments'])}
        correlation_df_short = correlation_df.copy()
        correlation_df_short.index = [short_names[inst] for inst in correlation_df_short.index]
        correlation_df_short.columns = [short_names[inst] for inst in correlation_df_short.columns]
        print(correlation_df_short.round(3))

def main():
    """Main execution function with returns generation"""
    
    strategies = {}
    
    # Buy and Hold Strategy
    selector_bh = InstrumentSelectionAlgorithm(capital=10000000, target_risk_pct=0.20)
    results_bh = selector_bh.run_selection_algorithm(strategy_type=strategy.StrategyType.BUY_AND_HOLD)
    returns_bh = selector_bh.generate_strategy_returns(start_date='2007-01-01')
    strategies['buy_and_hold_returns'] = returns_bh.copy()
    
    # EWMAC Strategy  
    selector_ewmac = InstrumentSelectionAlgorithm(capital=10000000, target_risk_pct=0.20)
    results_ewmac = selector_ewmac.run_selection_algorithm(strategy_type=strategy.StrategyType.EWMAC)
    returns_ewmac = selector_ewmac.generate_strategy_returns(start_date='2007-01-01')
    strategies['trend_up_returns'] = returns_ewmac.copy()
    
    # EWMAC Long/Short Strategy
    selector_ls = InstrumentSelectionAlgorithm(capital=10000000, target_risk_pct=0.20)
    results_ls = selector_ls.run_selection_algorithm(strategy_type=strategy.StrategyType.EWMAC_LONG_SHORT)
    returns_ls = selector_ls.generate_strategy_returns(start_date='2007-01-01')
    strategies['trend_up_down_returns'] = returns_ls.copy()
    
    # EWMAC Long/Short with forecast Strategy
    selector_forecast = InstrumentSelectionAlgorithm(capital=10000000, target_risk_pct=0.20)
    results_forecast = selector_forecast.run_selection_algorithm(strategy_type=strategy.StrategyType.TREND_FORECAST)
    returns_forecast = selector_forecast.generate_strategy_returns(start_date='2007-01-01')
    strategies['forecast_ls'] = returns_forecast.copy()

    # Add this to your code to confirm the issue
    equity_curve.plot_equity_curves_fixed(strategies)

    plt.show()

    # Plot performance
    # print("\nStep 3: Creating performance plots...")
    # selector.plot_strategy_performance()

    multi = {
        'Jumbo': {'returns': strategies['buy_and_hold_returns']['Daily_Return'], 'instrument': 'S&P 500'},
        'Slow Uptrend': {'returns': strategies['trend_up_returns']['Daily_Return'], 'instrument': 'S&P 500'},
        'trend_up_down_returns': {'returns': strategies['trend_up_down_returns']['Daily_Return'], 'instrument': 'S&P 500'},
        'forecast_ls': {'returns': strategies['forecast_ls']['Daily_Return'], 'instrument': 'S&P 500'}
    }
    median = ['US 5 Year T-Note Futures Historical Data.csv', 'Gold_data.csv', 'Copper_data.csv', 'Lean Hogs.csv', 'Natural_Gas_data.csv']
    for file in median:
        print(file)
        csv_path = os.path.join(os.path.dirname(__file__), '../individual_data', file)
        df = pd.read_csv(csv_path)
        
        # Prepare data
        df.columns = [col.capitalize() for col in df.columns]
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)  # ← Add this line
        
        multi[file] = {'returns': df['Close'].pct_change().dropna(), 'instrument': file}
    
    res = calc_table.create_multi_strategy_summary(multi)
    res.to_csv('multi_strategy_summary.csv')

if __name__ == "__main__":
        # Initialize the algorithm
    # Test function to validate your fixes
    main()