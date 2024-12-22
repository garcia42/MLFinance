import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from FinancialMachineLearning.bet_sizing.bet_sizing import get_tstats_betsize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

from FinancialMachineLearning.sample_weights.bootstrapping import *

from scipy.stats import norm

from FinancialMachineLearning.labeling.labeling import *
from FinancialMachineLearning.bet_sizing.bet_sizing import get_tstats_betsize
from FinancialMachineLearning.bet_sizing.bet_sizing import avg_active_signals
from FinancialMachineLearning.bet_sizing.bet_sizing import discrete_signal, bet_size_dynamic
from FinancialMachineLearning.features.volatility import daily_volatility, daily_volatility_intraday

from train_model import build_model, Model

def bet_size(current_position: int, features: pd.DataFrame, new_row: pd.DataFrame, model: Model):
    '''
    https://github.com/QuantifiSogang/2024-02SeniorMLFinance/blob/801b519689e1ccd4990bd2170346b5019dc85b2f/Notes/Week11BetSizing/02BetSizingFromPredictedProbabilities.ipynb
   
        
    # 1. Get the predicted probability of profit from the model
    # 2. Determine the active signals and average them to find the recommended active signal
    
    Think of the probability as your BASE decision, and the signals as MULTIPLIERS.
    You normally invest max $100 in ingredients
    Your 75% probability suggests investing $75 (75% of max)
    The positive signals (0.5) boost this by 50%
    Final investment = $75 x 1.5 = $112.50

    Based on Ensemble bet sizing
    '''

    # Get predictions and probabilities
    current_prices = features['close']
    volatility = daily_volatility_intraday(current_prices, lookback = 60)
    side_predictions = model.side_model.predict(features)
    side_predictions = model.side_model.predict_proba(features)
    
    # Add side predictions to features for size model
    features_with_side = features.copy()
    features_with_side['side'] = side_predictions

    size_predictions = model.size_model.predict(features)

    # Generate forecasts
    forecasted_prices = generate_forecasted_prices(
        current_prices=current_prices,
        volatility=volatility,
        side_predictions=side_predictions,
        probabilities=size_predictions
    )

    # Then use with bet sizing
    # Start small for now with just 3, roughly 18k worth
    bet_sizes = bet_size_dynamic(
        current_pos=current_position,
        max_pos=3,
        market_price=current_prices,
        forecast_price=forecasted_prices
    )
    
    return bet_sizes.iloc[-1]['t_pos']  # or t_pos if you want the actual position size


def generate_forecasted_prices(current_prices, volatility, side_predictions, probabilities, pt_sl=[2, 1]):
    """
    Generate forecasted prices based on triple barrier method and probabilities
    
    Parameters:
    -----------
    current_prices : pd.Series
        Current prices with datetime index
    volatility : pd.Series
        Volatility estimates with same index as current_prices
    side_predictions : pd.Series
        Predicted trade direction (-1, 0, 1) with same index as current_prices
    probabilities : pd.Series
        Predicted probabilities (0 to 1) with same index as current_prices
        Higher probability means more confidence in the prediction
    pt_sl : list
        [profit_target_multiplier, stop_loss_multiplier]
        
    Returns:
    --------
    pd.Series
        Forecasted prices
    """
    profit_target_mult, stop_loss_mult = pt_sl
    
    # Initialize forecasted prices
    forecasted_prices = pd.Series(index=current_prices.index, dtype=float)
    
    for idx in current_prices.index:
        current_price = current_prices[idx]
        current_vol = volatility[idx]
        side = side_predictions[idx]
        prob = probabilities[idx]
        
        # Scale the price move by probability (0.5 = no conviction, 1.0 = max conviction)
        # Subtract 0.5 to center around neutral probability
        prob_scaling = 2 * (prob - 0.5) if prob >= 0.5 else 2 * (0.5 - prob)
        
        if side == 1:  # Long position
            # Scale forecast by probability
            forecast = current_price * (1 + profit_target_mult * current_vol * prob_scaling)
        elif side == -1:  # Short position
            # Scale forecast by probability
            forecast = current_price * (1 - stop_loss_mult * current_vol * prob_scaling)
        else:  # No position
            forecast = current_price
            
        forecasted_prices[idx] = forecast
        
    return forecasted_prices
