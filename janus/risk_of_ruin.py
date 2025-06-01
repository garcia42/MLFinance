import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def monte_carlo_risk_of_ruin(
    starting_balance=1000,
    win_probability=0.65,
    avg_win=75,
    avg_loss=125,
    num_trades=1000,
    num_simulations=1000,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    # Store all simulation paths
    all_paths = np.zeros((num_simulations, num_trades + 1))
    all_paths[:, 0] = starting_balance
    
    # Run simulations
    ruined_count = 0
    for sim in range(num_simulations):
        balance = starting_balance
        
        for trade in range(1, num_trades + 1):
            # Generate random outcome
            if np.random.random() < win_probability:
                balance += avg_win
            else:
                balance -= avg_loss
            
            # Record balance
            all_paths[sim, trade] = max(balance, 0)  # Cap at zero to prevent negative balances
            
            # Check for ruin
            if balance <= 0:
                ruined_count += 1
                break
    
    # Calculate statistics
    risk_of_ruin = (ruined_count / num_simulations) * 100
    final_balances = all_paths[:, -1]
    avg_final_balance = np.mean(final_balances[final_balances > 0]) if np.any(final_balances > 0) else 0
    
    return {
        "risk_of_ruin_percent": risk_of_ruin,
        "ruined_count": ruined_count,
        "avg_final_balance": avg_final_balance,
        "paths": all_paths,
        "parameters": {
            "starting_balance": starting_balance,
            "win_probability": win_probability,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "num_trades": num_trades,
            "num_simulations": num_simulations
        }
    }
