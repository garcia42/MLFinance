import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass

class JANUS:
    """
    Python implementation of Gary Anderson's JANUS indicator family.
    
    This class implements a collection of cross-market analysis tools
    focusing on relative strength, relative momentum, and market dynamics.
    """
    
    def __init__(self, 
                 nbars: int,           # Number of bars in market history
                 n_markets: int,       # Number of markets
                 lookback: int,        # Number of bars in lookback window
                 spread_tail: float,   # Fraction of markets in each tail for spread
                 min_CMA: int,         # Minimum lookback for CMA
                 max_CMA: int):        # And maximum
        """Initialize the JANUS calculator"""
        
        self.nbars = nbars
        self.n_returns = nbars - 1
        self.n_markets = n_markets
        self.lookback = lookback
        self.spread_tail = spread_tail
        self.min_CMA = min_CMA
        self.max_CMA = max_CMA
        
        # Initialize arrays
        self.index = np.zeros(lookback)
        self.sorted = np.zeros(max(lookback, n_markets))
        # iwork = original_indices if emphasizing its role in preserving the original positions
        self.iwork = np.zeros(n_markets, dtype=int)
        
        # Main data arrays
        self.returns = np.zeros((n_markets, self.n_returns))
        self.mkt_index_returns = np.zeros(self.n_returns)
        self.dom_index_returns = np.zeros(self.n_returns)
        
        # CMA related arrays
        self.CMA_alpha = np.zeros(max_CMA - min_CMA + 1)
        self.CMA_smoothed = np.zeros(max_CMA - min_CMA + 1)
        self.CMA_equity = np.zeros(max_CMA - min_CMA + 1)
        
        # RS/RM related arrays
        self.rs = np.zeros((self.n_returns, n_markets))
        self.rs_fractile = np.zeros((self.n_returns, n_markets))
        self.rs_lagged = np.zeros((self.n_returns, n_markets))
        self.rs_leader = np.zeros(self.n_returns)
        self.rs_laggard = np.zeros(self.n_returns)
        
        #It's "out-of-sample" in the sense that it includes all markets, not just the ones identified as leaders or laggards, giving a broader market context for the strategy's performance.
        self.oos_avg = np.zeros(self.n_returns)
        self.rm_leader = np.zeros(self.n_returns)
        self.rm_laggard = np.zeros(self.n_returns)
        
        # RSS related arrays
        self.rss = np.zeros(self.n_returns)
        self.rss_change = np.zeros(self.n_returns)
        
        # DOM/DOE related arrays
        self.dom = np.zeros((self.n_returns, n_markets))
        self.doe = np.zeros((self.n_returns, n_markets))
        self.dom_index = np.zeros(self.n_returns)
        self.doe_index = np.zeros(self.n_returns)
        self.dom_sum = np.zeros(n_markets)
        self.doe_sum = np.zeros(n_markets)
        
        # RM related arrays
        self.rm = np.zeros((self.n_returns, n_markets))
        self.rm_fractile = np.zeros((self.n_returns, n_markets))
        self.rm_lagged = np.zeros((self.n_returns, n_markets))
        
        # CMA OOS arrays
        self.CMA_OOS = np.zeros(self.n_returns)
        self.CMA_leader_OOS = np.zeros(self.n_returns)
        
        # Success flag
        self.ok = True
        
    def prepare(self, prices: List[np.ndarray]) -> None:
        """
        Prepare market histories and compute returns and index returns.
        This must be the first routine called to process a price history.
        
        Parameters:
        -----------
        prices : List[np.ndarray]
            List of price arrays for each market
        """
        # Compute returns matrix
        for i, price in enumerate(prices):
            self.returns[i, :] = np.log(price[1:] / price[:-1])
            
        # Compute median returns across markets (the index)
        for i in range(self.n_returns):
            self.mkt_index_returns[i] = np.median(self.returns[:, i])
            
    def compute_rs(self, lag: int) -> None:
        """
        Compute relative strength and its rank transform.
        
        Parameters:
        -----------
        lag : int
            Lag for calculations (0 for normal RS, 1 for performance spread)
        """
        self.rs_lookahead = lag
        
        for ibar in range(self.lookback-1, self.n_returns):
            # Get index window
            self.index = self.mkt_index_returns[ibar-self.lookback+1:ibar+1][::-1]
            
            # Compute median excluding lagged values
            valid_indices = self.index[lag:]
            median = np.median(valid_indices)
            
            # Compute offensive and defensive index components
            index_offensive = max(1e-30, np.sum(valid_indices[valid_indices >= median] - median))
            index_defensive = min(-1e-30, np.sum(valid_indices[valid_indices < median] - median))
            
            # Compute RS for each market
            for imarket in range(self.n_markets):
                market_returns = self.returns[imarket, ibar-self.lookback+1:ibar+1][::-1]
                valid_returns = market_returns[lag:]
                
                # Compute offensive and defensive market components
                market_offensive = np.sum(valid_returns[valid_indices >= median] - median)
                market_defensive = np.sum(valid_returns[valid_indices < median] - median)
                
                # Calculate RS
                rs_value = 70.710678 * (market_offensive / index_offensive - 
                                      market_defensive / index_defensive)
                rs_value = np.clip(rs_value, -200, 200)
                
                if lag == 0:
                    self.rs[ibar, imarket] = rs_value
                else:
                    self.rs_lagged[ibar, imarket] = rs_value
                    
                self.sorted[imarket] = rs_value
                self.iwork[imarket] = imarket
                
            # Compute fractiles
            order = np.argsort(self.sorted[:self.n_markets])
            if lag == 0:
                self.rs_fractile[ibar, self.iwork[order]] = np.arange(self.n_markets) / (self.n_markets - 1)

    def compute_rss(self) -> None:
        """Compute relative strength spread."""
        for ibar in range(self.lookback-1, self.n_returns):
            rs_values = self.rs[ibar]
            rs_sorted = np.sort(rs_values)
            
            k = max(0, int(self.spread_tail * (self.n_markets + 1)) - 1)
            n = k + 1
            
            spreads = rs_sorted[-n:] - rs_sorted[:n]
            self.rss[ibar] = np.mean(spreads)
            
            if ibar == self.lookback-1:
                self.rss_change[ibar] = 0.0
            else:
                self.rss_change[ibar] = self.rss[ibar] - self.rss[ibar-1]

    def compute_dom_doe(self) -> None:
        """Compute DOM and DOE indicators."""
        dom_index_sum = doe_index_sum = 0.0
        
        for ibar in range(self.lookback-1, self.n_returns):
            if self.rss_change[ibar] > 0.0:  # Width expanding
                dom_index_sum += self.mkt_index_returns[ibar]
                self.dom_sum += self.returns[:, ibar]
                
            elif self.rss_change[ibar] < 0.0:  # Width contracting
                doe_index_sum += self.mkt_index_returns[ibar]
                self.doe_sum += self.returns[:, ibar]
                
            self.dom[ibar] = self.dom_sum
            self.doe[ibar] = self.doe_sum
            self.dom_index[ibar] = dom_index_sum
            self.doe_index[ibar] = doe_index_sum

    def compute_rm(self, lag: int) -> None:
        """
        Compute relative momentum indicators.
        
        Parameters:
        -----------
        lag : int
            Lag for calculations
        """
        self.rm_lookahead = lag
        
        # First compute DOM index returns
        for ibar in range(self.n_returns):
            if ibar < self.lookback:
                self.dom_index_returns[ibar] = np.median(self.returns[:, ibar])
            else:
                dom_changes = self.dom[ibar] - self.dom[ibar-1]
                self.dom_index_returns[ibar] = np.median(dom_changes)
                
        # Main RM computation
        for ibar in range(self.lookback-1, self.n_returns):
            # Get DOM index window
            self.index = self.dom_index_returns[ibar-self.lookback+1:ibar+1][::-1]
            
            # Compute median excluding lagged values
            valid_indices = self.index[lag:]
            median = np.median(valid_indices)
            
            # Compute offensive and defensive index components
            index_offensive = max(1e-30, np.sum(valid_indices[valid_indices >= median] - median))
            index_defensive = min(-1e-30, np.sum(valid_indices[valid_indices < median] - median))
            
            # Compute RM for each market
            for imarket in range(self.n_markets):
                market_returns = []
                for i in range(self.lookback):
                    if ibar-i < self.lookback:
                        ret = self.returns[imarket, ibar-i]
                    else:
                        ret = self.dom[ibar-i, imarket] - self.dom[ibar-i-1, imarket]
                    market_returns.append(ret)
                    
                market_returns = np.array(market_returns[::-1])
                valid_returns = market_returns[lag:]
                
                # Compute offensive and defensive market components
                market_offensive = np.sum(valid_returns[valid_indices >= median] - median)
                market_defensive = np.sum(valid_returns[valid_indices < median] - median)
                
                # Calculate RM
                rm_value = 70.710678 * (market_offensive / index_offensive - 
                                      market_defensive / index_defensive)
                rm_value = np.clip(rm_value, -300, 300)
                
                if lag == 0:
                    self.rm[ibar, imarket] = rm_value
                else:
                    self.rm_lagged[ibar, imarket] = rm_value
                    
                self.sorted[imarket] = rm_value
                self.iwork[imarket] = imarket
                
            # Compute fractiles
            if lag == 0:
                order = np.argsort(self.sorted[:self.n_markets])
                self.rm_fractile[ibar, self.iwork[order]] = np.arange(self.n_markets) / (self.n_markets - 1)

    def compute_rs_ps(self) -> None:
        """Compute RS performance spread indicators."""
        for ibar in range(self.lookback-1, self.n_returns):
            rs_values = self.rs_lagged[ibar]
            order = np.argsort(rs_values)
            
            k = max(0, int(self.spread_tail * (self.n_markets + 1)) - 1)
            n = k + 1
            
            # Compute leader and laggard returns
            leader_markets = order[-n:]
            laggard_markets = order[:n]
            
            leader_returns = np.zeros(n)
            laggard_returns = np.zeros(n)
            
            for i, (leader, laggard) in enumerate(zip(leader_markets, laggard_markets)):
                for j in range(self.rs_lookahead):
                    leader_returns[i] += self.returns[leader, ibar-j]
                    laggard_returns[i] += self.returns[laggard, ibar-j]
                    
            self.rs_leader[ibar] = np.mean(leader_returns) / self.rs_lookahead
            self.rs_laggard[ibar] = np.mean(laggard_returns) / self.rs_lookahead
            
            # Compute out-of-sample average
            self.oos_avg[ibar] = np.mean(self.returns[:, ibar])

    def compute_rm_ps(self) -> None:
        """Compute RM performance spread indicators."""
        for ibar in range(self.lookback-1, self.n_returns):
            rm_values = self.rm_lagged[ibar]
            order = np.argsort(rm_values)
            
            k = max(0, int(self.spread_tail * (self.n_markets + 1)) - 1)
            n = k + 1
            
            # Compute leader and laggard returns
            leader_markets = order[-n:]
            laggard_markets = order[:n]
            
            leader_returns = np.zeros(n)
            laggard_returns = np.zeros(n)
            
            for i, (leader, laggard) in enumerate(zip(leader_markets, laggard_markets)):
                for j in range(self.rm_lookahead):
                    leader_returns[i] += self.returns[leader, ibar-j]
                    laggard_returns[i] += self.returns[laggard, ibar-j]
                    
            self.rm_leader[ibar] = np.mean(leader_returns) / self.rm_lookahead
            self.rm_laggard[ibar] = np.mean(laggard_returns) / self.rm_lookahead

    def compute_CMA(self) -> None:
        """
        Compute OOS equity using DOM relative to CMA smoothed.
        This computes CMA_OOS (entire universe) and CMA_leader_OOS (leaders only).
        
        Must be called after compute_DOM_DOE().
        The first valid DOM is at lookback, though DOM at lookback-1 was set to 0.
        For simplicity we act as if the first valid DOM is at lookback-1.
        So the first valid change in DOM is at lookback, the first valid
        in-sample equity is at lookback+1, and the first valid OOS equity
        is at lookback+2. So we initialized 'smoothed' to 0, which is the
        implicit DOM at lookback-1.
        """
        # Initialize CMA parameters
        for i in range(self.min_CMA, self.max_CMA + 1):
            idx = i - self.min_CMA
            self.CMA_alpha[idx] = 2.0 / (i + 1.0)
            self.CMA_smoothed[idx] = 0.0
            self.CMA_equity[idx] = 0.0
            
        # Initialize OOS arrays
        self.CMA_OOS[:self.lookback+2] = 0.0
        self.CMA_leader_OOS[:self.lookback+2] = 0.0
        
        # Main loop
        for ibar in range(self.lookback+2, self.n_returns):
            # This loop finds the lookback that maximizes universe gain.
            # We use this lookback for OOS universe and leader gain.
            # One might argue that leader gain should be used to find the best
            # lookback for OOS leader gain, but this much smaller number of 
            # markets would introduce instability into the optimization,
            # so we use the universe optimal lookback for both OOS gains.
            best_equity = float('-inf')
            ibest = self.min_CMA
            
            for i in range(self.min_CMA, self.max_CMA + 1):
                idx = i - self.min_CMA
                
                # Check if in up trend and update equity
                if self.dom_index[ibar-2] > self.CMA_smoothed[idx]:
                    self.CMA_equity[idx] += self.oos_avg[ibar-1]
                
                # Track best performing lookback
                if self.CMA_equity[idx] > best_equity:
                    best_equity = self.CMA_equity[idx]
                    ibest = i
                
                # Update smoothed DOMs through ibar-2
                self.CMA_smoothed[idx] = (self.CMA_alpha[idx] * self.dom_index[ibar-2] + 
                                        (1.0 - self.CMA_alpha[idx]) * self.CMA_smoothed[idx])
            
            # We now have the lookback that maximizes gain in the complete universe.
            # Check if in up trend using best lookback
            if self.dom_index[ibar-1] > self.CMA_smoothed[ibest - self.min_CMA]:
                # Up trend - cumulate universe OOS
                self.CMA_OOS[ibar] = self.oos_avg[ibar]
                
                # Now do the leader OOS
                # Find the rm leader markets known as of ibar-1
                for imarket in range(self.n_markets):
                    self.sorted[imarket] = self.rm[ibar-1, imarket]
                    self.iwork[imarket] = imarket
                
                # Sort markets by RM value
                order = np.argsort(self.sorted[:self.n_markets])
                for i in range(self.n_markets):
                    self.iwork[i] = self.iwork[order[i]]
                
                # Calculate number of markets in each tail
                k = max(0, int(self.spread_tail * (self.n_markets + 1)) - 1)
                n = k + 1  # This many leader markets at end of sorted array
                
                # Reset leader OOS value
                self.CMA_leader_OOS[ibar] = 0.0
                
                # Sum returns for leader markets
                while k >= 0:
                    isub = self.iwork[self.n_markets-1-k]  # Index of leader
                    self.CMA_leader_OOS[ibar] += self.returns[isub, ibar]
                    k -= 1
                
                # Average the leader returns
                self.CMA_leader_OOS[ibar] /= n

    def get_market_index(self) -> np.ndarray:
        """Get cumulative market index returns."""
        dest = np.zeros(self.nbars)
        cumsum = 0.0
        for i in range(self.lookback, self.nbars):
            cumsum += self.mkt_index_returns[i-1]
            dest[i] = cumsum
        return dest

    def get_dom_index(self) -> np.ndarray:
        """Get cumulative DOM index returns."""
        dest = np.zeros(self.nbars)
        cumsum = 0.0
        for i in range(self.lookback, self.nbars):
            cumsum += self.dom_index_returns[i-1]
            dest[i] = cumsum
        return dest

    def get_rs(self, ord_num: int) -> np.ndarray:
        """Get relative strength for a specific market (1-based indexing)."""
        dest = np.zeros(self.nbars)
        for i in range(self.lookback, self.nbars):
            dest[i] = self.rs[i-1, ord_num-1]
        return dest

    def get_rs_fractile(self, ord_num: int) -> np.ndarray:
        """Get relative strength fractile for a specific market (1-based indexing)."""
        dest = np.zeros(self.nbars)
        for i in range(self.lookback, self.nbars):
            dest[i] = self.rs_fractile[i-1, ord_num-1]
        return dest

    def get_rss(self) -> np.ndarray:
        """Get relative strength spread."""
        dest = np.zeros(self.nbars)
        for i in range(self.lookback, self.nbars):
            dest[i] = self.rss[i-1]
        return dest

    def get_rss_change(self) -> np.ndarray:
        """Get relative strength spread change."""
        dest = np.zeros(self.nbars)
        for i in range(self.lookback, self.nbars):
            dest[i] = self.rss_change[i-1]
        return dest

    def get_dom(self, ord_num: int = 0) -> np.ndarray:
        """Get DOM values. If ord_num is 0, returns index values."""
        dest = np.zeros(self.nbars)
        for i in range(self.lookback, self.nbars):
            if ord_num == 0:
                dest[i] = self.dom_index[i-1]
            else:
                dest[i] = self.dom[i-1, ord_num-1]
        return dest

    def get_doe(self, ord_num: int = 0) -> np.ndarray:
        """Get DOE values. If ord_num is 0, returns index values."""
        dest = np.zeros(self.nbars)
        for i in range(self.lookback, self.nbars):
            if ord_num == 0:
                dest[i] = self.doe_index[i-1]
            else:
                dest[i] = self.doe[i-1, ord_num-1]
        return dest

    def get_rm(self, ord_num: int) -> np.ndarray:
        """Get relative momentum for a specific market (1-based indexing)."""
        dest = np.zeros(self.nbars)
        for i in range(self.lookback, self.nbars):
            dest[i] = self.rm[i-1, ord_num-1]
        return dest

    def get_rm_fractile(self, ord_num: int) -> np.ndarray:
        """Get relative momentum fractile for a specific market (1-based indexing)."""
        dest = np.zeros(self.nbars)
        for i in range(self.lookback, self.nbars):
            dest[i] = self.rm_fractile[i-1, ord_num-1]
        return dest

    def get_oos_avg(self) -> np.ndarray:
        """Get cumulative out-of-sample average returns."""
        dest = np.zeros(self.nbars)
        cumsum = 0.0
        for i in range(self.lookback, self.nbars):
            cumsum += self.oos_avg[i-1]
            dest[i] = cumsum
        return dest

    def get_rs_leader_equity(self) -> np.ndarray:
        """Get cumulative RS leader returns."""
        dest = np.zeros(self.nbars)
        cumsum = 0.0
        for i in range(self.lookback, self.nbars):
            cumsum += self.rs_leader[i-1]
            dest[i] = cumsum
        return dest

    def get_rs_laggard_equity(self) -> np.ndarray:
        """Get cumulative RS laggard returns."""
        dest = np.zeros(self.nbars)
        cumsum = 0.0
        for i in range(self.lookback, self.nbars):
            cumsum += self.rs_laggard[i-1]
            dest[i] = cumsum
        return dest

    def get_rs_ps(self) -> np.ndarray:
        """Get cumulative RS performance spread."""
        dest = np.zeros(self.nbars)
        cumsum = 0.0
        for i in range(self.lookback, self.nbars):
            cumsum += self.rs_leader[i-1] - self.rs_laggard[i-1]
            dest[i] = cumsum
        return dest

    def get_rs_leader_advantage(self) -> np.ndarray:
        """Get cumulative RS leader advantage over market."""
        dest = np.zeros(self.nbars)
        cumsum = 0.0
        for i in range(self.lookback, self.nbars):
            cumsum += self.rs_leader[i-1] - self.oos_avg[i-1]
            dest[i] = cumsum
        return dest

    def get_rs_laggard_advantage(self) -> np.ndarray:
        """Get cumulative RS laggard advantage over market."""
        dest = np.zeros(self.nbars)
        cumsum = 0.0
        for i in range(self.lookback, self.nbars):
            cumsum += self.rs_laggard[i-1] - self.oos_avg[i-1]
            dest[i] = cumsum
        return dest

    def get_rm_leader_equity(self) -> np.ndarray:
        """Get cumulative RM leader returns."""
        dest = np.zeros(self.nbars)
        cumsum = 0.0
        for i in range(self.lookback, self.nbars):
            cumsum += self.rm_leader[i-1]
            dest[i] = cumsum
        return dest

    def get_rm_laggard_equity(self) -> np.ndarray:
        """Get cumulative RM laggard returns."""
        dest = np.zeros(self.nbars)
        cumsum = 0.0
        for i in range(self.lookback, self.nbars):
            cumsum += self.rm_laggard[i-1]
            dest[i] = cumsum
        return dest

    def get_rm_ps(self) -> np.ndarray:
        """Get cumulative RM performance spread."""
        dest = np.zeros(self.nbars)
        cumsum = 0.0
        for i in range(self.lookback, self.nbars):
            cumsum += self.rm_leader[i-1] - self.rm_laggard[i-1]
            dest[i] = cumsum
        return dest

    def get_rm_leader_advantage(self) -> np.ndarray:
        """Get cumulative RM leader advantage over market."""
        dest = np.zeros(self.nbars)
        cumsum = 0.0
        for i in range(self.lookback, self.nbars):
            cumsum += self.rm_leader[i-1] - self.oos_avg[i-1]
            dest[i] = cumsum
        return dest

    def get_rm_laggard_advantage(self) -> np.ndarray:
        """Get cumulative RM laggard advantage over market."""
        dest = np.zeros(self.nbars)
        cumsum = 0.0
        for i in range(self.lookback, self.nbars):
            cumsum += self.rm_laggard[i-1] - self.oos_avg[i-1]
            dest[i] = cumsum
        return dest

    def get_CMA_OOS(self) -> np.ndarray:
        """Get cumulative CMA out-of-sample returns."""
        dest = np.zeros(self.nbars)
        cumsum = 0.0
        for i in range(self.lookback, self.nbars):
            cumsum += self.CMA_OOS[i-1]
            dest[i] = cumsum
        return dest

    def get_leader_CMA_OOS(self) -> np.ndarray:
        """Get cumulative CMA leader out-of-sample returns."""
        dest = np.zeros(self.nbars)
        cumsum = 0.0
        for i in range(self.lookback, self.nbars):
            cumsum += self.CMA_leader_OOS[i-1]
            dest[i] = cumsum
        return dest
