import numpy as np
from scipy.stats import norm, f
from indicators.Mult.trend_cmma import atr, trend, cmma
from indicators.Mult.evec_rs import evec_rs
from indicators.Mult.janus import JANUS

# Variable type constants
VAR_TREND_RANK = 1
VAR_TREND_MEDIAN = 2
VAR_TREND_RANGE = 3
VAR_TREND_IQR = 4
VAR_TREND_CLUMP = 5
VAR_CMMA_RANK = 6
VAR_CMMA_MEDIAN = 7
VAR_CMMA_RANGE = 8
VAR_CMMA_IQR = 9
VAR_CMMA_CLUMP = 10
VAR_MAHAL = 11
VAR_ABS_RATIO = 12
VAR_ABS_SHIFT = 13
VAR_COHERENCE = 14
VAR_DELTA_COHERENCE = 15
VAR_JANUS_INDEX_MARKET = 16
VAR_JANUS_INDEX_DOM = 17
VAR_JANUS_RAW_RS = 18
VAR_JANUS_FRACTILE_RS = 19
VAR_JANUS_DELTA_FRACTILE_RS = 20
VAR_JANUS_DELTA_FRACTILE_RM = 21
VAR_JANUS_RSS = 22
VAR_JANUS_DELTA_RSS = 23
VAR_JANUS_DOM = 24
VAR_JANUS_DOE = 25
VAR_JANUS_RAW_RM = 26
VAR_JANUS_FRACTILE_RM = 27
VAR_JANUS_RS_LEADER_EQUITY = 28
VAR_JANUS_RS_LAGGARD_EQUITY = 29
VAR_JANUS_RS_LEADER_ADVANTAGE = 30
VAR_JANUS_RS_LAGGARD_ADVANTAGE = 31
VAR_JANUS_RM_LEADER_EQUITY = 32
VAR_JANUS_RM_LAGGARD_EQUITY = 33
VAR_JANUS_RM_LEADER_ADVANTAGE = 34
VAR_JANUS_RM_LAGGARD_ADVANTAGE = 35
VAR_JANUS_RS_PS = 36
VAR_JANUS_RM_PS = 37
VAR_JANUS_CMA_OOS = 38
VAR_JANUS_OOS_AVG = 39
VAR_JANUS_LEADER_CMA_OOS = 40

def first_pctile(x: np.ndarray) -> float:
    """
    Compute percentile of the first case in an array.
    Returns value between 0-100 where smallest is 0.0 and largest is 100.0.
    """
    n = len(x)
    first = x[0]
    count = np.sum(x <= first)
    return 100.0 * count / (n - 1.0)

def median(x: np.ndarray) -> float:
    """Compute median of array."""
    return np.median(x)

def range_stat(x: np.ndarray) -> float:
    """Compute range (max - min) of array."""
    return np.max(x) - np.min(x)

def iqr(x: np.ndarray) -> float:
    """Compute interquartile range."""
    return np.percentile(x, 75) - np.percentile(x, 25)

def clump(x: np.ndarray) -> float:
    """
    Compute clumped 60 - returns 0.4 fractile if positive,
    0.6 fractile if negative, or 0 if they span zero.
    """
    n = len(x)
    x_sorted = np.sort(x)
    
    # Get 0.4 and 0.6 fractile indices
    k = int(0.4 * (n + 1)) - 1  # Index of 0.4 fractile
    k = max(0, k)
    m = n - k - 1  # Index of 0.6 fractile
    
    # Apply clump rule
    if x_sorted[k] > 0.0:
        return x_sorted[k]
    elif x_sorted[m] < 0.0:
        return x_sorted[m]
    else:
        return 0.0

class CompVar:
    """
    Compute multiple-market variables.
    
    This class implements the functionality from COMP_VAR.CPP in Python,
    reusing existing implementations from trend_cmma.py, evec_rs.py, and janus.py
    where possible.
    """
    
    def __init__(self, n: int, n_markets: int):
        """
        Initialize CompVar calculator.
        
        Parameters:
        -----------
        n : int
            Number of time periods
        n_markets : int
            Number of markets
        """
        self.n = n
        self.n_markets = n_markets
        
    def compute(self, var_num: int, param1: float, param2: float, param3: float, param4: float,
                open_prices: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> tuple[np.ndarray, int, int, int]:
        """
        Compute multiple-market variables.
        
        Parameters:
        -----------
        var_num : int
            Variable type to compute
        param1-4 : float
            Parameters for computation
        open_prices, high, low, close : np.ndarray
            Price data arrays of shape (n_markets, n)
            
        Returns:
        --------
        output : np.ndarray
            Computed variable values
        n_done : int
            Number of valid values computed
        first_date : int
            Index of first valid value
        last_date : int
            Index of last valid value
        """
        output = np.zeros(self.n)
        lookback = int(param1 + 0.5)
        
        # Handle trend/cmma rank/median/range/iqr/clump variables
        if var_num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:  # Replace with actual var_num constants
            atr_length = int(param2 + 0.5)
            front_bad = max(lookback-1, atr_length)
            
            # Initialize work array
            big_work = np.zeros((self.n, self.n_markets))
            
            # Compute trend or cmma for each market
            for i in range(self.n_markets):
                if var_num in [1, 2, 3, 4, 5]:  # TREND variables
                    result = trend(self.n, lookback, atr_length,
                                 open_prices[i], high[i], low[i], close[i])
                else:  # CMMA variables
                    result = cmma(self.n, lookback, atr_length,
                                open_prices[i], high[i], low[i], close[i])
                big_work[:, i] = result
                
            # Compute statistics across markets
            for i in range(front_bad, self.n):
                if var_num in [1, 6]:  # RANK
                    output[i] = first_pctile(big_work[i]) - 50.0
                elif var_num in [2, 7]:  # MEDIAN
                    output[i] = median(big_work[i])
                elif var_num in [3, 8]:  # RANGE
                    output[i] = range_stat(big_work[i])
                elif var_num in [4, 9]:  # IQR
                    output[i] = iqr(big_work[i])
                elif var_num in [5, 10]:  # CLUMP
                    output[i] = clump(big_work[i])
                    
            return output, self.n - front_bad, front_bad, self.n - 1
            
        # Handle Mahalanobis distance
        elif var_num == VAR_MAHAL:
            n_to_smooth = int(param2 + 0.5)
            front_bad = lookback
            
            for i in range(front_bad):
                output[i] = 0.0
                
            for i in range(front_bad, self.n):
                # Calculate mean of log changes
                work1 = np.zeros(self.n_markets)
                for j in range(self.n_markets):
                    work1[j] = np.log(close[j][i-1] / close[j][i-lookback]) / (lookback-1)
                    
                # Compute covariance matrix
                big_work = np.zeros((self.n_markets, self.n_markets))
                for j in range(self.n_markets):
                    for k in range(j+1):
                        for m in range(1, lookback):
                            diff_i = np.log(close[j][i-m] / close[j][i-m-1]) - work1[j]
                            diff_j = np.log(close[k][i-m] / close[k][i-m-1]) - work1[k]
                            big_work[j,k] += diff_i * diff_j
                        big_work[j,k] /= lookback - 1
                        if j != k:
                            big_work[k,j] = big_work[j,k]
                            
                try:
                    # Invert covariance matrix
                    big_work2 = np.linalg.inv(big_work)
                    
                    # Compute Mahalanobis distance
                    diff = np.zeros(self.n_markets)
                    for j in range(self.n_markets):
                        diff[j] = np.log(close[j][i] / close[j][i-1]) - work1[j]
                    
                    sum_val = 0.0
                    for j in range(self.n_markets):
                        for k in range(j+1):
                            value = diff[j] * diff[k] * big_work2[j,k]
                            if j == k:
                                sum_val += value
                            else:
                                sum_val += 2.0 * value
                                
                    # Final transformation
                    k = lookback - 1 - self.n_markets
                    sum_val *= (lookback - 1.0) * k
                    sum_val /= self.n_markets * (lookback - 2.0) * lookback
                    sum_val = f.cdf(sum_val, self.n_markets, k)
                    
                    if sum_val > 0.99999:
                        sum_val = 0.99999
                    if sum_val < 0.5:
                        sum_val = 0.5
                    output[i] = np.log(sum_val / (1.0 - sum_val))
                    
                except np.linalg.LinAlgError:
                    output[i] = 0.0
                    
            # Smooth if requested
            if n_to_smooth > 1:
                alpha = 2.0 / (n_to_smooth + 1.0)
                for i in range(front_bad+1, self.n):
                    output[i] = alpha * output[i] + (1.0 - alpha) * output[i-1]
                    
            return output, self.n - front_bad, front_bad, self.n - 1
            
        # Handle ABS_RATIO, ABS_SHIFT, COHERENCE, DELTA_COHERENCE
        elif var_num in [VAR_ABS_RATIO, VAR_ABS_SHIFT, VAR_COHERENCE, VAR_DELTA_COHERENCE]:
            front_bad = lookback - 1
            
            for i in range(front_bad):
                output[i] = 0.0
                
            alpha = 2.0 / (lookback / 2.0 + 1.0)
            
            for i in range(front_bad, self.n):
                # Calculate mean of log changes
                work1 = np.zeros(self.n_markets)
                for j in range(self.n_markets):
                    work1[j] = np.log(close[j][i] / close[j][i-lookback+1]) / (lookback-1)
                    
                # Compute covariance/correlation matrix
                big_work = np.zeros((self.n_markets, self.n_markets))
                np.fill_diagonal(big_work, 1e-60)  # Prevent division by zero
                
                for j in range(self.n_markets):
                    for k in range(j+1):
                        for m in range(lookback-1):
                            if close[j][i-m] <= 0 or close[j][i-m-1] <= 0:
                                print(f"Zero or negative price found in market {j}")
                            diff_i = np.log(close[j][i-m] / close[j][i-m-1]) - work1[j]
                            diff_j = np.log(close[k][i-m] / close[k][i-m-1]) - work1[k]
                            big_work[j,k] += diff_i * diff_j
                        big_work[j,k] /= lookback - 1
                        if j != k:
                            big_work[k,j] = big_work[j,k]
                            
                # Convert to correlation if needed
                if var_num in [VAR_COHERENCE, VAR_DELTA_COHERENCE]:
                    diag = np.sqrt(np.diag(big_work))
                    big_work /= diag[:, None]
                    big_work /= diag[None, :]
                    np.fill_diagonal(big_work, 1.0)
                    
                # Compute eigenvalues
                big_work = (big_work + big_work.T)/2
                evals = evec_rs(big_work, find_vec=False)
                
                if var_num in [VAR_ABS_RATIO, VAR_ABS_SHIFT]:
                    fraction = param2
                    k = max(1, int(fraction * self.n_markets + 0.5))
                    value = np.sum(evals[:k])
                    sum_val = np.sum(evals)
                    
                    if i == front_bad:
                        smoothed_numer = value
                        smoothed_denom = sum_val
                    else:
                        smoothed_numer = alpha * value + (1.0 - alpha) * smoothed_numer
                        smoothed_denom = alpha * sum_val + (1.0 - alpha) * smoothed_denom
                        
                    output[i] = 100.0 * smoothed_numer / (smoothed_denom + 1e-30)
                    
                elif var_num in [VAR_COHERENCE, VAR_DELTA_COHERENCE]:
                    factor = 0.5 * (self.n_markets - 1)
                    weights = (factor - np.arange(self.n_markets)) / factor
                    output[i] = 200.0 * (np.sum(weights * evals) / np.sum(evals) - 0.5)
                    
            # Handle ABS_SHIFT
            if var_num == VAR_ABS_SHIFT:
                long_lookback = int(param3 + 0.5)
                short_lookback = int(param4 + 0.5)
                if long_lookback < short_lookback + 1:
                    long_lookback = short_lookback + 1
                    
                for i in range(self.n-1, front_bad+long_lookback-1, -1):
                    short_sum = np.mean(output[i-short_lookback+1:i+1])
                    long_sum = np.mean(output[i-long_lookback+1:i+1])
                    variance = np.var(output[i-long_lookback+1:i+1], ddof=0)
                    
                    if variance <= 0.0:
                        output[i] = 0.0
                    else:
                        output[i] = (short_sum - long_sum) / np.sqrt(variance)
                        
                front_bad += long_lookback - 1
                output[:front_bad] = 0.0
                
            # Handle DELTA_COHERENCE
            elif var_num == VAR_DELTA_COHERENCE:
                long_lookback = int(param2 + 0.5)
                for i in range(self.n-1, front_bad+long_lookback-1, -1):
                    output[i] -= output[i-long_lookback]
                front_bad += long_lookback
                output[:front_bad] = 0.0
                
            return output, self.n - front_bad, front_bad, self.n - 1
            
        # Handle Janus variables
        elif var_num in range(VAR_JANUS_INDEX_MARKET, VAR_JANUS_LEADER_CMA_OOS + 1):
            lookback = int(param1 + 0.5)
            front_bad = lookback
            
            # Initialize output array
            output[:front_bad] = 0.0
            
            try:
                # Create JANUS instance
                janus = JANUS(self.n, self.n_markets, lookback, 0.1, 20, 60)
                
                # Prepare data
                janus.prepare(close)
                
                # Compute required components
                janus.compute_rs(0)
                janus.compute_rs(1)
                janus.compute_rss()
                janus.compute_dom_doe()
                janus.compute_rm(0)
                janus.compute_rm(1)
                janus.compute_rs_ps()
                janus.compute_rm_ps()
                janus.compute_CMA()
                
                # Get appropriate output based on variable type
                if var_num == VAR_JANUS_INDEX_MARKET:
                    temp = np.zeros(self.n)
                    output = janus.get_market_index()
                elif var_num == VAR_JANUS_INDEX_DOM:
                    temp = np.zeros(self.n)
                    output = janus.get_dom_index()
                elif var_num == VAR_JANUS_RAW_RS:
                    k = int(param2 + 0.5)
                    temp = np.zeros(self.n)
                    output = janus.get_rs(k)
                elif var_num == VAR_JANUS_FRACTILE_RS:
                    k = int(param2 + 0.5)
                    temp = np.zeros(self.n)
                    output = janus.get_rs_fractile(k)
                elif var_num == VAR_JANUS_DELTA_FRACTILE_RS:
                    k = int(param2 + 0.5)
                    temp = np.zeros(self.n)
                    output = janus.get_rs_fractile(k)
                    delta_k = int(param3 + 0.5)
                    temp = np.copy(output)
                    for i in range(self.n-1, front_bad+delta_k-1, -1):
                        output[i] = temp[i] - temp[i-delta_k]
                    front_bad += delta_k
                    output[:front_bad] = 0.0
                elif var_num == VAR_JANUS_RSS:
                    temp = np.zeros(self.n)
                    output = janus.get_rss()
                    n_to_smooth = int(param2 + 0.5)
                    if n_to_smooth > 1:
                        alpha = 2.0 / (n_to_smooth + 1.0)
                        for i in range(front_bad+1, self.n):
                            output[i] = alpha * output[i] + (1.0 - alpha) * output[i-1]
                elif var_num == VAR_JANUS_DELTA_RSS:
                    temp = np.zeros(self.n)
                    output = janus.get_rss_change()
                    n_to_smooth = int(param2 + 0.5)
                    if n_to_smooth > 1:
                        alpha = 2.0 / (n_to_smooth + 1.0)
                        for i in range(front_bad+1, self.n):
                            output[i] = alpha * output[i] + (1.0 - alpha) * output[i-1]
                elif var_num == VAR_JANUS_DOM:
                    k = int(param2 + 0.5)
                    temp = np.zeros(self.n)
                    output = janus.get_dom(k)
                elif var_num == VAR_JANUS_DOE:
                    k = int(param2 + 0.5)
                    temp = np.zeros(self.n)
                    output = janus.get_doe(k)
                elif var_num == VAR_JANUS_RAW_RM:
                    k = int(param2 + 0.5)
                    temp = np.zeros(self.n)
                    output = janus.get_rm(k)
                elif var_num == VAR_JANUS_FRACTILE_RM:
                    k = int(param2 + 0.5)
                    temp = np.zeros(self.n)
                    output = janus.get_rm_fractile(k)
                elif var_num == VAR_JANUS_DELTA_FRACTILE_RM:
                    k = int(param2 + 0.5)
                    temp = np.zeros(self.n)
                    output = janus.get_rm_fractile(k)
                    delta_k = int(param3 + 0.5)
                    temp = np.copy(output)
                    for i in range(self.n-1, front_bad+delta_k-1, -1):
                        output[i] = temp[i] - temp[i-delta_k]
                    front_bad += delta_k
                    output[:front_bad] = 0.0
                elif var_num == VAR_JANUS_RS_LEADER_EQUITY:
                    temp = np.zeros(self.n)
                    output = janus.get_rs_leader_equity()
                elif var_num == VAR_JANUS_RS_LAGGARD_EQUITY:
                    temp = np.zeros(self.n)
                    output = janus.get_rs_laggard_equity()
                elif var_num == VAR_JANUS_RS_LEADER_ADVANTAGE:
                    temp = np.zeros(self.n)
                    output = janus.get_rs_leader_advantage()
                elif var_num == VAR_JANUS_RS_LAGGARD_ADVANTAGE:
                    temp = np.zeros(self.n)
                    output = janus.get_rs_laggard_advantage()
                elif var_num == VAR_JANUS_RM_LEADER_EQUITY:
                    temp = np.zeros(self.n)
                    output = janus.get_rm_leader_equity()
                elif var_num == VAR_JANUS_RM_LAGGARD_EQUITY:
                    temp = np.zeros(self.n)
                    output = janus.get_rm_laggard_equity()
                elif var_num == VAR_JANUS_RM_LEADER_ADVANTAGE:
                    temp = np.zeros(self.n)
                    output = janus.get_rm_leader_advantage()
                elif var_num == VAR_JANUS_RM_LAGGARD_ADVANTAGE:
                    temp = np.zeros(self.n)
                    output = janus.get_rm_laggard_advantage()
                elif var_num == VAR_JANUS_RS_PS:
                    temp = np.zeros(self.n)
                    output = janus.get_rs_ps()
                elif var_num == VAR_JANUS_RM_PS:
                    temp = np.zeros(self.n)
                    output = janus.get_rm_ps()
                elif var_num == VAR_JANUS_CMA_OOS:
                    temp = np.zeros(self.n)
                    output = janus.get_CMA_OOS()
                elif var_num == VAR_JANUS_OOS_AVG:
                    temp = np.zeros(self.n)
                    output = janus.get_oos_avg()
                elif var_num == VAR_JANUS_LEADER_CMA_OOS:
                    temp = np.zeros(self.n)
                    output = janus.get_leader_CMA_OOS()
                    
                return output, self.n - front_bad, front_bad, self.n - 1
                
            except Exception as e:
                print(f"\n\nERROR... Computing JANUS: {str(e)}")
                output[:] = 0.0
                return output, 0, 0, self.n - 1
                
        return output, 0, 0, self.n - 1
