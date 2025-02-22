import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from enum import IntEnum

class ErrorCode(IntEnum):
    OK = 0
    INSUFFICIENT_MEMORY = 3

class FTI:
    """
    Implementation of Govinda Khalsa's FTI (Follow Through Index) indicator.
    Note: This implementation uses closing prices rather than high/low for channel widths
    for increased stability.
    """
    
    def __init__(self, 
                 use_log: bool,       # Take log of market prices?
                 min_period: int,     # Shortest period, at least 2
                 max_period: int,     # Longest period
                 half_length: int,    # Number of coefficients each side of center
                 block_length: int,   # Processing block length
                 beta: float,         # Fractile for width computation (0.8-0.99)
                 noise_cut: float):   # Fraction defining noise for FTI
        """Initialize the FTI calculator with given parameters."""
        
        self.error = ErrorCode.OK
        self.use_log = use_log
        self.min_period = max(2, min_period)
        self.max_period = max_period
        self.half_length = half_length
        self.lookback = block_length
        self.beta = beta
        self.noise_cut = noise_cut

        # Initialize arrays using numpy for better performance
        self.y = np.zeros(self.lookback + self.half_length)
        self.filtered = np.zeros(self.max_period - self.min_period + 1)
        self.width = np.zeros(self.max_period - self.min_period + 1)
        self.fti = np.zeros(self.max_period - self.min_period + 1)
        self.sorted = np.zeros(self.max_period - self.min_period + 1, dtype=int)
        self.diff_work = np.zeros(self.lookback)
        self.leg_work = np.zeros(self.lookback)
        self.sort_work = np.zeros(self.max_period - self.min_period + 1)
        
        # Pre-compute coefficients for all periods
        self.coefs = {}  # Dictionary to store coefficients for each period
        for period in range(self.min_period, self.max_period + 1):
            self.coefs[period] = self._find_coefs(period)

    def _find_coefs(self, period: int) -> np.ndarray:
        """Calculate filter coefficients for a specified period."""
        
        # Constants from Otnes' Applied Time Series Analysis
        d = np.array([0.35577019, 0.2436983, 0.07211497, 0.00630165])
        
        # Initialize coefficient array
        c = np.zeros(self.half_length + 1)
        
        # Calculate initial coefficients
        fact = 2.0 / period
        c[0] = fact
        
        fact *= np.pi
        for i in range(1, self.half_length + 1):
            c[i] = np.sin(i * fact) / (i * np.pi)
        
        # Taper the end point
        c[self.half_length] *= 0.5
        
        # Apply window function
        sumg = c[0]
        for i in range(1, self.half_length + 1):
            # Calculate window value
            window_sum = d[0]
            fact = i * np.pi / self.half_length
            for j in range(1, 4):
                window_sum += 2.0 * d[j] * np.cos(j * fact)
            
            c[i] *= window_sum
            sumg += 2.0 * c[i]
        
        # Normalize coefficients
        c /= sumg
        return c

    def process(self, data: np.ndarray, chronological: bool = True) -> None:
        """Process market data to calculate FTI values.
        
        Args:
            data: Array of price data
            chronological: If True, data is in chronological order
        """
        # Prepare data array
        if self.use_log:
            self.y[:self.lookback] = np.log(data[:self.lookback] if chronological 
                                          else data[self.lookback-1::-1])
        else:
            self.y[:self.lookback] = (data[:self.lookback] if chronological 
                                    else data[self.lookback-1::-1])

        # Fit least-squares line to recent data and extend
        x = np.arange(-self.half_length, 1)
        y = self.y[self.lookback-self.half_length-1:self.lookback]
        coeffs = np.polyfit(x, y, 1)
        x_ext = np.arange(1, self.half_length + 1)
        self.y[self.lookback:] = np.polyval(coeffs, x_ext)

        # Process each period
        for period in range(self.min_period, self.max_period + 1):
            idx = period - self.min_period
            coefs = self.coefs[period]
            
            # Apply filter to get filtered values and differences
            filtered_vals = np.zeros(self.lookback - self.half_length)
            for i in range(self.half_length, self.lookback):
                # Convolve with filter coefficients
                filtered_val = coefs[0] * self.y[i]
                for j in range(1, self.half_length + 1):
                    filtered_val += coefs[j] * (self.y[i+j] + self.y[i-j])
                filtered_vals[i-self.half_length] = filtered_val
                
                if i == self.lookback - 1:
                    self.filtered[idx] = filtered_val
            
            # Calculate differences for width calculation
            self.diff_work[:self.lookback-self.half_length] = np.abs(
                self.y[self.half_length:self.lookback] - filtered_vals)
            
            # Find legs and calculate FTI
            legs = []
            extreme_val = filtered_vals[0]
            extreme_type = 0  # 0=undefined, 1=high, -1=low
            
            for i in range(1, len(filtered_vals)):
                curr_val = filtered_vals[i]
                
                if extreme_type == 0:
                    if curr_val > extreme_val:
                        extreme_type = -1
                    elif curr_val < extreme_val:
                        extreme_type = 1
                
                elif i == len(filtered_vals) - 1:
                    if extreme_type:
                        legs.append(abs(extreme_val - curr_val))
                
                else:
                    if (extreme_type == 1 and curr_val > filtered_vals[i-1] or
                        extreme_type == -1 and curr_val < filtered_vals[i-1]):
                        legs.append(abs(extreme_val - filtered_vals[i-1]))
                        extreme_type *= -1
                        extreme_val = filtered_vals[i-1]
            
            # Calculate width
            self.width[idx] = np.percentile(self.diff_work[:self.lookback-self.half_length], 
                                          self.beta * 100)
            
            # Calculate FTI
            if legs:
                longest_leg = max(legs)
                noise_level = self.noise_cut * longest_leg
                significant_legs = [leg for leg in legs if leg > noise_level]
                if significant_legs:
                    self.fti[idx] = np.mean(significant_legs) / (self.width[idx] + 1e-5)
        
        # Sort FTI values
        local_maxima = []
        for i in range(len(self.fti)):
            if (i == 0 or i == len(self.fti)-1 or 
                (self.fti[i] >= self.fti[i-1] and self.fti[i] >= self.fti[i+1])):
                local_maxima.append((i, self.fti[i]))
        
        sorted_maxima = sorted(local_maxima, key=lambda x: x[1], reverse=True)
        self.sorted[:len(sorted_maxima)] = [x[0] for x in sorted_maxima]

    def get_filtered_value(self, period: int) -> float:
        """Get filtered value for specific period."""
        return self.filtered[period - self.min_period]
    
    def get_width(self, period: int) -> float:
        """Get width value for specific period."""
        return self.width[period - self.min_period]
    
    def get_fti(self, period: int) -> float:
        """Get FTI value for specific period."""
        return self.fti[period - self.min_period]
    
    def get_sorted_index(self, which: int) -> int:
        """Get index of nth largest FTI value."""
        return self.sorted[which]