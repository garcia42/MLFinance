import numpy as np
from typing import List, Optional

class Entropy:
    """Class for calculating entropy of sequences."""
    
    def __init__(self, maxlen: int):
        """Initialize with maximum length of sequences to analyze.
        
        Args:
            maxlen: Maximum length of sequences that will be analyzed
        """
        self.bins = np.zeros(maxlen, dtype=int)
        self.ok = True  # Status flag
        
    def entropy(self, x: np.ndarray) -> float:
        """
        Calculate normalized entropy using adaptive binning based on sample size.
        
        Parameters:
        -----------
        x : np.ndarray
            Input array of values
            
        Returns:
        --------
        float
            Normalized entropy value between 0 and 1
        """
        n = len(x)
        
        # Determine number of bins based on sample size
        if n >= 10000:
            nbins = 20
        elif n >= 1000:
            nbins = 10
        elif n >= 100:
            nbins = 5
        else:
            nbins = 3
            
        # Find min and max
        xmin = np.min(x)
        xmax = np.max(x)
        
        # Calculate bin factor
        factor = (nbins - 0.00000000001) / (xmax - xmin + 1e-60)
        
        # Assign values to bins
        bins = np.minimum(
            (factor * (x - xmin)).astype(int),
            nbins - 1  # Ensure we don't exceed nbins
        )
        
        # Count occurrences in each bin
        counts = np.bincount(bins, minlength=nbins)
        
        # Calculate entropy
        # Only include non-zero counts to avoid log(0)
        p = counts[counts > 0] / n
        entropy_sum = -np.sum(p * np.log(p))
        
        # Normalize by maximum possible entropy (log of number of bins)
        return entropy_sum / np.log(nbins)

class MutInf:
    """Class for calculating mutual information between sequences."""
    
    def __init__(self, maxlen: int):
        """Initialize with maximum length of sequences to analyze.
        
        Args:
            maxlen: Maximum length of sequences that will be analyzed
        """
        self.bins = np.zeros(maxlen, dtype=int)
        self.ok = True
        
    def mut_inf(self, x: np.ndarray, wordlen: int) -> float:
        """Calculate mutual information.
        
        Args:
            x: Input sequence
            wordlen: Word length for mutual information calculation
            
        Returns:
            Calculated mutual information value
        """
        if len(x) < 2 * wordlen:
            self.ok = False
            return 0.0
            
        n = len(x)
        joint_counts = {}
        x_counts = {}
        y_counts = {}
        
        # Count frequencies
        for i in range(n - wordlen):
            x_word = tuple(x[i:i+wordlen])
            y_word = tuple(x[i+1:i+1+wordlen])
            
            joint_key = (x_word, y_word)
            joint_counts[joint_key] = joint_counts.get(joint_key, 0) + 1
            x_counts[x_word] = x_counts.get(x_word, 0) + 1
            y_counts[y_word] = y_counts.get(y_word, 0) + 1
        
        # Calculate mutual information
        total = sum(joint_counts.values())
        mi = 0.0
        for (x_word, y_word), joint_count in joint_counts.items():
            p_xy = joint_count / total
            p_x = x_counts[x_word] / total
            p_y = y_counts[y_word] / total
            mi += p_xy * np.log(p_xy / (p_x * p_y))
            
        return mi


class FTI:
    """Follow Through Index indicator implementation."""
    
    def __init__(self, use_log: bool, min_period: int, max_period: int, 
                 half_length: int, block_length: int, beta: float, noise_cut: float):
        """Initialize FTI calculator.
        
        Args:
            use_log: Whether to use log of prices
            min_period: Minimum period to consider (at least 2)
            max_period: Maximum period to consider
            half_length: Number of coefficients each side of center
            block_length: Length of blocks for processing
            beta: Fractile for width calculation (usually 0.8-0.99)
            noise_cut: Fraction of longest move defining noise
        """
        self.error = 0
        self.use_log = use_log
        self.min_period = max(2, min_period)
        self.max_period = max_period
        self.half_length = half_length
        self.lookback = block_length
        self.beta = beta
        self.noise_cut = noise_cut
        
        # Initialize arrays
        self.y = np.zeros(block_length + 2 * half_length)
        self.coefs = np.zeros(2 * half_length + 1)
        self.filtered = np.zeros(max_period - min_period + 1)
        self.width = np.zeros(max_period - min_period + 1)
        self.fti = np.zeros(max_period - min_period + 1)
        self.sorted = np.arange(max_period - min_period + 1)
        self.diff_work = np.zeros(block_length)
        self.leg_work = np.zeros(block_length)
        self.sort_work = np.zeros(max_period - min_period + 1)
        
    def find_coefs(self, period: int) -> np.ndarray:
        """Calculate filter coefficients for given period.
        
        Args:
            period: Period to calculate coefficients for
            
        Returns:
            Array of calculated coefficients
        """
        n = 2 * self.half_length + 1
        coefs = np.zeros(n)
        
        # Calculate coefficients using trigonometric functions
        for i in range(n):
            angle = 2 * np.pi * (i - self.half_length) / period
            if angle != 0:
                coefs[i] = np.sin(angle) / angle
            else:
                coefs[i] = 1.0
                
        # Apply Hamming window
        for i in range(n):
            coefs[i] *= 0.54 + 0.46 * np.cos(np.pi * (i - self.half_length) / self.half_length)
            
        # Normalize
        coefs /= np.sum(coefs)
        return coefs
        
    def process(self, prices: np.ndarray, chronological: bool = True):
        """Process price data to calculate FTI values.
        
        Args:
            prices: Array of price values
            chronological: Whether data is in chronological order
        """
        if len(prices) < self.lookback:
            self.error = 1
            return
            
        # Prepare data
        data = np.log(prices) if self.use_log else prices.copy()
        if not chronological:
            data = data[::-1]
            
        # Process each period
        for period in range(self.min_period, self.max_period + 1):
            idx = period - self.min_period
            
            # Get coefficients and apply filter
            coefs = self.find_coefs(period)
            filtered = np.convolve(data, coefs, mode='valid')
            
            self.filtered[idx] = filtered[-1]
            
            # Calculate width
            diffs = np.abs(np.diff(filtered))
            self.width[idx] = np.percentile(diffs, self.beta * 100)
            
            # Calculate FTI
            max_move = np.max(diffs)
            noise_threshold = max_move * self.noise_cut
            self.fti[idx] = np.mean(diffs > noise_threshold)
            
        # Sort FTI values
        self.sorted = np.argsort(-self.fti)
        
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
        return self.sorted[which] + self.min_period