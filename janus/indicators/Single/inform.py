import numpy as np
from typing import Optional

class Entropy:
    """Class for calculating entropy of time series data."""
    
    def __init__(self, wordlen: int):
        """Initialize entropy calculator.
        
        Args:
            wordlen: Word length for pattern analysis. Number of bins will be 2^wordlen
        """
        if wordlen < 1:
            self.ok = False
            self.bins = None
            return
            
        self.wordlen = wordlen
        self.nbins = 2 ** wordlen
        self.bins = np.zeros(self.nbins, dtype=np.int32)
        self.ok = True
        
    def entropy(self, x: np.ndarray, wordlen: int) -> float:
        """Calculate entropy of a time series.
        
        Args:
            x: Time series data in reverse chronological order
            wordlen: Word length for pattern analysis
            
        Returns:
            Normalized entropy value between 0 and 1
        """
        # Reset bins
        self.bins.fill(0)
        nx = len(x)
        
        # Count patterns
        for i in range(wordlen, nx):
            # Build binary pattern
            pattern = 0
            for j in range(wordlen):
                pattern *= 2
                if x[i-j-1] > x[i-j]:
                    pattern += 1
            self.bins[pattern] += 1
        
        # Calculate probabilities and entropy
        n = nx - wordlen
        probabilities = self.bins / n
        # Only include non-zero probabilities in entropy calculation
        probabilities = probabilities[probabilities > 0]
        entropy = -np.sum(probabilities * np.log(probabilities))
        
        # Normalize by maximum possible entropy
        return entropy / np.log(2 ** wordlen)


class MutInf:
    """Class for calculating mutual information in time series data."""
    
    def __init__(self, wordlen: int):
        """Initialize mutual information calculator.
        
        Args:
            wordlen: Word length for pattern analysis. Number of bins will be 2^(wordlen+1)
        """
        if wordlen < 1:
            self.ok = False
            self.bins = None
            return
            
        self.wordlen = wordlen
        self.nbins = 2 ** (wordlen + 1)  # Need twice as many bins as Entropy
        self.bins = np.zeros(self.nbins, dtype=np.int32)
        self.ok = True
        
    def mut_inf(self, x: np.ndarray, wordlen: int) -> float:
        """Calculate mutual information between current and historical values.
        
        Follows TradeStation convention where current point has subscript zero
        and the series is time-reversed.
        
        Args:
            x: Time series data in reverse chronological order
            wordlen: Word length for pattern analysis
            
        Returns:
            Mutual information value
        """
        # Reset bins
        self.bins.fill(0)
        nx = len(x)
        n = nx - wordlen - 1  # Number of usable cases
        m = self.nbins // 2   # Number of history categories
        
        # Initialize marginal probabilities for dependent variable
        dep_marg = np.zeros(2)
        
        # Count patterns and calculate dependent marginals
        for i in range(n):
            # Current value (dependent variable)
            current = 1 if x[i] > x[i+1] else 0
            dep_marg[current] += 1
            
            # Build pattern including history
            pattern = current
            for j in range(1, wordlen + 1):
                pattern *= 2
                if x[i+j] > x[i+j+1]:
                    pattern += 1
            self.bins[pattern] += 1
        
        # Convert counts to probabilities
        dep_marg /= n
        
        # Calculate mutual information
        MI = 0.0
        for i in range(m):
            # Calculate history marginal
            hist_marg = (self.bins[i] + self.bins[i+m]) / n
            
            # Calculate joint probabilities
            p0 = self.bins[i] / n      # Probability for current=0
            p1 = self.bins[i+m] / n    # Probability for current=1
            
            # Add to mutual information if probabilities are non-zero
            if p0 > 0:
                MI += p0 * np.log(p0 / (hist_marg * dep_marg[0]))
            if p1 > 0:
                MI += p1 * np.log(p1 / (hist_marg * dep_marg[1]))
                
        return MI
