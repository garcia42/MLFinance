import numpy as np

def legendre_3(n):
    """
    Compute first, second, and third-order normalized orthogonal Legendre coefficients 
    for n data points.
    
    Args:
        n: Number of points
        
    Returns:
        c1, c2, c3: Arrays containing the coefficients
    """
    # Initialize arrays
    c1 = np.zeros(n)
    c2 = np.zeros(n)
    c3 = np.zeros(n)
    
    # Compute c1
    for i in range(n):
        c1[i] = 2.0 * i / (n - 1.0) - 1.0
    
    # Normalize c1
    c1 /= np.sqrt(np.sum(c1 * c1))
    
    # Compute c2
    c2 = c1 * c1
    mean = np.mean(c2)
    c2 -= mean  # Center it
    c2 /= np.sqrt(np.sum(c2 * c2))  # Normalize to unit length
    
    # Compute c3
    c3 = c1 * c1 * c1
    mean = np.mean(c3)
    c3 -= mean  # Center it
    c3 /= np.sqrt(np.sum(c3 * c3))  # Normalize to unit length
    
    # Remove the projection of c1 from c3
    proj = np.sum(c1 * c3)
    c3 -= proj * c1
    c3 /= np.sqrt(np.sum(c3 * c3))  # Renormalize
    
    return c1, c2, c3
