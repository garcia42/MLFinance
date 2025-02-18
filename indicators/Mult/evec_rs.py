import numpy as np

def evec_rs(mat_in, find_vec=True):
    """
    Compute eigenvalues and eigenvectors of a real symmetric matrix.
    
    Parameters:
    -----------
    mat_in : ndarray
        Input matrix (must be symmetric)
    find_vec : bool, optional
        Whether to compute eigenvectors (default True)
    
    Returns:
    --------
    eigenvalues : ndarray
        Array of eigenvalues in descending order
    eigenvectors : ndarray
        Matrix of eigenvectors (each column is an eigenvector)
        Only returned if find_vec=True
    """
    
    # Ensure matrix is symmetric
    n = mat_in.shape[0]
    if not np.allclose(mat_in, mat_in.T):
        print(mat_in)
        raise ValueError("Input matrix must be symmetric")
        
    # Compute eigenvalues and eigenvectors
    if find_vec:
        eigenvalues, eigenvectors = np.linalg.eigh(mat_in)
        
        # Sort in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Flip signs of eigenvectors if majority of elements are negative
        for i in range(n):
            if np.sum(eigenvectors[:, i] < 0) > n/2:
                eigenvectors[:, i] *= -1
                
        return eigenvalues, eigenvectors
    else:
        eigenvalues = np.linalg.eigvalsh(mat_in)
        return np.sort(eigenvalues)[::-1]