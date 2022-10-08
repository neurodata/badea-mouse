import numpy as np

def squareize(n, vec):
    """
    
    """
    out = np.ones((n, n))
    out[np.triu_indices_from(out)] = vec
    out[np.tril_indices_from(out)] = out.T[np.tril_indices_from(out)]

    return out