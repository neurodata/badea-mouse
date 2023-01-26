import numpy as np
from scipy.stats import rankdata

def squareize(n, vec):
    """ """
    out = np.ones((n, n))
    out[np.triu_indices_from(out)] = vec
    out[np.tril_indices_from(out)] = out.T[np.tril_indices_from(out)]

    return out

def rank(data):
    n = data.shape[0]
    vec = data[np.triu_indices_from(data)]
    
    ranked = rankdata(vec, method='ordinal', nan_policy='omit')
    
    out = squareize(n, ranked)
    
    return out