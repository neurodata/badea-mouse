import numpy as np

from scipy.stats import rankdata

import networkx as nx


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

def binarize(arrs, steps=0.01):
    """Finds the highest quantile value in which all the arrays are connected"""
    thresholded = []
    for arr in arrs:
        for percent in np.linspace(0, 1, int(1 / steps) + 1)[1:]:
            percentile = np.quantile(np.abs(arr), 1 - percent)
            s = arr.copy()
            s[np.abs(s) < percentile] = 0

            if nx.is_connected(nx.from_numpy_array(s)):
                thresholded.append(s)
                break

    binarized = []
    for arr in thresholded:
        arr[arr != 0] = 1
        binarized.append(arr)
    return binarized
            
            
def threshold(arrs, steps=0.01):
    """Finds the highest quantile value in which all the arrays are connected"""
    for percent in np.linspace(0, 1, int(1 / steps) + 1)[1:]:
        percentile = np.quantile(np.abs(arrs), 1 - percent)
        binarized = []
        for v in arrs:
            s = v.copy()
            s[np.abs(s) < percentile] = 0
            binarized.append(s)

        connected = [nx.is_connected(nx.from_numpy_array(v)) for v in binarized]
        if np.all(connected):
            return binarized
        else:
            continue
            
def threshold2(arrs, steps=0.01):
    """Finds the highest quantile value in which all the arrays are connected"""
    thresholded = []
    for arr in arrs:
        for percent in np.linspace(0, 1, int(1 / steps) + 1)[1:]:
            percentile = np.quantile(np.abs(arr), 1 - percent)
            s = arr.copy()
            s[np.abs(s) < percentile] = 0

            if nx.is_connected(nx.from_numpy_array(s)):
                thresholded.append(s)
                break

    return thresholded
