import numpy as np
from scipy import stats

def mode_wrapper(*args, **kwargs):
    return stats.mode(*args, **kwargs)[0]


AGGREGATORS = {
    "mean": np.mean,
    "mode": mode_wrapper,
    "median": np.median,
    "max": np.max,
    "min": np.min,
}

def from_str(s):
    return AGGREGATORS[s]
