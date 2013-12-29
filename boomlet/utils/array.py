import numpy as np


def to_2d(m):
    if len(m.shape) == 2:
        return m
    elif len(m.shape) == 1:
        return m.reshape(-1, 1)
    else:
        raise Exception("Improper shape: {}".fotmat(m.shape))
