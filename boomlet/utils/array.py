import itertools

import numpy as np


def _to_2d(m):
    if not isinstance(m, np.ndarray):
        m = np.array(m)
    if len(m.shape) == 2:
        return m
    elif len(m.shape) == 1:
        return m.reshape(-1, 1)
    else:
        raise Exception("Improper shape: {}".fotmat(m.shape))


def to_2d(*Xs):
    """
    combine features into one 2D matrix
    """
    if len(Xs) == 1:
        return _to_2d(Xs[0])
    elif len(Xs) > 1:
        return np.hstack(map(_to_2d, Xs))
    else:
        raise Exception("to_2d can't be called with no arguments")


def column_combinations(*Xs):
    """
    creates a feature for each column combination in Xs

    e.g. for column a in A, column b in B, and column c in C,
    a * b * c will be in column_combinations(A, B, C)
    """
    assert len(Xs) > 1

    def col_generator(X):
        for col in X.T:
            yield col

    generator = itertools.product(*map(col_generator, Xs))
    return to_2d(*[np.product(cols, axis=0) for cols in generator])
