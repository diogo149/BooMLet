import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class InfinityRemover(BaseEstimator, TransformerMixin):

    """ Removes columns with infinite values.

        Looks for infinite values twice because the columns with infinite values in the training set may be different than in the test set. (fills these values with 0)
    """

    def fit(self, X, y=None):
        self.no_inf_ = np.isinf(X).sum(axis=0) == 0

    def transform(self, X):
        X_no_inf = X[:, self.no_inf_]
        # this assumes that the previous statement makes a copy
        # and we aren't mutating the original X
        X_no_inf[np.isinf(X_no_inf)] = 0.0
        return X_no_inf


class NearZeroVarianceFilter(BaseEstimator, TransformerMixin):

    """ Removes columns with standard deviation below a threshold
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.keep_cols_ = X.std(axis=0) > self.threshold

    def transform(self, X):
        return X[:, self.keep_cols_]