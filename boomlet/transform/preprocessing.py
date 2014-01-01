import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class InvalidRemover(BaseEstimator, TransformerMixin):

    """
    Removes columns with invalid (infinite or nan) values, and only
    transforming will zero out invalid values.

    Looks for invalid values twice because the columns with invalid
    values in the training set may be different than in the test set.
    """

    def fit(self, X, y=None):
        self.valid_ = (np.isinf(X) + np.isnan(X)).sum(axis=0) == 0
        return self

    def transform(self, X):
        X_valid = X[:, self.valid_]
        # this assumes that the previous statement makes a copy
        # and we aren't mutating the original X
        X_valid[np.isinf(X_valid)] = 0.0
        X_valid[np.isnan(X_valid)] = 0.0
        return X_valid


class NearZeroVarianceFilter(BaseEstimator, TransformerMixin):

    """ Removes columns with standard deviation below a threshold
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.keep_cols_ = X.std(axis=0) > self.threshold
        return self

    def transform(self, X):
        return X[:, self.keep_cols_]


class InfinityReplacer(BaseEstimator, TransformerMixin):

    """
    Replaces values of infinity with a value, and negative of that value
    for negative infinity.

    (default: 2 ** 30 - 1 for positive infinity)

    Stateless.
    """

    def __init__(self, replace_val=int((2 << 30) - 1)):
        self.replace_val = replace_val

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tmp = X.copy()
        X_tmp[X == float("inf")] = self.replace_val
        X_tmp[X == float("-inf")] = -self.replace_val
        return X_tmp
