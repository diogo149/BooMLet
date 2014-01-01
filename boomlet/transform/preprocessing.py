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
        X_valid = X.copy()
        if hasattr(self, "valid_"):
            X_valid = X_valid[:, self.valid_]
        # this assumes that the previous statement makes a copy
        # and we aren't mutating the original X
        X_valid[np.isinf(X_valid)] = 0.0
        X_valid[np.isnan(X_valid)] = 0.0
        return X_valid


class NearZeroVarianceFilter(BaseEstimator, TransformerMixin):
    """
    Removes columns with standard deviation below a threshold
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


class PercentileScaler(BaseEstimator, TransformerMixin):
    """
    Scales to a range where 0 is equivalent to a given lower
    percentile and 1 is equivalent to a given higher percentile
    """
    def __init__(self,
                 lower=0.03,
                 upper=0.97,
                 copy=True,
                 epsilon=1e-5,
                 squash=False):
        self.lower = lower
        self.upper = upper
        self.copy = copy
        self.epsilon = epsilon
        self.squash = squash

    def fit(self, X, y=None):
        assert 0.0 <= self.lower < self.upper <= 1.0
        l, u = np.percentile(X, [self.lower * 100, self.upper * 100], axis=0)
        self.subtract_ = l
        self.divide_ = u - l
        self.divide_[self.divide_ < self.epsilon] = self.epsilon
        return self

    def transform(self, X):
        if self.copy:
            X = X.copy()
        X -= self.subtract_
        X /= self.divide_
        if self.squash:
            X[X < 0] = 0
            X[X > 1] = 1
        return X
