from copy import deepcopy
import numpy as np
from sklearn.base import BaseEstimator

from boomlet.utils.collection import group_by


class BinningEstimator(BaseEstimator):
    """Creates a separate estimator for each different value of a feature. Returns 0 if the value didn't appear in the training set."""

    def __init__(self, clf, meta_index):
        self.clf = deepcopy(clf)
        self.meta_index = meta_index

    def fit(self, X, y=None):
        grouped = group_by(range(X.shape[0]), lambda i: X[i, self.meta_index])

        self.clfs_ = {}
        for k, indices in grouped.items():
            tmp_clf = deepcopy(self.clf)
            if y is None:
                tmp_clf.fit(X[indices])
            else:
                tmp_clf.fit(X[indices], y[indices])
            self.clfs_[k] = tmp_clf
        return self

    def binning_apply(self, name, X, *args, **kwargs):
        grouped = group_by(range(X.shape[0]), lambda i: X[i, self.meta_index])

        output = None
        for k, indices in grouped.items():
            if k in self.clfs_:
                values = getattr(self.clfs_[k], name)(X[indices], *args, **kwargs)
                if output is None:
                    if len(values.shape) == 1:
                        output = np.zeros(X.shape[0])
                    else:
                        output = np.zeros([X.shape[0], values.shape[1]])
                output[indices] = values
        if output is None:
            raise Exception("no keys in test set present in training set")
        return output

    def predict(self, X):
        return self.binning_apply("predict", X)

    def predict_proba(self, X):
        return self.binning_apply("predict_proba", X)

    def transform(self, X):
        return self.binning_apply("transform", X)
