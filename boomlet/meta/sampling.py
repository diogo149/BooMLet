import numpy as np

from copy import deepcopy
from sklearn.base import BaseEstimator
from sklearn.cross_validation import ShuffleSplit

from boomlet.utils import aggregators
from boomlet.utils.estimators import flexible_int


class RowSampler(BaseEstimator):

    """ Class that trains several models on a small sample of the data and returns a combination of their predictions.

    Make sure classifiers have the right set of classes, because due to sampling, not all classes may be represented in each training set.
    """

    def __init__(self,
                 clf,
                 aggregator="mean",
                 sample_size="sqrt",
                 n_iter=10,
                 random_state=None):
        self.clf = deepcopy(clf)
        self.sample_size = sample_size
        self.n_iter = n_iter
        self.random_state = random_state
        if isinstance(aggregator, str):
            self.aggregator = aggregators.from_str(aggregator)
        else:
            self.aggregator = aggregator

    def fit(self, X, y):
        train = np.array(X)
        assert len(train.shape) == 2
        assert len(y.shape) == 1
        ss = ShuffleSplit(
            n=X.shape[0],
            n_iter=self.n_iter,
            random_state=self.random_state,
            test_size=flexible_int(X.shape[0], self.sample_size)
        )
        self.clfs_ = []
        for _, indices in ss:
            tmp_clf = deepcopy(self.clf)
            tmp_clf.fit(train[indices], y[indices])
            self.clfs_.append(tmp_clf)
        return self

    def predict(self, X):
        data = []
        for clf in self.clfs_:
            data.append(clf.predict(X))
        return self.aggregator(data, axis=0)

    def predict_proba(self, X):
        data = []
        for clf in self.clfs_:
            data.append(clf.predict_proba(X))
        return self.aggregator(data, axis=0)


class ColSampler(BaseEstimator):

    """ Class that trains several models on a small sample of the features and returns a combination of their predictions.
    """

    def __init__(self,
                 clf,
                 aggregator="mean",
                 sample_size="sqrt",
                 n_iter=10,
                 random_state=None):
        self.clf = deepcopy(clf)
        self.sample_size = sample_size
        self.n_iter = n_iter
        self.random_state = random_state
        if isinstance(aggregator, str):
            self.aggregator = aggregators.from_str(aggregator)
        else:
            self.aggregator = aggregator

    def fit(self, X, y):
        train = np.array(X)
        assert len(train.shape) == 2
        assert len(y.shape) == 1
        ss = ShuffleSplit(
            n=X.shape[1],
            n_iter=self.n_iter,
            random_state=self.random_state,
            test_size=flexible_int(X.shape[1], self.sample_size)
        )
        self.indices_ = [indices for _, indices in ss]
        self.clfs_ = []
        for indices in self.indices:
            tmp_clf = deepcopy(self.clf)
            tmp_clf.fit(train[:, indices], y)
            self.clfs_.append(tmp_clf)
        return self

    def predict(self, X):
        data = []
        for indices, clf in zip(self.indices_, self.clfs_):
            data.append(clf.predict(X[:, indices]))
        return self.aggregator(data, axis=0)

    def predict_proba(self, X):
        data = []
        for indices, clf in zip(self.indices_, self.clfs_):
            data.append(clf.predict_proba(X[:, indices]))
        return self.aggregator(data, axis=0)
