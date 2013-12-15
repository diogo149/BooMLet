from sklearn.base import BaseEstimator, TransformerMixin


class BaseWrapper(BaseEstimator):

    def __init__(self, clf):
        self.clf = clf

    def __getattr__(self, name):
        return getattr(self.clf, name)


class NoFit(BaseWrapper):
    def fit(self, X, y=None):
        pass


class RowSubset(BaseWrapper):
    def __init__(self, clf, subset):
        self.clf = clf
        self.subset = subset

    def fit(self, X, y = None):
        X = X[self.subset, :]
        y = y[self.subset, :] if y is not None else y
        self.clf.fit(X, y)
        return self


class ColumnSubset(BaseWrapper):
    def __init__(self, clf, subset):
        self.clf = clf
        self.subset = subset

    def fit(self, X, y=None):
        X = X[:, self.subset]
        self.fit(X, y)
        return self

    def predict(self, X):
        X = X[:, self.subset]
        return self.predict(X)

    def transform(self, X):
        X = X[:, self.subset]
        return self.transform(X)

    def preidct_proba(self, X):
        X = X[:, self.subset]
        return self.preidct_proba(X)

class To1D(BaseWrapper):
    """wraps an estimator that requires 1 dimensional input to take 2D input"""

    def fit(self, X, y=None):
        assert X.shape[1] == 1, X.shape
        self.fit(X.flatten(), y)
        return self

    def predict(self, X):
        assert X.shape[1] == 1, X.shape
        return self.predict(X.flatten())

    def transform(self, X):
        assert X.shape[1] == 1, X.shape
        return self.transform(X.flatten())

    def preidct_proba(self, X):
        assert X.shape[1] == 1, X.shape
        return self.preidct_proba(X.flatten())


class From1D(BaseWrapper):
    """wraps an estimator that outputs 1 dimensional input to output in 2D"""

    def predict(self, X):
        return self.predict(X).reshape(-1, 1)

    def transform(self, X):
        return self.transform(X).reshape(-1, 1)

    def preidct_proba(self, X):
        return self.preidct_proba(X).reshape(-1, 1)
