import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class OrthogonalPolynomialTransformer(BaseEstimator, TransformerMixin):
    """
    creates an orthogonal basis equivalent to a polynomial transform, so that variables are uncorrelated

    adapted from: http://davmre.github.io/python/2013/12/15/orthogonal_poly/

    UNTESTED
    """
    def __init__(self, degree=2):
        self.degree = degree

    def fit(self, X, y=None):
        degree = self.degree
        assert len(X.shape) == 1, X.shape
        assert degree < len(np.unique(X)), "'degree' must be less than number of unique points"
        assert degree > 1, degree
        xbar = np.mean(X)
        X = X - xbar
        X = np.fliplr(np.vander(X, degree + 1))
        q,r = np.linalg.qr(X)

        z = np.diag(np.diag(r))
        raw = np.dot(q, z)

        norm2 = np.sum(raw ** 2, axis=0)
        print X.shape
        print raw.shape
        alpha = (np.sum((raw ** 2) * X, axis=0) / norm2 + xbar)[:degree]
        self.norm2_, self.alpha_ = norm2, alpha
        return self

    def transform(self, X):
        assert len(X.shape) == 1, X.shape
        degree, alpha, norm2 = self.degree, self.alpha_, self.norm2_
        Z = np.empty((len(X), degree + 1))
        Z[:,0] = 1
        if degree > 0:
            Z[:, 1] = X - alpha[0]
        if degree > 1:
          for i in np.arange(1,degree):
              Z[:, i+1] = (X - alpha[i]) * Z[:, i] - (norm2[i] / norm2[i-1]) * Z[:, i-1]
        Z /= np.sqrt(norm2)
        return Z
