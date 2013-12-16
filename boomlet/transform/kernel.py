import math
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from boomlet.utils.estimators import gaussian_kernel_median_trick


class RandomizedFourierKernel(BaseEstimator, TransformerMixin):
    """
    Approximation of fourier basis kernel
    From: http://www.machinedlearnings.com/2013/08/cosplay.html

    Recommendations:
    -Scale and perform PCA
    -Estimate kernel bandwidth with the 'median trick'
    -Perform Logistic Regression w/ L2 loss

    TODO:
    -seed random state
    """

    def __init__(self, n_components=100, scale=None):
        self.n_components = n_components
        self.scale = scale

    def fit(self, X, y=None):
        if self.scale is not None:
            self.scale_ = self.scale
        else:
            self.scale_ = gaussian_kernel_median_trick(X)
        self.r_ = np.random.randn(X.shape[0], self.n_components)
        self.b_ = 2 * math.pi * np.random.uniform(size=self.n_components)
        return self

    def transform(self, X):
        return np.cos(self.scale_ * np.dot(X, self.r_) + self.b_)
