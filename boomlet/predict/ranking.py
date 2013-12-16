import itertools
import time
import random
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier


class SGDRanker(BaseEstimator):

    """ Ranking predictor using stochastic gradient descent

    TODO:
    -allow configurable parameters for classifier
    -seed random state
    """

    def __init__(self, seconds=10):
        self.clf = SGDClassifier(loss='hinge')
        self.clf.fit_intercept = False
        self.clf.classes_ = np.array([-1, 1])
        self.seconds = seconds

    def fit(self, X, y):
        rows = X.shape[0]
        start_time = time.time()
        for i in itertools.count():
            if time.time() - start_time > self.seconds:
                return self
            idx1 = random.randint(0, rows - 1)
            idx2 = random.randint(0, rows - 1)
            y1, y2 = y[idx1], y[idx2]
            if y1 == y2:
                continue
            self.clf.partial_fit(X[idx1] - X[idx2], np.sign(y1 - y2))

    def predict(self, X):
        return np.dot(X, self.clf.coef_.T)
