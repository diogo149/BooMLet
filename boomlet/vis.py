import numpy as np
import pylab
from sklearn.decomposition import RandomizedPCA


def gbm_learning_curve(clf):
    pylab.plot(range(len(clf.oob_score_)),
               clf.oob_score_, 'o-r',
               range(len(clf.oob_score_)),
               clf.train_score_, 'o-b')


def plot2d(X, y, reducer=RandomizedPCA(2)):
    min_y = y.min()
    max_y = y.max()
    diff = max_y - min_y
    def scale(v):
        return ((v - min_y) / diff, 0.5, 0.5)

    new_X = reducer.fit_transform(X)
    col1, col2 = new_X.T
    for p1, p2, c in zip(col1, col2, y):
        pylab.scatter(p1, p2, color=scale(c))
