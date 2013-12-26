from collections import Counter

import numpy as np


def gini_coefficient(x):
    # half of relative mean difference
    sorted_x = sorted(x)
    len_x = len(x)
    tot = 0.0
    for i, xi in enumerate(sorted_x):
        added = i
        subtracted = len_x - i - 1
        tot += (added - subtracted) * xi
    return tot / (len_x ** 2)


def max_error(y_true, pred):
    return max(np.abs(y_true - pred))


def error_variance(y_true, pred):
    return np.std(y_true - pred) ** 2


def relative_error_variance(y_true, pred):
    return (np.std(y_true - pred) / np.std(y_true)) ** 2


def gini_loss(y_true, pred):
    return gini_coefficient(y_true - pred)


def categorical_gini_coefficient(x):
    len_x = len(x)
    counter = Counter(x)
    total = 0.0
    for _, count in counter.items():
        total += len_x - count
    return total / (len_x ** 2)


def categorical_gini_loss(y_true, y_pred):
    # this is kind of random
    return categorical_gini_coefficient(y_true != y_pred)
