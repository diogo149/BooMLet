import math
from copy import deepcopy
from sklearn.cross_validation import ShuffleSplit

def flexible_int(size, in_val=None):
    """ allows for flexible input as a size
    """
    if in_val is None:
        return size
    elif isinstance(in_val, (float, int)):
        if isinstance(in_val, float):
            assert abs(in_val) <= 1.0, in_val
            in_val = int(round(in_val * size))
        if in_val < 0:
            in_val += size  # setting negative values as amount not taken
        return max(0, min(size, in_val))
    elif isinstance(in_val, str):
        if in_val == "sqrt":
            return int(round(math.sqrt(size)))
        elif in_val == "log2":
            return int(round(math.log(size) / math.log(2)))
        elif in_val == "auto":
            return size
    raise Exception("Improper flexible_int input: {}".format(in_val))


def fit_predict(clf, X, y, X_test):
    tmp_clf = deepcopy(clf)
    tmp_clf.fit(X, y)
    return tmp_clf.predict(X_test)


def quick_cv(clf,
             X,
             y,
             score_func,
             n_iter=3,
             test_size=0.1,
             random_state=None):
    """ returns the cross validation """
    cv = ShuffleSplit(y.shape[0],
                      n_iter=n_iter,
                      test_size=test_size,
                      random_state=random_state,
    )
    scores = []
    for train, test in cv:
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        preds = fit_predict(clf, X_train, y_train, X_test)
        scores.append(score_func(y_test, preds))
    return sum(scores) / float(len(scores))


def quick_score(clf,
                X,
                y,
                score_func,
                X_valid=None,
                y_valid=None,
                n_iter=3,
                test_size=0.1,
                random_state=None):
    """scores on a validation set, if available, else with cross validation"""
    if X_valid is not None:
        return score_func(y_valid, fit_predict(clf, X, y, X_valid)
    else:
        return quick_cv(clf, X, y, score_func, n_iter, test_size, random_state)
