import numpy as np
from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer

from boomlet.utils.collection import group_by
from boomlet.utils.aggregators import to_aggregator
from boomlet.utils.matrix_decoders import to_decoder


""" Numerical -> Categorical """


class Discretizer(BaseEstimator, TransformerMixin):
    """
    Numerical to Categorical converter. Converts a numerical array
    to discrete evenly spaced percentiles.

    Can also be used to convert
    Numerical to Numerical because the output is still ordered.
    """

    def __init__(self, levels=10):
        self.levels = levels

    def fit(self, X, y=None):
        assert len(X.shape) == 1
        percentiles = np.linspace(0, 100, self.levels + 1)[1:-1]
        self.cutoffs_ = [np.percentile(X, p) for p in percentiles]
        return self

    def transform(self, X):
        assert len(X.shape) == 1
        result = np.zeros(X.shape)
        for cutoff in self.cutoffs_:
            result += X >= cutoff
        return result


class Clusterizer(BaseEstimator, TransformerMixin):
    """
    Numerical to Categorical converter. Converts a numerical array
    to discrete values with a given clustering algorithm.
    """

    def __init__(self, clusterer):
        self.clusterer = deepcopy(clusterer)

    def fit(self, X, y=None):
        assert len(X.shape) == 1
        self.clusterer.fit(X.reshape(-1, 1), y)
        return self

    def transform(self, X):
        """
        This may fail because some of scikit-learn's clustering
        estimators do not implement a predict method (only
        fit_predict).
        """
        assert len(X.shape) == 1
        return self.clusterer.predict(X.reshape(-1, 1))

    def fit_transform(self, X, y=None):
        assert len(X.shape) == 1
        return self.clusterer.fit_predict(X.reshape(-1, 1), y)


class TransformDecoder(BaseEstimator, TransformerMixin):
    """
    Numerical to Categorical converter. Converts a numerical array
    to disrete values by applying a trasform, then a decoding
    algorithm to the results of that transform.
    """

    def __init__(self, transformer, decoder="max"):
        self.transformer = deepcopy(transformer)
        self.decoder = decoder

    def fit(self, X, y=None):
        assert len(X.shape) == 1
        self.transformer.fit(X.reshape(-1, 1), y)
        return self

    def transform(self, X):
        assert len(X.shape) == 1
        transformed = self.transformer.transform(X.reshape(-1, 1))
        decoder = to_decoder(self.decoder)
        return decoder(transformed)

    def fit_transform(self, X, y=None):
        assert len(X.shape) == 1
        transformed = self.transformer.fit_transform(X.reshape(-1, 1), y)
        decoder = to_decoder(self.decoder)
        return decoder(transformed)


""" Categorical -> Numerical """


class BinaryTransformer(BaseEstimator, TransformerMixin):
    """
    Categorical to Numerical converter. Takes in a categorical array
    and performs a one hot encoding and optionally applies a
    transformer (generally a dimensionality reduction algorithm) on
    the result.
    """


    def __init__(self, transformer=None):
        self.transformer = deepcopy(transformer)

    def fit(self, X, y=None):
        assert len(X.shape) == 1
        self.binarizer_ = LabelBinarizer()
        newX = self.binarizer_.fit_transform(X)
        if self.transformer is not None:
            self.transformer.fit(newX)
        return self

    def transform(self, X):
        assert len(X.shape) == 1
        newX = self.binarizer_.transform(X)
        if self.transformer is not None:
            return self.transformer.transform(newX)
        else:
            return newX

    def fit_transform(self, X, y=None):
        assert len(X.shape) == 1
        self.binarizer_ = LabelBinarizer()
        newX = self.binarizer_.fit_transform(X)
        if self.transformer is not None:
            return self.transformer.fit_transform(newX)
        else:
            return newX


class DiscreteConstantPredictor(BaseEstimator, TransformerMixin):
    """
    Categorical to Numerical converter. Predicts a constant for each
    discrete value in the input. Returns 0 for values not in the
    training set.
    """

    def __init__(self, aggregator="mean"):
        self.aggregator = aggregator

    def fit(self, X, y):
        assert len(X.shape) == 1
        enumerated_y = list(enumerate(y))
        grouped = group_by(enumerated_y, lambda i: X[i[0]])
        aggregator = to_aggregator(self.aggregator)
        self.value_map_ = {}
        for k, v in grouped.items():
            self.value_map_[k] = aggregator(np.array([i[1] for i in v]))
        return self

    def transform(self, X):
        assert len(X.shape) == 1
        return np.array([self.value_map_.get(x, 0) for x in X])


class DiscreteOrdinalPredictor(BaseEstimator, TransformerMixin):
    """
    Predicts a rank for each discrete value in the input. Can
    be seen as a regularized version of DiscreteConstantPredictor.
    """

    def __init__(self, aggregator="mean"):
        self.aggregator = aggregator

    def fit(self, X, y):
        assert len(X.shape) == 1
        self.dcp_ = DiscreteConstantPredictor(self.aggregator)
        self.dcp_.fit(X, y)
        items = self.dcp_.value_map_.items()
        sorted_items = sorted(items, key=lambda x: x[1])
        self.value_map_ = {x[0]: idx for idx, x in enumerate(sorted_items)}
        return self

    def transform(self, X):
        assert len(X.shape) == 1
        return np.array([self.value_map_.get(x, 0) for x in X])
