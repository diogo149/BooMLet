import numpy as np


def max_decoder(X):
    assert len(X.shape) == 2
    return X.argmax(axis=1)


def min_decoder(X):
    assert len(X.shape) == 2
    return X.argmin(axis=1)


def binary_decoder(X):
    assert len(X.shape) == 2
    assert X.min() >= 0
    assert X.max() <= 1
    n = X.shape[1]
    binary = X.round().astype(np.int)
    return binary.dot([2 ** x for x in xrange(n)])


DECODERS = dict(
    max=max_decoder,
    min=min_decoder,
    binary=binary_decoder,
)


def from_str(s):
    return DECODERS[s]


def to_decoder(decoder):
    if isinstance(decoder, str):
        return from_str(decoder)
    else:
        return decoder
