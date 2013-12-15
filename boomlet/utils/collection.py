import itertools
from collections import defaultdict


def grouper(n, iterable, fillvalue=None):
    """ groups an iterable into chunks of a certain size
    >>> grouper(3, 'ABCDEFG', 'x')
    ABC DEF Gxx

    from: http://stackoverflow.com/questions/1624883/alternative-way-to-split-a-list-into-groups-of-n
    """
    args = [iter(iterable)] * n
    return itertools.izip_longest(*args, fillvalue=fillvalue)


def group_by(coll, key_fn):
    """returns a dict with keys as the unique values of a key function applied to all items in a collection and values as a list of items that correspond to that key"""
    out = defaultdict(list)
    for v in coll:
        out[key_fn(v)].append(v)
    return dict(out)
