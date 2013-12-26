try:
    import cPickle as pickle
except ImportError:
    import pickle
import joblib
import zlib
import os
import glob

import dill


def mkdir(filename):
    """ try to make directory
    """
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError:
        pass


def exists(filename):
    return os.path.exists(filename)


def writes(filename, s):
    with open(filename, 'w') as outfile:
        outfile.write(s)


def reads(filename):
    with open(filename) as infile:
        return infile.read()


def compress(s, level=9):
    return zlib.compress(s, level)


def decompress(s):
    return zlib.decompress(s)


def pickle_load(filename):
    with open(filename) as infile:
        return pickle.load(infile)


def pickle_dump(filename, obj):
    with open(filename, 'w') as outfile:
        pickle.dump(obj, outfile, pickle.HIGHEST_PROTOCOL)


def pickle_loads(s):
    return pickle.loads(s)


def pickle_dumps(obj):
    return pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)


def joblib_load(filename, mmap_mode='r'):
    return joblib.load(filename, mmap_mode)


def joblib_dump(filename, obj, compress=0):
    return joblib.dump(obj, filename, compress)


def dill_load(filename):
    with open(filename) as infile:
        return dill.load(infile)


def dill_dump(filename, obj):
    with open(filename, 'w') as outfile:
        dill.dump(obj, outfile)


def dill_loads(s):
    return dill.loads(s)


def dill_dumps(obj):
    return dill.dumps(obj)


def glob_one(*args):
    if len(args) == 1:
        dirname = "."
        hint, = args
    elif len(args) == 2:
        dirname, hint = args
    else:
        raise Exception("improper argument count: {}".format(args))

    globbed = glob.glob1(dirname, "*" + hint + "*")
    assert len(globbed) == 1, (dirname, hint, globbed)
    return os.path.join(dirname, globbed[0])
