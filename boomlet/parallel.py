import multiprocessing

from pickle import PicklingError
from functools import partial
from joblib import Parallel, delayed

from boomlet.settings import PARALLEL


def joblib_parmap(func, generator):
    """ parallel map using joblib, but it pickles input arguments and thus can't be used for dynamically generated functions.
    """
    try:
        new_func = delayed(func)
    except TypeError as e:
        raise PicklingError(e)
    return joblib_run(new_func(item) for item in generator)


def joblib_run(delayed_generator):
    """ runs a generator of joblib tasks
    NOTE: the functions run do not have to be homogeneous, you can make arbitrary generators with whatever functions as long as they are pickle-able
    """
    return Parallel(n_jobs=PARALLEL.JOBS, verbose=PARALLEL.JOBLIB_VERBOSE, pre_dispatch=PARALLEL.JOBLIB_PRE_DISPATCH)(delayed_generator)


def no_pickle_parmap(func, generator):
    """ alternative parallel map that allows for unpicklable items by using pipes

    source: http://stackoverflow.com/questions/3288595/multiprocessing-using-pool-map-on-a-function-defined-in-a-class
    """
    def spawn(func):
        def fun(q_in, q_out):
            while True:
                i, x = q_in.get()
                if i is None:
                    break
                q_out.put((i, func(x)))
        return fun

    n_jobs = multiprocessing.cpu_count() if PARALLEL.JOBS == -1 else PARALLEL.JOBS

    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=spawn(func), args=(q_in, q_out)) for _ in range(n_jobs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(generator)]
    [q_in.put((None, None)) for _ in range(n_jobs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]


def pmap(func, generator, *args, **kwargs):
    """ parallel map that only parallelizes if not already within a pmap
    """
    new_func = partial(func, *args, **kwargs) if args or kwargs else func
    if not PARALLEL.PMAP:
        return map(new_func, generator)
    else:
        # don't allow parallel maps under this to parallelize
        # because computation is already in parallel
        PARALLEL.PMAP = False
        try:
            return joblib_parmap(new_func, generator)
        except PicklingError as e:
            # allow parallel maps under this to try to perform
            # in parallel
            print("PicklingError: {}".format(e))
            return no_pickle_parmap(new_func, generator)
        finally:
            PARALLEL.PMAP = True
