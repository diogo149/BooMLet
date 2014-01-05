from time import time
from contextlib import contextmanager
import random

import numpy as np

@contextmanager
def timer(name=""):
    start_time = time()
    yield
    print("{} took {}s".format(name, time() - start_time))


@contextmanager
def seed_random(seed=None):
    """
    seeds RNG for both random and numpy.random
    """
    if seed is None:
        yield
    elif isinstance(seed, int):
        # save state
        random_state = random.getstate()
        np_random_state = np.random.get_state()
        # set state
        random.seed(seed)
        np.random.seed(seed)
        yield
        # reset state
        random.setstate(random_state)
        np.random.set_state(np_random_state)
    else:
        raise TypeError("Improper random seed type")
