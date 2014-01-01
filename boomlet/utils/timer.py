from time import time
from contextlib import contextmanager


@contextmanager
def timer(name=""):
    start_time = time()
    yield
    print("{} took {}s".format(name, time() - start_time))
