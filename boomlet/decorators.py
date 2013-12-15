import warnings
import logging
import functools
from time import time
from pdb import set_trace


def default_on_fail(default_value):
    """
    If the decorated function fails we instead use a decorated value.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except:
                return default_value
        return wrapped
    return decorator


def log(func):
    """
    Logs input, output, and time takes of a decorated function.
    """
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        logging.debug('Calling %s', func)
        logging.debug('INPUT (args)  : %s', args)
        logging.debug('INPUT (kwargs): %s', kwargs)
        start_time = time()
        output = func(*args, **kwargs)
        logging.debug('Returning %s', func)
        logging.debug('Took: %lf secs', time() - start_time)
        logging.debug('OUTPUT (kwargs): %s', output)
        return output
    return wrapped


def timer(func):
    """
    Times the decorated function.
    """
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        start_time = time()
        output = func(*args, **kwargs)
        print("Function {} took {} seconds.".format(func, time() - start_time))
        return output
    return wrapped


def trace_error(func):
    """ python debugger is started if functions throws an exception
    """
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print("{} in {}: {}".format(e.__class__, func, e.message))
            set_trace()
            return func(*args, **kwargs)
    return wrapped


def ignore_args(func):
    """ decorated functions ignores input arguments
    """
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        return func()
    return wrapped


def deprecated(func):
    """ warns if decorated function is used
    """
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        warnings.warn("Deprecated: {}".format(func))
        return func(*args, **kwargs)
    return wrapped


def untested(func):
    """ warns if decorated function is used
    """
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        warnings.warn("Untested: {}".format(func))
        return func(*args, **kwargs)
    return wrapped


def todo(msg):
    """ alerts about a todo if decorated function is used
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            warnings.warn("TODO: {}; {}".format(func, msg))
            return func(*args, **kwargs)
        return wrapped
    return decorator
