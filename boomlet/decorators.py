import warnings
import logging
import functools
from time import time
from pdb import set_trace
from contextlib import contextmanager


def to_decorator(wrapped_func):
    """
    Encapsulates the decorator logic for most common use cases.

    Expects a wrapped function with compatible type signature to:
        wrapped_func(func, args, kwargs, *outer_args, **outer_kwargs)

    Example:

    @to_decorator
    def foo(func, args, kwargs):
        print(func)
        return func(*args, **kwargs)

    @foo()
    def bar():
        print(42)
    """
    @functools.wraps(wrapped_func)
    def arg_wrapper(*outer_args, **outer_kwargs):
        def decorator(func):
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                return wrapped_func(func,
                                    args,
                                    kwargs,
                                    *outer_args,
                                    **outer_kwargs)
            return wrapped
        return decorator
    return arg_wrapper


def g_decorator(generator_expr):
    """
    Converts generator expression into a decorator

    Takes in a generator expression, such as one accepted by
    contextlib.contextmanager, converts it to a context manager,
    and returns a decorator equivalent to being within that
    context manager.

    TODO do something with yielded value

    Example:

    @g_decorator
    def foo():
        print("Hello")
        yield
        print("World")

    @foo()
    def bar():
        print("Something")
    """
    cm = contextmanager(generator_expr)

    @to_decorator
    def wrapped_func(func, args, kwargs, *outer_args, **outer_kwargs):
        with cm(*outer_args, **outer_kwargs) as yielded:
            return func(*args, **kwargs)

    return wrapped_func


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
