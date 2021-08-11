from functools import wraps
from time import time

def tik_tok(func):
    """
    keep track of time for each process.
    Args:
        func:
    Returns:
    """
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = time()
        try:
            return func(*args, **kwargs)
        finally:
            end_ = time()
            print("time: {:.03f}s".format(end_ - start))

    return _time_it