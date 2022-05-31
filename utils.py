import time

def time_section(func):
    """Decorator appending the time it takes to execute a function."""
    def timing_wrapper(*args, **kwargs):
        start = time.time()
        retval = func(*args, **kwargs)
        end = time.time()

        if type(retval) is not tuple:
            retval = (retval,)
        return *retval, end-start
    return timing_wrapper
