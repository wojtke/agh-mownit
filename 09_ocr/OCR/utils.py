import numpy as np
import imageio
import matplotlib.pyplot as plt
import functools
import time

plt.rcParams['figure.figsize'] = (6,6)


def display(*args, **kwargs):
    plt.figure(**kwargs)
    for i in range(len(args)):
        plt.subplot(1, len(args), i+1)
        imgplot = plt.imshow(args[i], cmap='Greys_r')
    plt.show()

def debug(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({signature})")

        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()

        print(f"{func.__name__!r} returned {value!r}, {end_time - start_time:.4f} secs")
        return value
    return wrapper

    '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQ...