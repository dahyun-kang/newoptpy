import time
import numpy as np


def tuple2colvec(x: tuple) -> np.array:
    return np.array(x)[np.newaxis, ...].T

def colvec2tuple(x: np.array) -> tuple:
    return tuple(x.T[0])

def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("Wall clock time [{}]: {:.4f} sec".format(original_fn.__name__, end_time-start_time))
        return result
    return wrapper_fn


class TrajectoryMem:
    def __init__(self):
        self.xlist = []
        self.x = None
        self.x_prev = None
        self.G = None
        self.G_prev = None
        self.H = None

    def update(self, x, G, H=None):
        self.xlist.append(x)
        self.x_prev = self.x
        self.x = x
        self.G_prev = self.G
        self.G = G
        if H is not None:
            self.H = H
