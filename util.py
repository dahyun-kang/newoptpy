import numpy as np


def tuple2colvec(x: tuple) -> np.array:
    return np.array(x)[np.newaxis, ...].T

def colvec2tuple(x: np.array) -> tuple:
    return tuple(x.T[0])


class TrajectoryMem:
    def __init__(self):
        self.xlist = []
        self.x = None
        self.x_prev = None
        self.G = None
        self.G_prev = None
        self.H = None

    def update(self, x, G, H):
        self.xlist.append(x)
        self.x_prev = self.x
        self.x = x
        self.G_prev = self.G
        self.G = G
        self.H = H
