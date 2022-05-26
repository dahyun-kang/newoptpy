import numpy as np
from abc import abstractmethod
from util import colvec2tuple, tuple2colvec


class ObjectiveFunction:
    @abstractmethod
    def __call__(self, x: np.array) -> float:
        pass

    @abstractmethod
    def compute_grad(self, x: np.array) -> np.array:
        pass

    @abstractmethod
    def hessian(self, x: np.array) -> np.array:
        pass


class Paraboloid(ObjectiveFunction):
    def __call__(self, x) -> float:
        x1, x2 = x if isinstance(x, tuple) else colvec2tuple(x)
        return (x1 - 2.) ** 2 + (x2 - 2.) ** 2

    def compute_grad(self, x: np.array) -> np.array:
        x1, x2 = colvec2tuple(x)
        grad_tuple = (2. * (x1 - 2.), 2. * (x2 - 2.))
        return tuple2colvec(grad_tuple)

    def hessian(self, x) -> np.array:
        x1, x2 = colvec2tuple(x)
        return np.array([[2. * x1, 0.],
                         [0., 2. * x2]])


class SkewedParaboloid(ObjectiveFunction):
    def __call__(self, x) -> float:
        x1, x2 = x if isinstance(x, tuple) else colvec2tuple(x)
        return 10 * (x1 - 2.) ** 2 + (x2 - 2.) ** 2

    def compute_grad(self, x: np.array) -> np.array:
        x1, x2 = colvec2tuple(x)
        grad_tuple = (20. * (x1 - 2.), 2. * (x2 - 2.))
        return tuple2colvec(grad_tuple)

    def hessian(self, x) -> np.array:
        x1, x2 = colvec2tuple(x)
        return np.array([[20. * x1, 0.],
                         [0., 2. * x2]])
