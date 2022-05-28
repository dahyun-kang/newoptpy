import numpy as np
from abc import abstractmethod
from util import colvec2tuple, tuple2colvec


class ObjectiveFunction:
    @abstractmethod
    def __call__(self, x: np.array) -> float:
        pass

    @abstractmethod
    def first_order_grad(self, x: np.array) -> np.array:
        pass

    @abstractmethod
    def second_order_grad(self, x: np.array) -> np.array:
        pass


class Paraboloid(ObjectiveFunction):
    def __call__(self, x) -> float:
        x1, x2 = x if isinstance(x, tuple) else colvec2tuple(x)
        return (x1 - 2.) ** 2 + (x2 - 2.) ** 2

    def first_order_grad(self, x: np.array) -> np.array:
        x1, x2 = colvec2tuple(x)
        return np.array([[2. * (x1 - 2.)],
                         [2. * (x2 - 2.)]])

    def second_order_grad(self, x) -> np.array:
        x1, x2 = colvec2tuple(x)
        return np.array([[2. * x1, 0.],
                         [0., 2. * x2]])


class SkewedParaboloid(ObjectiveFunction):
    def __call__(self, x) -> float:
        x1, x2 = x if isinstance(x, tuple) else colvec2tuple(x)
        return 10 * (x1 - 2.) ** 2 + (x2 - 2.) ** 2

    def first_order_grad(self, x: np.array) -> np.array:
        x1, x2 = colvec2tuple(x)
        grad_tuple = (20. * (x1 - 2.), 2. * (x2 - 2.))
        return tuple2colvec(grad_tuple)

    def second_order_grad(self, x) -> np.array:
        x1, x2 = colvec2tuple(x)
        return np.array([[20. * x1, 0.],
                         [0., 2. * x2]])


class SteepSidedValley(ObjectiveFunction):
    def __call__(self, x) -> float:
        x1, x2 = x if isinstance(x, tuple) else colvec2tuple(x)
        return 100 * ((x2 - (x1 ** 2)) ** 2) + ((1 - x1) ** 2)

    def first_order_grad(self, x: np.array) -> np.array:
        x1, x2 = colvec2tuple(x)
        '''
        grad_x1 = 200 * (-2 * x1) * (x2 - (x1 ** 2)) - 2 * (1 - x1)
        grad_x2 = 200 * (x2 - (x1 ** 2))
        '''
        grad_x1 = -400 * x1 * (x2 - (x1 ** 2)) - 2 * (1 - x1)
        grad_x2 = 200 * (x2 - (x1 ** 2))
        return tuple2colvec((grad_x1, grad_x2))

    def second_order_grad(self, x) -> np.array:
        x1, x2 = colvec2tuple(x)
        grad_x1x1 = -400 * (x2 - 3 * (x1 ** 2)) + 2
        grad_x1x2 = -400 * x1
        grad_x2x1 = -400 * x1
        grad_x2x2 = 200
        return np.array([[grad_x1x1, grad_x1x2],
                         [grad_x2x1, grad_x2x2]])
