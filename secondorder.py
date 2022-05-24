import argparse
import numpy as np
from abc import abstractmethod


def tuple2colvec(x: tuple) -> np.array:
    return np.array(x)[np.newaxis, ...].T

def colvec2tuple(x: np.array) -> tuple:
    return tuple(x.T[0])


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
    def __call__(self, x: np.array) -> float:
        x1, x2 = colvec2tuple(x)
        return (x1 - 2.) ** 2 + (x2 - 2.) ** 2

    def compute_grad(self, x: np.array) -> np.array:
        x1, x2 = colvec2tuple(x)
        grad_tuple = (2. * (x1 - 2.), 2. * (x2 - 2.))
        return tuple2colvec(grad_tuple)

    def hessian(self, x) -> np.array:
        x1, x2 = colvec2tuple(x)
        return np.array([[2 * x1, 0.],
                         [0., 2 * x2]])


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


class SecondOrderOptimizer:
    def __init__(self, args):
        self.maxiter = args.maxiter
        self.stepsize = args.stepsize

        # store x, G (first-order gradient), H (inverse of second-order gradient)
        self.mem = TrajectoryMem()

    @abstractmethod
    def estimate_inv_hessian(self, x: np.array):
        """
        Take x as an input and estimates the inverse of Hessian

        :param x: input x
        :return: inverse of Hessian
        """
        pass

    def set_obj(self, f):
        self.f = f
        return self

    def is_not_improved(self, x_prev, x, eps=1e-6):
        return np.abs(x_prev - x).sum() < eps

    def fit(self, x_0: tuple):
        x = tuple2colvec(x_0)

        for i in range(self.maxiter):
            print(f'iter: {i:02d} | x_opt: {colvec2tuple(x)} | value: {self.f(x)}')
            G = self.f.compute_grad(x)
            H = self.estimate_inv_hessian(x)
            x_prev = x

            x = x - self.stepsize * H @ G

            if self.is_not_improved(x_prev, x):
                break

            self.mem.update(x, G, H)
        print(f'\nGlobal optimum found at:\niter: {i:02d} | x_opt: {colvec2tuple(x)} | val_opt: {self.f(x)}')


class VanillaNewtonsMethod(SecondOrderOptimizer):
    def estimate_inv_hessian(self, x):
        H = self.f.hessian(x)
        return np.linalg.inv(H)


class Davidon(SecondOrderOptimizer):
    def estimate_inv_hessian(self, x):
        eye = np.identity(x.shape[0])
        # Initialization of B is an arbitrary positive-definite matrix
        if self.mem.x_prev is None:
            return eye

        s = self.mem.x - self.mem.x_prev
        y = self.mem.G - self.mem.G_prev
        H = self.mem.H

        H_next = H + ((s - H @ y) @ (s - H @ y).T) / (((s - H @ y).T @ y) + 1e-6)
        return H_next


class DavidonFletcherPowell(SecondOrderOptimizer):
    def estimate_inv_hessian(self, x):
        # Initialization of B is an arbitrary positive-definite matrix
        eye = np.identity(x.shape[0])
        if self.mem.x_prev is None:
            return eye

        s = self.mem.x - self.mem.x_prev
        y = self.mem.G - self.mem.G_prev
        H = self.mem.H

        H_next = H - ((H @ y @ y.T @ H) / (y.T @ H @ y)) + ((s @ s.T) / (y.T @ s))
        return H_next


class BroydenFletcherGoldfarbShanno(SecondOrderOptimizer):
    def estimate_inv_hessian(self, x):
        # Initialization of B is an arbitrary positive-definite matrix
        eye = np.identity(x.shape[0])
        if self.mem.x_prev is None:
            return eye

        s = self.mem.x - self.mem.x_prev
        y = self.mem.G - self.mem.G_prev
        H = self.mem.H

        H_next = (eye - (s @ y.T) / (y.T @ s)) @ H @ (eye - (y @ s.T) / (y.T @ s)) + ((s @ s.T) / (y.T @ s))
        return H_next


if __name__ == '__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Second-order gradient method from scratch')
    parser.add_argument('--maxiter', type=int, default=100, help='Max iteration until it halts')
    parser.add_argument('--stepsize', type=float, default=1.0, help='Step size eta')
    args = parser.parse_args()

    # practice 1
    x_0 = (100., 100.)
    # VanillaNewtonsMethod(args).set_obj(Paraboloid()).fit(x_0)
    # Davidon(args).set_obj(Paraboloid()).fit(x_0)  # goes towards infinity x0x
    # DavidonFletcherPowell(args).set_obj(Paraboloid()).fit(x_0)
    BroydenFletcherGoldfarbShanno(args).set_obj(Paraboloid()).fit(x_0)
