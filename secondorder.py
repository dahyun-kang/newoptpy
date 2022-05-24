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
        return x1 ** 2 + x2 ** 2

    def compute_grad(self, x: np.array) -> np.array:
        x1, x2 = colvec2tuple(x)
        grad_tuple = (2. * x1, 2. * x2)
        return tuple2colvec(grad_tuple)

    def hessian(self, x) -> np.array:
        x1, x2 = colvec2tuple(x)
        return np.array([[2 * x1, 0.],
                         [0., 2 * x2]])


class SecondOrderOptimizer:
    def __init__(self, args):
        self.maxiter = args.maxiter
        self.stepsize = args.stepsize

        # store x, G (first-order gradient), H (inverse of second-order gradient)
        self.trajectory = {'x': [], 'G': [], 'H': []}

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

    def is_improved(self, x_prev, x, eps=1e-6):
        return np.abs(x_prev - x).sum() < eps

    def update_trajectory(self, x, G, H):
        self.trajectory['x'].append(x)
        self.trajectory['G'].append(G)
        self.trajectory['H'].append(H)

    def fit(self, x_0: tuple):
        x = tuple2colvec(x_0)

        for i in range(self.maxiter):
            print(f'iter: {i:02d} | x_opt: {colvec2tuple(x)} | value: {self.f(x)}')
            G = self.f.compute_grad(x)
            H = self.estimate_inv_hessian(x)
            x_prev = x

            x = x - self.stepsize * H @ G

            if self.is_improved(x_prev, x):
                break

            self.update_trajectory(x, G, H)
        print(f'\nGlobal optimum found at:\niter: {i:02d} | x_opt: {colvec2tuple(x)} | val_opt: {self.f(x)}')


class VanillaNewtonsMethod(SecondOrderOptimizer):
    def estimate_inv_hessian(self, x):
        H = self.f.hessian(x)
        return np.linalg.inv(H)


class SymmetricRank1Update(SecondOrderOptimizer):
    def estimate_inv_hessian(self, x):
        # Initialization of B is an arbitrary positive-definite matrix
        if len(self.trajectory['x']) <= 1:
            return np.identity(x.shape[0])

        x_prev, x = self.trajectory['x'][-2], self.trajectory['x'][-1]
        G_prev, G = self.trajectory['G'][-2], self.trajectory['G'][-1]
        H = self.trajectory['H'][-1]

        s = x - x_prev
        y = G - G_prev
        H_next = H + ((s - H @ y) @ (s - H @ y).T) / (((s - H @ y).T @ y) + 1e-6)
        return H_next


class SymmetricRank2Update(SecondOrderOptimizer):
    def estimate_inv_hessian(self, x):
        # Initialization of B is an arbitrary positive-definite matrix
        eye = np.identity(x.shape[0])
        if len(self.trajectory['x']) <= 1:
            return eye

        x_prev, x = self.trajectory['x'][-2], self.trajectory['x'][-1]
        G_prev, G = self.trajectory['G'][-2], self.trajectory['G'][-1]
        H = self.trajectory['H'][-2]

        s = x - x_prev
        y = G - G_prev
        H_next = H - ((H @ y @ y.T @ H) / (y.T @ H @ y)) + ((s @ s.T) / (y.T @ s))
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
    # SymmetricRank1Update(args).set_obj(Paraboloid()).fit(x_0)  # goes towards infinity x0x
    SymmetricRank2Update(args).set_obj(Paraboloid()).fit(x_0)
