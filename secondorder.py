import argparse
import numpy as np


def tuple2colvec(x: tuple) -> np.array:
    return np.array(x)[np.newaxis, ...].T

def colvec2tuple(x: np.array) -> tuple:
    return tuple(x.T[0])


class ObjectiveFunction:
    def __call__(self, x: np.array) -> float:
        pass

    def grad(self, x: np.array) -> np.array:
        pass

    def hessian(self, x: np.array) -> np.array:
        pass


class Paraboloid(ObjectiveFunction):
    def __call__(self, x: np.array) -> float:
        x1, x2 = colvec2tuple(x)
        return x1 ** 2 + x2 ** 2

    def grad(self, x: np.array) -> np.array:
        x1, x2 = colvec2tuple(x)
        grad_tuple = (2. * x1, 2. * x2)
        return tuple2colvec(grad_tuple)

    def hessian(self, x) -> np.array:
        x1, x2 = colvec2tuple(x)
        H = np.array([[2 * x1, 0.],
                      [0., 2 * x2]])
        return H


class SecondOrderOptimizer:
    def __init__(self, args):
        self.maxiter = args.maxiter
        self.stepsize = args.stepsize

    def set_obj(self, f, optimum):
        self.f = f
        self.f.optimum = tuple2colvec(optimum)
        return self

    def get_B(self, x):
        pass

    def fit(self, x_0: tuple):
        x = tuple2colvec(x_0)

        for i in range(self.maxiter):
            print(f'iter: {i:02d} | x_opt: {colvec2tuple(x)} | value: {self.f(x)}')
            x = x - self.stepsize * self.get_B(x) @ self.f.grad(x)
            if np.abs(self.f.optimum - x).sum() < 1e-6:
                break
        print(f'\nGlobal optimum found at:\niter: {i:02d} | x_opt: {colvec2tuple(x)} | val_opt: {self.f(x)}')


class VanillaNewtonsMethod(SecondOrderOptimizer):
    def __init__(self, args):
        super(VanillaNewtonsMethod, self).__init__(args)

    def get_B(self, x):
        H = self.f.hessian(x)
        return np.linalg.inv(H)


if __name__ == '__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Second-order gradient method from scratch')
    parser.add_argument('--maxiter', type=int, default=100, help='Max iteration until it halts')
    parser.add_argument('--stepsize', type=float, default=1.0, help='Step size eta')
    args = parser.parse_args()

    # practice 1
    x_0 = (100., 100.)
    VanillaNewtonsMethod(args).set_obj(Paraboloid(), optimum=(0., 0.)).fit(x_0)
