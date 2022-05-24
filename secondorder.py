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

        # store x, grad_f, B_f (the estimated inverse of Hessian)
        self.trajectory = {'x': [], 'grad': [], 'B': []}

    def get_B(self, x: np.array):
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

    def update_trajectory(self, x, grad, B):
        self.trajectory['x'].append(x)
        self.trajectory['grad'].append(grad)
        self.trajectory['B'].append(B)

    def fit(self, x_0: tuple):
        x = tuple2colvec(x_0)

        for i in range(self.maxiter):
            print(f'iter: {i:02d} | x_opt: {colvec2tuple(x)} | value: {self.f(x)}')
            grad = self.f.grad(x)
            B = self.get_B(x)
            x_prev = x

            x = x - self.stepsize * B @ grad

            if self.is_improved(x_prev, x):
                break

            self.update_trajectory(x, grad, B)
        print(f'\nGlobal optimum found at:\niter: {i:02d} | x_opt: {colvec2tuple(x)} | val_opt: {self.f(x)}')


class VanillaNewtonsMethod(SecondOrderOptimizer):
    def __init__(self, args):
        super(VanillaNewtonsMethod, self).__init__(args)

    def get_B(self, x):
        H = self.f.hessian(x)
        return np.linalg.inv(H)


class SymmetricRank1Update(SecondOrderOptimizer):
    def __init__(self, args):
        super(SymmetricRank1Update, self).__init__(args)

    def get_B(self, x):
        # Initialization of B is an arbitrary positive-definite matrix
        if len(self.trajectory['x']) <= 1:
            return np.identity(x.shape[0])

        x_prev, x = self.trajectory['x'][-2], self.trajectory['x'][-1]
        grad_prev, grad = self.trajectory['grad'][-2], self.trajectory['grad'][-1]
        B = self.trajectory['B'][-1]

        s = x - x_prev
        y = grad - grad_prev
        B_next = B + ((s - B @ y) @ (s - B @ y).T) / (((s - B @ y).T @ y) + 1e-6)
        return B_next


if __name__ == '__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Second-order gradient method from scratch')
    parser.add_argument('--maxiter', type=int, default=100, help='Max iteration until it halts')
    parser.add_argument('--stepsize', type=float, default=1.0, help='Step size eta')
    args = parser.parse_args()

    # practice 1
    x_0 = (100., 100.)
    # VanillaNewtonsMethod(args).set_obj(Paraboloid()).fit(x_0)
    SymmetricRank1Update(args).set_obj(Paraboloid()).fit(x_0)
