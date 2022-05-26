import argparse
from optim import FirstOrderGradOptimizer, VanillaNewtonsMethod, Davidon, DavidonFletcherPowell, BroydenFletcherGoldfarbShanno
from objfunction import Paraboloid, SkewedParaboloid


if __name__ == '__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Second-order gradient method from scratch')
    parser.add_argument('--optim', type=str, default='newton', help='Optimizer')
    parser.add_argument('--func', type=str, default='paraboloid', help='Objective function')
    parser.add_argument('--maxiter', type=int, default=10000, help='Max iteration until it halts')
    parser.add_argument('--stepsize', type=float, default=0.1, help='Step size eta')
    parser.add_argument('--viz', action='store_true', help='Flag to visualize')
    args = parser.parse_args()

    # Optimizer
    optimdict = dict(firstorder=FirstOrderGradOptimizer,
                     newton=VanillaNewtonsMethod,
                     davidon=Davidon,
                     dfp=DavidonFletcherPowell,
                     bfgs=BroydenFletcherGoldfarbShanno,
                     )
    optim = optimdict[args.optim]

    # Objective function
    funcdict = dict(paraboloid=Paraboloid,
                    skewedparaboloid=SkewedParaboloid,
                    )
    func = funcdict[args.func]

    x_0 = (100., 100.)
    # ex) BroydenFletcherGoldfarbShanno(args).set_obj(Paraboloid()).fit(x_0)
    optim(args).set_obj(func()).fit(x_0)
