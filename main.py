import argparse
from optim import FirstOrderGradOptimizer, VanillaNewtonsMethod, Davidon, DavidonFletcherPowell, BroydenFletcherGoldfarbShanno
from objfunction import Paraboloid


if __name__ == '__main__':
    # Arguments parsing
    optim_choices = ['firstorder', 'newton', 'davidon', 'dfp', 'bfgs']
    objfunc_choices = ['paraboloid', 'skeepsidedvally']

    parser = argparse.ArgumentParser(description='Second-order gradient method from scratch')
    parser.add_argument('--optim', type=str, default='newton', choices=optim_choices, help='Optimizer')
    parser.add_argument('--maxiter', type=int, default=10000, help='Max iteration until it halts')
    parser.add_argument('--stepsize', type=float, default=0.1, help='Step size eta')
    parser.add_argument('--vis', action='store_true', help='Flag to visualize')
    args = parser.parse_args()

    # Optim
    optimdict = dict(firstorder=FirstOrderGradOptimizer,
                     newton=VanillaNewtonsMethod,
                     davidon=Davidon,
                     dfp=DavidonFletcherPowell,
                     bfgs=BroydenFletcherGoldfarbShanno)
    optim = optimdict[args.optim]


    x_0 = (100., 100.)
    # ex) BroydenFletcherGoldfarbShanno(args).set_obj(Paraboloid()).fit(x_0)
    optim(args).set_obj(Paraboloid()).fit(x_0)
