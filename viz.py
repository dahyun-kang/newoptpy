import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class Visualizer:
    def __init__(self, f, xlist):
        self.f = f
        self.xlist = xlist

    def run(self, methodname: str, fname: str, args: dict):

        # path
        X1_path = np.hstack(self.xlist)[0]
        X2_path = np.hstack(self.xlist)[1]
        Y_path = [self.f((x1, x2)) for x1, x2 in zip(X1_path, X2_path)]
        totalstep = len(Y_path) - 1

        X1absmax = max(abs(X1_path.min()), abs(X1_path.max()))
        X2absmax = max(abs(X2_path.min()), abs(X2_path.max()))

        X1 = np.linspace(-X1absmax, X1absmax, 100)
        X2 = np.linspace(-X2absmax, X2absmax, 100)

        # convex obj function landscape
        X1_2d, X2_2d = np.meshgrid(X1, X2)
        Y_2d = np.array([self.f((x1, x2)) for x2 in X2 for x1 in X1])

        fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection":"3d"})
        fig.set_size_inches(5, 5, True)
        fontlabel = {"fontsize":"large", "color":"black", "fontweight":"bold"}
        ax.set_xlabel("x1", fontdict=fontlabel, labelpad=16)
        ax.set_ylabel('x2', fontdict=fontlabel, labelpad=16)
        ax.set_title(f"{fname}\n\n{methodname} took {totalstep} steps", fontdict=fontlabel)
        ax.view_init(elev=30., azim=120)

        # scatter landscape
        # cmap = 'inferno'
        ax.scatter(X1_2d, X2_2d, Y_2d, c=Y_2d, cmap="cool", s=5, alpha=0.1)
        # scatter path
        ax.scatter(X1_path, X2_path, Y_path, c='red', s=8, alpha=1.)

        def animate(i):
            ax.view_init(elev=10., azim=i)
            return fig,

        anim = animation.FuncAnimation(fig, animate, frames=275, interval=20, blit=True)
        filename = f"results/{methodname}_{fname.replace(' ', '_')}_stepsize{args.stepsize}_init{args.init}.gif"
        print(f'Visualizing the trajectory at {filename} ...')
        anim.save(filename, fps=30)
