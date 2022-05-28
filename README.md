<div align="center">
  <h1> NewOptpy </h1>
  <h3> <ins>New</ins>ton’s second-order <ins>Opt</ins>imization methods in <ins>Py</ins>thon </h3>
</div>

<br/>

<div align="center">
  <img src="assets/newoptpylogo.png" alt="logo" width="600"/>
</div>

Python implementation of Newton's and Quasi-Newton's second-order optimization methods


<div align="center">
  <img src="assets/FirstOrderGradOptimizer_SkewedParaboloid_stepsize0.02_init[100.0,10.0].gif" alt="firstorder" width="300"/>
  <img src="assets/VanillaNewtonsMethod_SkewedParaboloid_stepsize1.0_init[100.0,10.0].gif" alt="newton" width="300"/>
  <img src="assets/DavidonFletcherPowell_SkewedParaboloid_stepsize0.001_init[100.0,10.0].gif" alt="bfp" width="300"/>
</div>

## :school: Disclaimer
This project was carried out as a part of a coursework term project [CSED490Y]: Optimization for Machine Learning @ POSTECH.


## :pencil: Main features
* Implemented from scratch with minimal dependency (e.g. numpy, matplotlib)
* Four Newton’s method supported
* Two-variate convex functions supported
* Nice visualization


## :gear: Requirements
* [python 3.x](https://pytorch.org)
* [numpy](https://numpy.org)
* [matplotlib](https://matplotlib.org)



## :pushpin: Quick start
### Important arguments
* `optim`: choice of optimizer among `{firstorder, newton, davidon, dfp, bfgs}`.
* `func`: choice of two-variate function. The example functions are `{paraboloid, skewedparaboloid, steepsidedvalley}`.
* `maxiter: int`: number of max iteration.
* `stepsize: float`: step size (or learning rate)
* `init`: two-dimensional coordinate.
* `viz`: flag to visualize the landscape.

### Example run:
```bash
python main.py --optim dfp \
               --func skewedparaboloid \
               --stepsize 0.002 \
               --init 100 10 \
               --viz
```
