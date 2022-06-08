<div align="center">
  <h1> NewOptPy </h1>
  <h3> <ins>New</ins>ton’s second-order <ins>Opt</ins>imization methods in <ins>Py</ins>thon </h3>
</div>

<br/>

<div align="center">
  <img src="assets/newoptpylogo.png" alt="logo" width="600"/>
  <h4> Python implementation of Newton's and Quasi-Newton's second-order optimization methods </h4>
</div>


<div align="center">
  <img src="assets/FirstOrderGradOptimizer_SkewedParaboloid_stepsize0.02_init[100.0,10.0].gif" alt="firstorder" width="300"/>
  <img src="assets/VanillaNewtonsMethod_SkewedParaboloid_stepsize1.0_init[100.0,10.0].gif" alt="newton" width="300"/>
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

### :runner: Example run:
```bash
python main.py --optim dfp \
               --func skewedparaboloid \
               --stepsize 0.002 \
               --init 100 10 \
               --viz
```

## :alarm_clock: Time complexity comparison
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-0pky" colspan="2">total steps until convg.</th>
    <th class="tg-0pky" colspan="2">computation time per iterations</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">method</td>
    <td class="tg-0pky">s. paraboloid</td>
    <td class="tg-0pky">steep valley</td>
    <td class="tg-0pky">s. paraboloid</td>
    <td class="tg-0pky">steep valley</td>
  </tr>
  <tr>
    <td class="tg-0pky">First-order gradient method</td>
    <td class="tg-0pky">311</td>
    <td class="tg-0pky">4431</td>
    <td class="tg-0pky">0.231</td>
    <td class="tg-0pky">0.267</td>
  </tr>
  <tr>
    <td class="tg-0lax">Vanilla Newton’s method</td>
    <td class="tg-0lax">126</td>
    <td class="tg-0lax">5</td>
    <td class="tg-0lax">0.582</td>
    <td class="tg-0lax">0.549</td>
  </tr>
  <tr>
    <td class="tg-0lax">Davidon’s method</td>
    <td class="tg-0lax">292</td>
    <td class="tg-0lax">4</td>
    <td class="tg-0lax">0.489</td>
    <td class="tg-0lax">0.538</td>
  </tr>
  <tr>
    <td class="tg-0lax">Davidon-Fletcher-Powell method</td>
    <td class="tg-0lax">134</td>
    <td class="tg-0lax">4</td>
    <td class="tg-0lax">0.537</td>
    <td class="tg-0lax">0.747</td>
  </tr>
  <tr>
    <td class="tg-0lax">Broyden-Fletcher-Goldfarb-Shanno</td>
    <td class="tg-0lax">77</td>
    <td class="tg-0lax">4</td>
    <td class="tg-0lax">0.564</td>
    <td class="tg-0lax">0.556</td>
  </tr>
</tbody>
</table>
