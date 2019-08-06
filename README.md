# Operator Inference

This is a Python implementation of the operator learning approach for projection-based reduced order models of systems of ordinary differential equations.
The methodology is described in detail the following papers:

\[1\] Peherstorfer, B. and Willcox, K.
[Data-driven operator inference for non-intrusive projection-based model reduction.](https://www.sciencedirect.com/science/article/pii/S0045782516301104)
Computer Methods in Applied Mechanics and Engineering, 306:196-215, 2016.
([Download](https://cims.nyu.edu/~pehersto/preprints/Non-intrusive-model-reduction-Peherstorfer-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{Peherstorfer16DataDriven,
    title     = {Data-driven operator inference for nonintrusive projection-based model reduction},
    author    = {Peherstorfer, B. and Willcox, K.},
    journal   = {Computer Methods in Applied Mechanics and Engineering},
    volume    = {306},
    pages     = {196--215},
    year      = {2016},
    publisher = {Elsevier}
    }
}</pre></details>

\[2\] Qian, E., Kramer, B., Marques, A., and Willcox, K.
[Transform & Learn: A data-driven approach to nonlinear model reduction](https://arc.aiaa.org/doi/10.2514/6.2019-3707).
In the AIAA Aviation 2019 Forum, June 17-21, Dallas, TX. ([Download](https://www.dropbox.com/s/5znea6z1vntby3d/QKMW_aviation19.pdf?dl=0))<details><summary>BibTeX</summary><pre>
@inbook{QKMW2019aviation,
    author = {Qian, E. and Kramer, B. and Marques, A. N. and Willcox, K. E.},
    title  = {Transform \&amp; Learn: A data-driven approach to nonlinear model reduction},
    booktitle = {AIAA Aviation 2019 Forum},
    doi    = {10.2514/6.2019-3707},
    URL    = {https://arc.aiaa.org/doi/abs/10.2514/6.2019-3707},
    eprint = {https://arc.aiaa.org/doi/pdf/10.2514/6.2019-3707}
}</pre></details>

**Contributors**: [Renee Swischuk](mailto:swischuk@mit.edu), [Shane McQuarrie](https://github.com/shanemcq18), [Elizabeth Quian](), [Boris Kramer](http://web.mit.edu/bokramer/www/index.html).

See [this repository](https://github.com/elizqian/operator-inference) for a MATLAB implementation.

## Mathematical Setting

Consider the (possibly nonlinear) system of ordinary differential equations

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\dot{\mathbf{x}}(t)=\mathbf{f}(t,\mathbf{x}(t)),"/>
</p>

where

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\mathbf{x}:\mathbb{R}\to\mathbb{R}^n,\qquad\mathbf{f}:\mathbb{R}\times\mathbb{R}^n\to\mathbb{R}^n."/>
</p>

This system is called the _full-order model_ (FOM).
If _n_ is large (as it often is in applications), it is computationally expensive to numerically solve the FOM.
This package provides tools for constructing a _reduced-order model_ (ROM) that is linear or quadratic in the state **x**, possibly with a constant term **c**, and with optional control inputs **u**.
In the most general case, the code can construct and solve a system of the form

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}(t)=\hat{A}\hat{\mathbf{x}}+\hat{H}(\hat{\mathbf{x}}\otimes\hat{\mathbf{x}})+\hat{B}\mathbf{u}(t)+\sum_{k=1}^m\hat{N}_{k}\hat{\mathbf{x}}u_{k}(t)+\hat{\mathbf{c}},"/>
</p>

where now

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\hat{\mathbf{x}}:\mathbb{R}\to\mathbb{R}^r,\qquad\mathbf{u}:\mathbb{R}\to\mathbb{R}^m,\qquad\hat{\mathbf{c}}\in\mathbb{R}^r,\qquad%20r\ll%20n,"/>
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\hat{A}\in\mathbb{R}^{r\times%20r},\qquad\hat{H}\in\mathbb{R}^{r\times%20r^2},\qquad\hat{B}\in\mathbb{R}^{r\times%20m},\qquad\hat{N}_{k}\in\mathbb{R}^{r\times%20r}."/>
</p>

This reduced low-dimensional system approximates the original high-dimensional system, but it is much easier (faster) to solve because of its low dimension _r_ << _n_.


## Quick Start

#### Installation

```bash
$ pip3 install -i https://test.pypi.org/simple/operator-inference
```

_This installation command is temporary!_

#### Example

<!-- TODO: what are these variables?? -->

```python
from operator_inference import OpInf

# Define a model of the form x' = Ax + c (no input).
>>> lc_model = OpInf.Model('Lc', inp=False)

# Fit the model by solving for the operators A and c.
>>> lc_model.fit(r, k, xdot, xhat)

# Simulate the learned model for 10 timesteps of length .01.
>>> xr, n_steps = lc_model.predict(init=xhat[:,0],
                                   n_timesteps=10,
                                   dt=.01)
# Reconstruct the predictions.
>>> xr_rec = U[:,:r] @ xr
```

See [`opinf_demo.py`](https://github.com/swischuk/operator_inference/blob/master/opinf_demo.py) for a more complete working example.


## Documentation

### Model class

The following commands will initialize an operator inference `Model`.

```python
from operator_inference import OptInf

my_model = OptInf.Model(degree, inp)
```

Here `degree` is a string denoting the structure of
the model with the following options.

| `degree` | Model Description | Model Equation |
| :------- | :---------------- | :------------- |
|  `"L"`   |  linear | <img src="https://latex.codecogs.com/svg.latex?\dot{\mathbf{x}}(t)=A\mathbf{x}(t)"/>
|  `"Lc"`  |  linear with constant | <img src="https://latex.codecogs.com/svg.latex?\dot{\mathbf{x}}(t)=A\mathbf{x}(t)+\mathbf{c}"/>
|  `"Q"`   |  quadratic | <img src="https://latex.codecogs.com/svg.latex?\dot{\mathbf{x}}(t)=F\mathbf{x}^2(t)"/>
|  `"Qc"`  |  quadratic with constant | <img src="https://latex.codecogs.com/svg.latex?\dot{\mathbf{x}}(t)=F\mathbf{x}^2(t)+\mathbf{c}"/>
|  `"LQ"`  |  linear quadratic | <img src="https://latex.codecogs.com/svg.latex?\dot{\mathbf{x}}(t)=A\mathbf{x}(t)+F\mathbf{x}^2(t)"/>
|  `"LQc"` |  linear quadratic with constant | <img src="https://latex.codecogs.com/svg.latex?\dot{\mathbf{x}}(t)=A\mathbf{x}(t)+F\mathbf{x}^2(t)+\mathbf{c}"/>

The `inp` argument is a boolean (`True` or `False`) denoting whether or not there is an additive input term of the form <img src="https://latex.codecogs.com/svg.latex?B\mathbf{u}(t)"/>.

The script `opinf_demo.py` demonstrates the use of the operator inference model on data generated from the heat equation.
See [@mythesis] for the problem setup.

##### Methods

- `Model.fit(r, reg, xdot, xhat, u=None)`: Compute the operators of the reduced-order model that best fit the data by solving the regularized least
    squares problem
<p align="center"><img src="https://latex.codecogs.com/svg.latex?\underset{\mathbf{o}_i}{\text{min}}||D\mathbf{o}_i-\mathbf{r}||_2^2+k||P\mathbf{o}_i||_2^2."/></p>

- `predict(init, n_timesteps, dt, u=None)`: Simulate the learned model with an explicit Runge-Kutta scheme tailored to the structure of the model.

- `get_residual()`: Return the residuals of the least squares problem
<p align="center"><img src="https://latex.codecogs.com/svg.latex?||DO^T-\dot{X}^T||_F^2\qquad\text{and}\qquad||O^T||_F^2."/></p>

- `get_operators()`: Return each of the learned operators.

- `relative_error(predicted_data, true_data, thresh=1e-10)`: Compute the relative error between predicted data and true data, i.e.,
<p align="center"><img src="https://latex.codecogs.com/svg.latex?\frac{||\texttt{true\_data}-\texttt{predicted\_data}||}{||\texttt{true\_data}||}"./></p>
Computes absolute error (numerator only) if <img src="https://latex.codecogs.com/svg.latex?||\texttt{true\_data}||<\texttt{thresh}"/>.


### `opinf_helper.py`

Import the helper script with the following line.

```python
from operator_inference import opinf_helper
```

##### Functions

This file contains routines that are used within `OpInf.Model.fit()`.

- `normal_equations(D, r, k, num)`: Solve the normal equations corresponding to the regularized ordinary least squares problem <img src="https://latex.codecogs.com/svg.latex?\underset{\mathbf{o}_i}{\text{min}}||D\mathbf{o}_i-\mathbf{r}||_2^2+k||P\mathbf{o}_i||_2^2"/>.

-  `get_x_sq(X)`: Compute squared snapshot data.

-  `F2H(F)`: Convert quadratic operator `F` to symmetric quadratic operator `H` for simulating the learned system.


<!-- ### `integration_helpers.py`

Import the integration helper script with the following line.

```python
from operator_inference import integration_helpers
```

##### Functions

This file contains Runge-Kutta integrators that are used within `OpInf.Model.predict()`.
The choice of integrator depends on `Model.degree`.

- `rk4advance_L(x, dt, A, B=0, u=0)`
- `rk4advance_Lc(x, dt, A, c, B=0, u=0)`
- `rk4advance_Q(x, dt, H, B=0, u=0)`
- `rk4advance_Qc(x, dt, H, c, B=0, u=0)`
- `rk4advance_LQ(x, dt, A, H, B=0, u=0)`
- `rk4advance_LQc(x, dt, A, H, c, B=0, u=0)`

**Parameters**:
- `x ((r,) ndarray)`: The current (reduced-dimension) state.
- `dt (float)`: Time step size.
- `A ((r,r) ndarray)`: The linear state operator.
- `H ((r,r**2) ndarray)`: The matricized quadratic state operator.
- `c ((r,) ndarray)`: The constant term.
- `B ((r,p) ndarray)`: The input operator; only needed if `Model.inp` is `True`.
- `u ((p,) ndarray)`: The input at the current time; only needed if `Model.inp` is `True`.

**Returns**:
- `x_next ((r,) ndarray)`: The next (reduced-dimension) state. -->

## Summary of Mathematical Details

For a full treatment, see [\[1\]](https://www.sciencedirect.com/science/article/pii/S0045782516301104).


<p align="center"><img src="https://raw.githubusercontent.com/elizqian/operator-inference/master/modelform.png"
alt="$\dot{\hat {\mathbf{x}}} = \hat{\mathbf{A}}\hat{\mathbf{x}}+\hat{\mathbf{H}}(\hat{\mathbf{x}}\otimes\hat{\mathbf{x}})+\hat{\mathbf{B}}\mathbf{u}(t)+\sum_{i=1}^m\hat{\mathbf{N}}\hat{\mathbf{x}}u_i(t)+\hat{\mathbf{C}}$"
height=60/></p>

In the discrete-time setting the learned model can have any of the terms on the right-hand side of the following general model:

<p align="center"><img src="https://raw.githubusercontent.com/elizqian/operator-inference/master/model_discrete.png"
alt="$\hat {\mathbf{x}}_{k+1} = \hat{\mathbf{A}}\hat{\mathbf{x}}_k+\hat{\mathbf{H}}(\hat{\mathbf{x}}_k\otimes\hat{\mathbf{x}}_k)+\hat{\mathbf{B}}\mathbf{u}_k+\sum_{i=1}^m\hat{\mathbf{N}}\hat{\mathbf{x}}_ku_{i,k}+\hat{\mathbf{C}}$"
height=60/></p>

## Examples
The [`examples/`](https://github.com/swischuk/operator_inference/blob/master/examples/) folder contains scripts that set up and run several examples:
- The heat equation example from [\[1\]](https://www.sciencedirect.com/science/article/pii/S0045782516301104).
- The Burgers' equation from [\[1\]](https://www.sciencedirect.com/science/article/pii/S0045782516301104).
- The Euler equation example from [\[2\]](https://arc.aiaa.org/doi/10.2514/6.2019-3707). This example uses MATLAB's Curve Fitting Toolbox to generate the random initial conditions.
