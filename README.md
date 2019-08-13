# Operator Inference

This is a Python implementation of the operator learning approach for projection-based reduced order models of systems of ordinary differential equations.
The procedure is **data-driven** and **non-intrusive**, making it a viable candidate for model reduction of black-box or complex systems.
The methodology is described in detail the following papers:

- \[1\] Peherstorfer, B. and Willcox, K.
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
}</pre></details>

- \[2\] Qian, E., Kramer, B., Marques, A., and Willcox, K.
[Transform & Learn: A data-driven approach to nonlinear model reduction](https://arc.aiaa.org/doi/10.2514/6.2019-3707).
In the AIAA Aviation 2019 Forum, June 17-21, Dallas, TX. ([Download](https://www.dropbox.com/s/5znea6z1vntby3d/QKMW_aviation19.pdf?dl=0))<details><summary>BibTeX</summary><pre>
@inbook{QKMW2019aviation,
    author    = {Qian, E. and Kramer, B. and Marques, A. N. and Willcox, K. E.},
    title     = {Transform \&amp; Learn: A data-driven approach to nonlinear model reduction},
    booktitle = {AIAA Aviation 2019 Forum},
    doi       = {10.2514/6.2019-3707},
    URL       = {https://arc.aiaa.org/doi/abs/10.2514/6.2019-3707},
    eprint    = {https://arc.aiaa.org/doi/pdf/10.2514/6.2019-3707}
}</pre></details>

**Contributors**: [Renee Swischuk](https://github.com/swischuk), [Shane McQuarrie](https://github.com/shanemcq18), [Elizabeth Qian](https://github.com/elizqian), [Boris Kramer](https://github.com/bokramer).

See [this repository](https://github.com/elizqian/operator-inference) for a MATLAB implementation.

**Contents**
- [**Problem Statement**](#problem-statement)
- [**Quick Start**](#quick-start)
- [**Documentation**](#documentation)
- [**Examples**](#examples)

## Problem Statement

Consider the (possibly nonlinear) system of _n_ ordinary differential equations with state variable **x**, input (control) variable **u**, and independent variable _t_:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\dot{\mathbf{x}}(t)=\mathbf{f}(t,\mathbf{x}(t),\mathbf{u}(t)),"/>
</p>

where

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\mathbf{x}:\mathbb{R}\to\mathbb{R}^n,\qquad\mathbf{u}:\mathbb{R}\to\mathbb{R}^m,\qquad\mathbf{f}:\mathbb{R}\times\mathbb{R}^n\times\mathbb{R}^m\to\mathbb{R}^n."/>
</p>

This system is called the _full-order model_ (FOM).
If _n_ is large, as it often is in high-consequence engineering applications, it is computationally expensive to numerically solve the FOM.
This package provides tools for constructing a _reduced-order model_ (ROM) that is linear or quadratic in the state **x**, possibly with a constant term **c**, and with optional control inputs **u**.
The procedure is data-driven, non-intrusive, and relatively inexpensive.
In the most general case, the code can construct and solve a system of the form

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}(t)=\hat{A}\hat{\mathbf{x}}(t)+\hat{H}(\hat{\mathbf{x}}\otimes\hat{\mathbf{x}})(t)+\hat{B}\mathbf{u}(t)+\hat{\mathbf{c}},"/>
</p>

<!-- <p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}(t)=\hat{A}\hat{\mathbf{x}}(t)+\hat{H}(\hat{\mathbf{x}}\otimes\hat{\mathbf{x}})(t)+\hat{B}\mathbf{u}(t)+\sum_{i=1}^m\hat{N}_{i}\hat{\mathbf{x}}(t)u_{i}(t)+\hat{\mathbf{c}},"/>
</p> -->

where now

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\hat{\mathbf{x}}:\mathbb{R}\to\mathbb{R}^r,\qquad\mathbf{u}:\mathbb{R}\to\mathbb{R}^m,\qquad\hat{\mathbf{c}}\in\mathbb{R}^r,\qquad%20r\ll%20n,"/>
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\hat{A}\in\mathbb{R}^{r\times%20r},\qquad\hat{H}\in\mathbb{R}^{r\times%20r^2},\qquad\hat{B}\in\mathbb{R}^{r\times%20m}."/>
</p>

<!-- <p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\hat{A}\in\mathbb{R}^{r\times%20r},\qquad\hat{H}\in\mathbb{R}^{r\times%20r^2},\qquad\hat{B}\in\mathbb{R}^{r\times%20m},\qquad\hat{N}_{i}\in\mathbb{R}^{r\times%20r}."/>
</p> -->

This reduced low-dimensional system approximates the original high-dimensional system, but it is much easier (faster) to solve because of its low dimension _r_ << _n_.

See [DETAILS.md](DETAILS.md) for more mathematical details and an index of notation.

## Quick Start

#### Installation

```bash
$ pip3 install -i https://test.pypi.org/simple/ rom-operator-inference-shanemcq18
```

_**This installation command is very temporary!**_

#### Example

<!-- TODO: what are these variables?? -->

```python
import rom_operator_inference as roi

# Define a model of the form x' = Ax + c (no input).
>>> lc_model = roi.ReducedModel('Lc', inp=False)

# Fit the model to projected data X_, the projected derivative Xdot_,
# and the linear basis Vr by solving for the operators A_ and c_.
>>> lc_model.fit(X, Xdot, Vr)

# Simulate the learned model over the time domain [0,1] with 100 timesteps.
>>> t = np.linspace(0, 1, 100)
>>> X_ROM = lc_model.predict(X[:,0], t)
```

See the [**Examples**](#examples) section for a list of complete working examples.


## Documentation

#### ReducedModel class

The following commands will initialize an operator inference `ReducedModel`.

```python
import rom_operator_inference as roi

my_model = roi.ReducedModel(modelform, inp)
```

Here `modelform` is a string denoting the structure of
the desired ROM with the following options.

| `modelform` | Model Description | Model Equation |
| :------- | :---------------- | :------------- |
|  `"L"`   |  linear | <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}(t)=\hat{A}{\hat{\mathbf{x}}(t)"/>
|  `"Lc"`  |  linear with constant | <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}(t)=\hat{A}{\hat{\mathbf{x}}(t)+\hat{\mathbf{c}}"/>
|  `"Q"`   |  quadratic | <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}(t)=\hat{H}(\hat{\mathbf{x}}\otimes\hat{\mathbf{x}})(t)"/>
|  `"Qc"`  |  quadratic with constant | <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}(t)=\hat{H}(\hat{\mathbf{x}}\otimes\hat{\mathbf{x}})(t)+\hat{\mathbf{c}}"/>
|  `"LQ"`  |  linear quadratic | <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}(t)=\hat{A}\hat{\mathbf{x}}(t)+\hat{H}(\hat{\mathbf{x}}\otimes\hat{\mathbf{x}})(t)"/>
|  `"LQc"` |  linear quadratic with constant | <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}(t)=\hat{A}\hat{\mathbf{x}}(t)+\hat{H}(\hat{\mathbf{x}}\otimes\hat{\mathbf{x}})(t)+\hat{\mathbf{c}}"/>

The `inp` argument is a boolean (`True` or `False`) denoting whether or not there is an additive input term of the form <img src="https://latex.codecogs.com/svg.latex?B\mathbf{u}(t)"/>.


##### Methods

- `ReducedModel.fit(X_, Xdot_, U=None, reg=0)`: Compute the operators of the reduced-order model that best fit the data by solving the regularized least
    squares problem
<p align="center"><img src="https://latex.codecogs.com/svg.latex?\underset{\mathbf{o}_i}{\text{min}}||D\mathbf{o}_i-\mathbf{r}||_2^2+\lambda||P\mathbf{o}_i||_2^2."/></p>

- `ReducedModel.predict(init, n_timesteps, dt, u=None)`: Simulate the learned model with an explicit Runge-Kutta scheme tailored to the structure of the model.

- `ReducedModel.get_residual()`: Return the residuals of the least squares problem
<p align="center"><img src="https://latex.codecogs.com/svg.latex?||DO^T-\dot{X}^T||_F^2\qquad\text{and}\qquad||O^T||_F^2."/></p>

- `ReducedModel.get_operators()`: Return each of the learned operators.

- `ReducedModel.relative_error(predicted_data, true_data, thresh=1e-10)`: Compute the relative error between predicted data and true data, i.e.,
<p align="center"><img src="https://latex.codecogs.com/svg.latex?\frac{||\texttt{true\_data}-\texttt{predicted\_data}||}{||\texttt{true\_data}||}"./></p> Computes absolute error (numerator only) in the case that <img src="https://latex.codecogs.com/svg.latex?||\texttt{true\_data}||<\texttt{thresh}."/>


#### `opinf_helper.py`

Import the helper script with the following line.

```python
from rom_operator_inference import opinf_helper
```

##### Functions

This file contains helper routines that are used internally for `OpInf.ReducedModel.fit()`.

- `normal_equations(D, r, k, num)`: Solve the normal equations corresponding to the regularized ordinary least squares problem
<p align="center"><img src="https://latex.codecogs.com/svg.latex?\underset{\mathbf{o}_i}{\text{min}}||D\mathbf{o}_i-\mathbf{r}||_2^2+\lambda||P\mathbf{o}_i||_2^2."/></p>

-  `get_x_sq(X)`: Compute squared snapshot data.

-  `F2H(F)`: Convert quadratic operator `F` to "symmetric" quadratic operator `H` for simulating the learned system.


## Examples

_**WARNING: under construction!!**_

The [`examples/`](examples/) folder contains scripts and notebooks that set up and run several examples:
- The heat equation example from [\[1\]](https://www.sciencedirect.com/science/article/pii/S0045782516301104).
- The Burgers' equation from [\[1\]](https://www.sciencedirect.com/science/article/pii/S0045782516301104).
- The Euler equation example from [\[2\]](https://arc.aiaa.org/doi/10.2514/6.2019-3707).
This example uses MATLAB's Curve Fitting Toolbox to generate the random initial conditions.
- The script `opinf_demo.py` demonstrates the use of the operator inference model on data generated from the heat equation.
See [@ReneeThesis] for the problem setup.

<!-- TODO: actual links to the folders or files -->
