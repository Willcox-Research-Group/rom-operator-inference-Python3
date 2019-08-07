# Operator Inference

This is a Python implementation of the operator learning approach for projection-based reduced order models of systems of ordinary differential equations.
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

**Contributors**: [Renee Swischuk](mailto:swischuk@mit.edu), [Shane McQuarrie](https://github.com/shanemcq18), [Elizabeth Quian](), [Boris Kramer](http://web.mit.edu/bokramer/www/index.html).

See [this repository](https://github.com/elizqian/operator-inference) for a MATLAB implementation.

## Problem Setting

Consider the (possibly nonlinear) system of _n_ ordinary differential equations

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

#### Model class

The following commands will initialize an operator inference `Model`.

```python
from operator_inference import OpInf

my_model = OpInf.Model(degree, inp)
```

Here `degree` is a string denoting the structure of
the desired ROM with the following options.

| `degree` | Model Description | Model Equation |
| :------- | :---------------- | :------------- |
|  `"L"`   |  linear | <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}(t)=\hat{A}{\hat{\mathbf{x}}(t)"/>
|  `"Lc"`  |  linear with constant | <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}(t)=\hat{A}{\hat{\mathbf{x}}(t)+\hat{\mathbf{c}}"/>
|  `"Q"`   |  quadratic | <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}(t)=\hat{H}(\hat{\mathbf{x}}\otimes\hat{\mathbf{x}})(t)"/>
|  `"Qc"`  |  quadratic with constant | <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}(t)=\hat{H}(\hat{\mathbf{x}}\otimes\hat{\mathbf{x}})(t)+\hat{\mathbf{c}}"/>
|  `"LQ"`  |  linear quadratic | <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}(t)=\hat{A}\hat{\mathbf{x}}(t)+\hat{H}(\hat{\mathbf{x}}\otimes\hat{\mathbf{x}})(t)"/>
|  `"LQc"` |  linear quadratic with constant | <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}(t)=\hat{A}\hat{\mathbf{x}}(t)+\hat{H}(\hat{\mathbf{x}}\otimes\hat{\mathbf{x}})(t)+\hat{\mathbf{c}}"/>

The `inp` argument is a boolean (`True` or `False`) denoting whether or not there is an additive input term of the form <img src="https://latex.codecogs.com/svg.latex?B\mathbf{u}(t)"/>.


##### Methods

- `Model.fit(r, reg, xdot, xhat, u=None)`: Compute the operators of the reduced-order model that best fit the data by solving the regularized least
    squares problem
<p align="center"><img src="https://latex.codecogs.com/svg.latex?\underset{\mathbf{o}_i}{\text{min}}||D\mathbf{o}_i-\mathbf{r}||_2^2+\lambda||P\mathbf{o}_i||_2^2."/></p>

- `predict(init, n_timesteps, dt, u=None)`: Simulate the learned model with an explicit Runge-Kutta scheme tailored to the structure of the model.

- `get_residual()`: Return the residuals of the least squares problem
<p align="center"><img src="https://latex.codecogs.com/svg.latex?||DO^T-\dot{X}^T||_F^2\qquad\text{and}\qquad||O^T||_F^2."/></p>

- `get_operators()`: Return each of the learned operators.

- `relative_error(predicted_data, true_data, thresh=1e-10)`: Compute the relative error between predicted data and true data, i.e.,
<p align="center"><img src="https://latex.codecogs.com/svg.latex?\frac{||\texttt{true\_data}-\texttt{predicted\_data}||}{||\texttt{true\_data}||}"./></p> Computes absolute error (numerator only) in the case that <img src="https://latex.codecogs.com/svg.latex?||\texttt{true\_data}||<\texttt{thresh}."/>


#### `opinf_helper.py`

Import the helper script with the following line.

```python
from operator_inference import opinf_helper
```

##### Functions

This file contains helper routines that are used internally for `OpInf.Model.fit()`.

- `normal_equations(D, r, k, num)`: Solve the normal equations corresponding to the regularized ordinary least squares problem
<p align="center"><img src="https://latex.codecogs.com/svg.latex?\underset{\mathbf{o}_i}{\text{min}}||D\mathbf{o}_i-\mathbf{r}||_2^2+\lambda||P\mathbf{o}_i||_2^2."/></p>

-  `get_x_sq(X)`: Compute squared snapshot data.

-  `F2H(F)`: Convert quadratic operator `F` to "symmetric" quadratic operator `H` for simulating the learned system.


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
However, note that some notation has been altered for coding convenience and clarity.

### Projection-based Model Reduction

Model reduction via projection occurs in three steps:
1. (**Data Collection**) Gather snapshot data, i.e., solutions to the full-order model (FOM) at various times / parameters.
2. (**Representation**) Compute a low-rank basis (which defines a low-dimensional linear subspace) that captures most of the behavior of the snapshots.
3. (**Projection**) Use the low-rank basis to construct a low-dimensional ODE (the ROM) that approximates the FOM.

<!-- These steps comprise what is called the _offline phase_ in the literature, since they can all be done before the resulting ROM is simulated. -->

This package focuses on step 3, constructing the ROM given the snapshot data and the low-rank basis from steps 1 and 2, respectively.

Let _X_ be the _n_ x _k_ matrix whose _k_ columns are snapshots of length _n_ (step 1), and let _V_<sub>_r_</sub> be an orthonormal _n_ x _r_ matrix representation for an _r_-dimensional subspace (step 2).
For example, a common choice for _V_<sub>_r_</sub> is the POD Basis of rank _r_, the matrix comprised of the first _r_ singular vectors of _X_.

The classical approach to step 3 is to make the Ansatz

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\mathbf{x}(t)\approx%20V_{r}\hat{\mathbf{x}}(t),">
</p>

which, inserted into the FOM, yields

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?V_{r}\dot{\hat{\mathbf{x}}}(t)=\mathbf{f}(t,V_{r}\hat{\mathbf{x}}(t))."/>
</p>

Since _V_<sub>_r_</sub> is orthogonal, multiplying both sides by the transpose gives

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}(t)=V_{r}^\mathsf{T}\mathbf{f}(t,V_{r}\hat{\mathbf{x}}(t))=:\hat{\mathbf{f}}(t,\hat{\mathbf{x}}(t))."/>
</p>

Note that this system is _r_-dimensional.
If the FOM operator **f** is known and has a nice structure, this reduced system can be solved cheaply by precomputing any involved matrices and then applying a time-stepping scheme.
For example, if **f** is linear in **x**, then

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\mathbf{f}(t,\mathbf{x}(t))=A\mathbf{x}(t)\qquad\Longrightarrow\qquad\hat{\mathbf{f}}(t,\hat{\mathbf{x}}(t))=V_r^\mathsf{T}AV_r\hat{\mathbf{x}}(t)=\hat{A}\hat{\mathbf{x}},"/>
</p>

where

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\hat{A}:=V_r^\mathsf{T}AV_r\in\mathbb{R}^{r\times%20r}."/>
</p>

However, this approach breaks down if the FOM operator **f** is unknown, uncertain, or highly nonlinear.

### Operator Inference via Least Squares

Instead of directly computing reduced operators, the operator inference framework takes a more data-driven approach: assuming a specific structure of the ROM (linear, quadratic, etc.), solve for the involved operators that best fit the data.
For example, suppose that we seek a ROM of the form

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}(t)=\hat{A}\hat{\mathbf{x}}(t)+\hat{H}(\hat{\mathbf{x}}\otimes\hat{\mathbf{x}})(t)+\hat{B}\mathbf{u}(t),"/>
</p>

where we have now introduced a control (input) variable **u**.
We have only the snapshot matrix _X_, the low-rank basis matrix _V_<sub>_r_</sub>, the inputs _U_, and perhaps the snapshot velocities _X'_ (if not, these must be approximated).
Here the (_ij_)<sup>th</sup> entry of _U_ is the _i_<sup>th</sup> component of **u** at the time corresponding to the _j_<sup>th</sup> snapshot.
To solve for the linear operators on the right-hand side of the preceding equation, we solve the least squares problem

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\underset{\hat{A}\in\mathbb{R}^{r\times%20r},\hat{H}\in\mathbb{R}^{r\times%20r^2},\hat{B}\in\mathbb{R}^{r\times%20r}}{\text{min}}\,\Big\|\hat{X}^\mathsf{T}\hat{A}^\mathsf{T}+\big(\hat{X}\otimes\hat{X}\big)^\mathsf{T}\hat{H}^\mathsf{T}+U^\mathsf{T}\hat{B}^\mathsf{T}-\dot{\hat{X}}^\mathsf{T}\Big\|_{F}^2."/>
</p>

This problem decouples into _r_ independent least-squares problems, so it is relatively inexpensive to solve.
The code allows for a Tikhonov regularization factor, which prevents numerical instabilities from dominating the computation.

It can be shown [\[1\]](https://www.sciencedirect.com/science/article/pii/S0045782516301104) that under some idealized assumptions, these inferred operators converge to the operators computed by explicit projection.

##### Kronecker Product

The vector [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product) ⊗ introduces some redundancies.
For example, the product **x** ⊗ **x** contains both _x_<sub>1</sub>_x_<sub>2</sub> and _x_<sub>2</sub>_x_<sub>1</sub>.
To avoid these redundancies, we introduce a "compact" Kronecker product <img src="https://latex.codecogs.com/svg.latex?\widetilde{\otimes}" height=10/> which only computes the unique terms of the usual vector Kronecker product:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\mathbf{x}\,\widetilde{\otimes}\,\mathbf{x}=\left[\begin{array}{c}\mathbf{x}^{(1)}\\\vdots\\\mathbf{x}^{(n)}\end{array}\right]\in\mathbb{R}^{n(n+1)/2},\qquad\text{where}\qquad\mathbf{x}^{(i)}=x_{i}\left[\begin{array}{c}x_{1}\\\vdots\\x_{i}\end{array}\right]."/>
</p>

When the compact Kronecker product is used, we call the resulting operator _F_ instead of _H_.
Thus, the full reduced order model becomes

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}(t)=\hat{A}\hat{\mathbf{x}}(t)+\hat{F}(\hat{\mathbf{x}}\,\widetilde{\otimes}\,\hat{\mathbf{x}})(t)+\hat{B}\mathbf{u}(t),"/>
</p>

and the corresponding operator inference least squares problem is

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\underset{\hat{A}\in\mathbb{R}^{r\times%20r},\,\hat{F}\in\mathbb{R}^{r\times(r(r+1)/2)},\,\hat{B}\in\mathbb{R}^{r\times%20r}}{\text{min}}\,\Big\|\hat{X}^\mathsf{T}\hat{A}^\mathsf{T}+\big(\hat{X}\,\widetilde{\otimes}\,\hat{X}\big)^\mathsf{T}\hat{F}^\mathsf{T}+U^\mathsf{T}\hat{B}^\mathsf{T}-\dot{\hat{X}}^\mathsf{T}\Big\|_{F}^2."/>
</p>

For our purposes, any ⊗ or <img src="https://latex.codecogs.com/svg.latex?\widetilde{\otimes}" height=10/> between used for matrices denotes a column-wise Kronecker product (also called the [Khatri-Rao product](https://en.wikipedia.org/wiki/Kronecker_product#Khatri%E2%80%93Rao_product)).


### Index of Notation

We generally denote scalars in lower case, vectors in bold lower case, matrices in upper case, and indicate low-dimensional quantities with a hat.
In the code, a low-dimensional quantity ends with an underscore, which matches scikit-learn conventions for the `Model` class.

##### Dimensions

| Symbol | Code | Description |
| :----: | :--- | :---------- |
| <img src="https://latex.codecogs.com/svg.latex?n"/> | `n`  | Dimension of the full-order system (large) |
| <img src="https://latex.codecogs.com/svg.latex?r"/> | `r`  | Dimension of the reduced-order system (small) |
| <img src="https://latex.codecogs.com/svg.latex?m"/> | `m`  | Dimension of the input **u** |
| <img src="https://latex.codecogs.com/svg.latex?k"/> | `k`  | Number of state snapshots, i.e., the number of training points |
| <img src="https://latex.codecogs.com/svg.latex?s"/> | `s`  | Number of unique quadratic reduced-state interactions, _r_(_r_+1)/2 |

<!-- TODO: number of time steps -->

<!-- | <img src="https://latex.codecogs.com/svg.latex?q"/> | `q`             | Dimension of the output **y** | -->
<!-- | <img src="https://latex.codecogs.com/svg.latex?p"/> | `p`             | Dimension of the paramteter space | -->
<!-- | <img src="https://latex.codecogs.com/svg.latex?d"/> | `d`             | Dimension of the spatial domain | -->

##### Vectors

<!-- \sigma_j\in\text{diag}(\Sigma) &= \textrm{singular value of }X\\
\boldsymbol{\mu}\in\mathcal{P} &= \text{system parameter}\\
\mathcal{P}\subset\mathbb{R}^{p} &= \text{parameter space}\\
\Omega\subset\mathbb{R}^{d} &= \text{spatial domain}\\
% \omega\in\Omega &= \text{spatial point (one dimension)}\\
\boldsymbol{\omega}\in\Omega &= \text{spatial point}\\
t\ge 0 &= \text{time}\\
\hat{} &= \textrm{reduced variable, e.g., }\hat{\mathbf{x}}\textrm{ or }\hat{A}\\
\dot{} = \frac{d}{dt} &= \text{time derivative} -->


| Symbol | Code | Size | Description |
| :----: | :--- | :--: | :---------- |
| <img src="https://latex.codecogs.com/svg.latex?\mathbf{x}"/> | `x` | <img src="https://latex.codecogs.com/svg.latex?n"/> | Full-order state vector |
| <img src="https://latex.codecogs.com/svg.latex?\hat{\mathbf{x}}"/> | `x_` | <img src="https://latex.codecogs.com/svg.latex?r"/> | Reduced-order state vector |
| <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}"/> | `xdot_` | <img src="https://latex.codecogs.com/svg.latex?r"/> | Reduced-order state velocity vector |
| <img src="https://latex.codecogs.com/svg.latex?\mathbf{x}_\text{ROM}"/> | `x_ROM` | <img src="https://latex.codecogs.com/svg.latex?n"/> | Approximation to **x** produced by ROM |
| <img src="https://latex.codecogs.com/svg.latex?\mathbf{u}"/> | `u` | <img src="https://latex.codecogs.com/svg.latex?m"/> | Input vector  |
| <img src="https://latex.codecogs.com/svg.latex?\hat{\mathbf{f}}"/> | `f_()` | <img src="https://latex.codecogs.com/svg.latex?n"/>  | ROM system operator |
| <img src="https://latex.codecogs.com/svg.latex?\mathbf{x}\otimes\mathbf{x}"/> | `np.kron(x,x)` | <img src="https://latex.codecogs.com/svg.latex?n^2"/> | Kronecker product of full state (quadratic terms) |
| <img src="https://latex.codecogs.com/svg.latex?\hat{\mathbf{x}}\otimes\hat{\mathbf{x}}"/> | `np.kron(x_,x_)` | <img src="https://latex.codecogs.com/svg.latex?r^2"/>  | Kronecker product of reduced state (quadratic terms) |
| <img src="https://latex.codecogs.com/svg.latex?\hat{\mathbf{x}}\,\widetilde{\otimes}\,\hat{\mathbf{x}}"/> | `kron_thin(x_,x_)` | <img src="https://latex.codecogs.com/svg.latex?\frac{r(r+1)}{2}"/>  | Compact Kronecker product of reduced state (quadratic terms) |
| <img src="https://latex.codecogs.com/svg.latex?\mathbf{v}_j"/> | `vj` | <img src="https://latex.codecogs.com/svg.latex?n"/> | _j_<sup>th</sup> POD basis vector, i.e., column _j_ of _V_<sub>_r_</sub> |

<!-- | **y**  | `y`             | Output vector | -->
<!-- | **y_ROM**, **y~** | `y_ROM`      | Approximation to **y** produced by ROM | -->

##### Matrices

| Symbol | Code | Shape | Description |
| :----: | :--- | :---: | :---------- |
| <img src="https://latex.codecogs.com/svg.latex?X"> | `X` | <img src="https://latex.codecogs.com/svg.latex?n\times%20k"/> | Snapshot matrix |
| <img src="https://latex.codecogs.com/svg.latex?\dot{X}"> | `Xdot` | <img src="https://latex.codecogs.com/svg.latex?n\times%20k"/> | Snapshot velocity matrix |
| <img src="https://latex.codecogs.com/svg.latex?V_r"/> | `Vr` | <img src="https://latex.codecogs.com/svg.latex?n\times%20r"/> | POD basis of rank _r_ |
| <img src="https://latex.codecogs.com/svg.latex?U"/> | `U` | <img src="https://latex.codecogs.com/svg.latex?m\times%20k"/> | Input matrix (inputs corresonding to the snapshots) |
| <img src="https://latex.codecogs.com/svg.latex?\hat{A}"/> | `A_` | <img src="https://latex.codecogs.com/svg.latex?r\times%20r"/> | Learned state matrix |
| <img src="https://latex.codecogs.com/svg.latex?\hat{B}"/> | `B_` | <img src="https://latex.codecogs.com/svg.latex?r\times%20m"/> | Learned input matrix |
| <img src="https://latex.codecogs.com/svg.latex?\hat{H}"/> | `H_` | <img src="https://latex.codecogs.com/svg.latex?r\times%20r^2"/> | Learned matricized quadratic tensor |
| <img src="https://latex.codecogs.com/svg.latex?\hat{F}"/> | `F_` | <img src="https://latex.codecogs.com/svg.latex?r\times\frac{r(r+1)}{2}"/> | Learned matricized quadratic tensor without redundancy |

<!-- | <img src="https://latex.codecogs.com/svg.latex?\hat{N}_i"/> | `Ni_` | <img src="https://latex.codecogs.com/svg.latex?r\times%20r"/> | Bilinear state-input matrix for _i_th input | -->

<!-- | <img src="https://latex.codecogs.com/svg.latex?\hat{C}"/> | `C_` | <img src="https://latex.codecogs.com/svg.latex?q\times%20r"/> | Learned output matrix | -->
<!-- | <img src="https://latex.codecogs.com/svg.latex?\hat{E}"/> | `E_` | <img src="https://latex.codecogs.com/svg.latex?r\times%20r"/> | Learned mass matrix | -->

<!-- I_{a\times%20a}\in\mathbb{R}^{a\times a} | | identity matrix\\ -->
<!-- \Sigma \in \mathbb{R}^{\ell\times\ell} &= \text{diagonal singular value matrix}\\ -->

### Other References

- \[3\] Swischuk, R. and Mainini, L. and Peherstorfer, B. and Willcox, K.
[Projection-based model reduction: Formulations for physics-based machine learning.](https://www.sciencedirect.com/science/article/pii/S0045793018304250)
Computers & Fluids 179:704-717, 2019.
([Download](https://kiwi.oden.utexas.edu/papers/Physics-based-machine-learning-swischuk-willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{swischuk2019projection,
  title    = {Projection-based model reduction: Formulations for physics-based machine learning},
  author   = {Swischuk, Renee and Mainini, Laura and Peherstorfer, Benjamin and Willcox, Karen},
  journal  = {Computers \& Fluids},
  volume   = {179},
  pages    = {704--717},
  year     = {2019},
  publisher={Elsevier}
}</pre></details>

- [4] Swischuck, R. Physics-based machine learning and data-driven reduced-order modeling, MIT Thesis
<!-- TODO: Renee's MIT masters thesis -->

## Examples

_**WARNING: under construction!!**_

The [`examples/`](https://github.com/swischuk/operator_inference/blob/master/examples/) folder contains scripts that set up and run several examples:
- The heat equation example from [\[1\]](https://www.sciencedirect.com/science/article/pii/S0045782516301104).
- The Burgers' equation from [\[1\]](https://www.sciencedirect.com/science/article/pii/S0045782516301104).
- The Euler equation example from [\[2\]](https://arc.aiaa.org/doi/10.2514/6.2019-3707).
This example uses MATLAB's Curve Fitting Toolbox to generate the random initial conditions.
- The script `opinf_demo.py` demonstrates the use of the operator inference model on data generated from the heat equation.
See [@ReneeThesis] for the problem setup.
