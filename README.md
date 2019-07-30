# Operator Inference Package

**Author**: Renee Swischuk (swischuk@mit.edu)

**Contributors**: [Shane McQuarrie](https://github.com/shanemcq18)

Consider the possibly nonlinear ordinary differential equation
<img src="https://latex.codecogs.com/svg.latex?\dot{\mathbf{x}}(t)%20=%20\mathbf{f}(t,\mathbf{x}(t)),"/>
where <img src="https://latex.codecogs.com/svg.latex?\mathbf{x}(t)"/> is a vector-valued function mapping to <img src="https://latex.codecogs.com/svg.latex?\mathbb{R}^n"/>, called the _state_.
The `operator_inference` package provides tools for constructing a reduced-order model that is linear or quadratic in <img src="https://latex.codecogs.com/svg.latex?\mathbf{x}"/>, possibly with a constant term <img src="https://latex.codecogs.com/svg.latex?\mathbf{c}"/>, and with optional linear control inputs <img src="https://latex.codecogs.com/svg.latex?\mathbf{u}(t)"/>.

#### Package Contents
- The [`Model`](https://github.com/swischuk/operator_inference#model-class) class,
- A helper script [`opinf_helper.py`](https://github.com/swischuk/operator_inference#opinf-helper), and
- A helper script [`integration_helpers.py`](https://github.com/swischuk/operator_inference#integration-helpers).

## Quick Start

#### Installation

Install the package on the command line with the following command.

`pip3 install -i https://test.pypi.org/simple/operator-inference`

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

## Model class

The following commands will initialize an operator inference `Model`.

```python
from operator_inference import OptInf

my_model = OptInf.Model(degree, inp)
```

Here `degree` is a string denoting the structure of
the model with the following options.

| `degree` | Model Description | Model Equation |
| :------- | :---------------- | :------------- |
|  `"L"`   |  linear | <img src="https://latex.codecogs.com/svg.latex?\dot{\mathbf{x}}(t) = A\mathbf{x}(t)"/>
|  `"Lc"`  |  linear with constant | <img src="https://latex.codecogs.com/svg.latex?\dot{\mathbf{x}}(t) = A\mathbf{x}(t) + \mathbf{c}"/>
|  `"Q"`   |  quadratic | <img src="https://latex.codecogs.com/svg.latex?\dot{\mathbf{x}}(t) = F\mathbf{x}^2(t)"/>
|  `"Qc"`  |  quadratic with constant | <img src="https://latex.codecogs.com/svg.latex?\dot{\mathbf{x}}(t) = F\mathbf{x}^2(t) + \mathbf{c}"/>
|  `"LQ"`  |  linear quadratic | <img src="https://latex.codecogs.com/svg.latex?\dot{\mathbf{x}}(t) = A\mathbf{x}(t) + F\mathbf{x}^2(t)"/>
|  `"LQc"` |  linear quadratic with constant | <img src="https://latex.codecogs.com/svg.latex?\dot{\mathbf{x}}(t) = A\mathbf{x}(t) + F\mathbf{x}^2(t) + \mathbf{c}"/>

The `inp` argument is a boolean (`True` or `False`) denoting whether or not there is an additive input term of the form <img src="https://latex.codecogs.com/svg.latex?B\mathbf{u}(t)"/>.

The script `opinf_demo.py` demonstrates the use of the operator inference model on data generated from the heat equation.
See [@mythesis] for the problem setup.

#### Methods

- `Model.fit(r, reg, xdot, xhat, u=None)`: Compute the operators of the reduced-order model that best fit the data by solving the regularized least
    squares problems <img src="https://latex.codecogs.com/svg.latex?\underset{\mathbf{o}_i}{\text{min}}||D \mathbf{o}_i - \mathbf{r}||_2^2 + k||P \mathbf{o}_i||_2^2"/>.

- `predict(init, n_timesteps, dt, u = None)`: Simulate the learned model with an explicit Runge-Kutta scheme tailored to the structure of the model.

- `get_residual()`: Return the residuals of the least squares problem <img src="https://latex.codecogs.com/svg.latex?||D O^T - \dot{X}^T||_F^2"/> and <img src="https://latex.codecogs.com/svg.latex?||O^T||_F^2"/>.

- `get_operators()`: Return each of the learned operators.

- `relative_error(predicted_data, true_data, thresh=1e-10)`: Compute the relative error between predicted data and true data, i.e., <img src="https://latex.codecogs.com/svg.latex?||\text{true} - \text{predicted}|| / ||\text{true}||"/>. Computes absolute error (numerator only) if <img src="https://latex.codecogs.com/svg.latex?||\text{true}|| < "/> `thresh`.


## `opinf_helper.py`

Import the helper script with the following line.

```python
from operator_inference import opinf_helper
```

#### Functions

This file contains routines that are used within `OpInf.Model.fit()`.

- `normal_equations(D, r, k, num)`: Solve the normal equations corresponding to the regularized ordinary least squares problem <img src="https://latex.codecogs.com/svg.latex?\underset{\mathbf{o}_i}{\text{min}}||D \mathbf{o}_i - \mathbf{r}||_2^2 + k||P \mathbf{o}_i||_2^2"/>.

-  `get_x_sq(X)`: Compute squared snapshot data as in [@ben].

-  `F2H(F)`: Convert quadratic operator `F` to symmetric quadratic operator `H` for simulating the learned system.


<!-- ## `integration_helpers.py`

Import the integration helper script with the following line.

```python
from operator_inference import integration_helpers
```

#### Functions

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
