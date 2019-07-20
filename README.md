# Operator Inference Package

**Author**: Renee Swischuk (swischuk@mit.edu)

**Contributors**: Shane McQuarrie (shanemcq@utexas.edu)

Consider the (possibly nonlinear) differential equation `x'(t) = f(t,x(t))` where `x(t)` is a vector-valued function.
The `operator_inference` package provides tools for constructing a reduced-order model that is linear or quadratic in `x`, possibly with a constant term `c`, and with optional control inputs `u(t)`.

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
>>> linear_c = OpInf.Model('Lc', inp=False)

# Fit the model by solving for the operators A and c.
>>> linear_c.fit(r, k, xdot, xhat)

# Simulate the learned model for 10 timesteps of length .01.
>>> xr, n_steps = linear_c.predict(init=xhat[:,0], n_timesteps=10, dt=.01)

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
the model with the following options:
- `"L"`:  a linear model, `x'(t) = Ax(t)`.
- `"Lc"`:  a linear model with a constant, `x'(t) = Ax(t) + c`.
- `"LQ"`:  a linear model, `x'(t) = Ax(t) + Fx^2(t)`.
- `"LQc"`:  a linear model with a constant, `x'(t) = Ax(t) + Fx^2(t) + c`.
- `"Q"`:  a strictly quadratic model, `x'(t) = Fx^2(t)`.
- `"Qc"`:  a quadratic model with a constant, `x'(t) = Fx^2(t) + c`.

The `inp` argument is a boolean (`True` or `False`) denoting whether or not there is an additive input term of the form `BU`.

The script `opinf_demo.py` demonstrates the use of the operator inference model on data generated from the heat equation.
See [@mythesis] for the problem setup.

#### Methods

- `Model.fit(r, reg, xdot, xhat, u=None)`: Compute the operators of the reduced-order model that best fit the data by solving the regularized least
    squares problems `||D o - r||_2^2 + k||P o||_2^2`.

- `predict(init, n_timesteps, dt, u = None)`: Simulate the learned model with an explicit Runge-Kutta scheme tailored to the structure of the model.

- `get_residual()`: Return the residuals of the least squares problem `||D O^T - X'^T||_F^2` and `||O^T||_F^2`.

- `get_operators()`: Return each of the learned operators.

- `relative_error(predicted_data, true_data, thresh=1e-10)`: Compute the relative error between predicted data and true data, i.e., |`true_data` - `predicted_data`| / |`true_data`|. Computes absolute error (numerator only) if `|true_data|` < `thresh`.


## `opinf_helper.py`

Import the helper script with the following line.

```python
from operator_inference import opinf_helper
```

#### Functions

This file contains routines that are used within `OpInf.Model.fit()`.

- `normal_equations(D, r, k, num)`: Solve the normal equations corresponding to the regularized ordinary least squares problem `min ||D o - r||_2^2 + k||P o||_2^2`.

-  `get_x_sq(X)`: Compute squared snapshot data as in [@ben].

-  `F2H(F)`: Convert quadratic operator `F` to symmetric quadratic operator `H` for simulating the learned system.

## `integration_helpers.py`

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
- `x_next ((r,) ndarray)`: The next (reduced-dimension) state.
