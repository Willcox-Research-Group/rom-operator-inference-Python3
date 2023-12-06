# `opinf.models`

:::{eval-rst}
.. automodule:: opinf.models
:::

:::{admonition} Overview
:class: note
Model classes represent a set equations describing the dynamics of the model state.
The user specifies the structure of the dynamics by providing a list of [operators](opinf.operators_new) to the constructor.
Model dynamics are calibrated through a least-squares regression of available state and input data.

```python
import opinf

# Specify the model structure through a list of operators.
rom = opinf.models.ContinuousModel(
    operators=[
        opinf.operators.LinearOperator(),
        opinf.operators.InputOperator(),
    ]
)

# Calibrate the model through Operator Inference.
rom.fit(state_snapshots, state_time_derivatives, corresponding_inputs)

# Solve the model.
result = rom.predict(initial_condition, time_domain, input_function)
```

:::

## Types of Models

::::{margin}
:::{note}
Spatially discretizing a partial differential equation results in a continuous-time ODE model; discretizing a continuous-time ODE model in time yields a discrete-time model.
:::
::::

The models defined in this module can be classified in a few ways.

1. **Continuous vs Discrete:** _Continuous-time_ models are for systems of ordinary differential equations (ODEs), and _discrete-time_ models are for discrete dynamical systems.
2. **Parametric vs Nonparametric:** In a _parametric_ model, the dynamics depend on one or more external parameters; a _nonparametric_ model has no external parameter dependence.

<!-- 2. **Monolithic vs Multilithic:** A _monolithic_ model defines a single set of equations for the state variable, while a _multilithic_ model defines specific equations for individual parts of the state variable. -->

<!--
:::{dropdown} Multilithic System Example: Linear Hamiltonian System
:class: tip

Consider the system of ODEs given by

$$
\begin{align*}
    \frac{\text{d}}{\text{d}t}\q(t)
    = \frac{\text{d}}{\text{d}t}\left[\begin{array}{c}
    \q_{0}(t) \\ \q_{1}(t)
    \end{array}\right]
    = \left[\begin{array}{cc}
    \mathbf{0} & \A_{0,1} \\ \A_{1,0} & \mathbf{0}
    \end{array}\right]\left[\begin{array}{c}
    \q_{0}(t) \\ \q_{1}(t)
    \end{array}\right]
    = \A\q(t),
\end{align*}
$$

where $\q_{0}(t),\q_{1}(t)\in\RR^{n/2}$, $\A_{0,1},\A_{1,0}\in\RR^{n/2\times n/2}$, and

$$
\begin{align*}
    \q(t) = \left[\begin{array}{c}
    \q_{0}(t) \\ \q_{1}(t)
    \end{array}\right]\in\RR^{n},
    \qquad
    \A = \left[\begin{array}{cc}
    \mathbf{0} & \A_{0,1} \\ \A_{1,0} & \mathbf{0}
    \end{array}\right]\in\RR^{n\times n}.
\end{align*}
$$

If a monolithic dimensionality reduction technique is used, the structure of the system is lost:
approximating $\q(t) \approx \Vr\qhat$ where $\qhat(t)\in\RR^{r}$ and $\Vr\in\RR^{n\times r}$ has orthogonal columns,
Galerkin projection leads to the model

$$
\begin{align*}
    \frac{\text{d}}{\text{d}t}\qhat(t)
    = \Ahat\qhat(t),
    \qquad
    \Ahat = \Vr\trp\A\Vr.
\end{align*}
$$

In most cases, $\Ahat$ will be dense and not have the block structure of $\A$.
Alternatively, consider the multilithic approximation $\q_{0}(t) \approx \mathbf{V}_{0}\qhat_{0}$ and $\q_{1}(t) \approx \mathbf{V}_{1}\qhat_{1}$ where $\qhat_{0},\qhat_{1}\in\RR^{r/2}$ and $\mathbf{V}_{0},\mathbf{V}_{1}\in\RR^{n/2\times r/2}$, i.e.,

$$
\begin{align*}
    \q(t)
    = \left[\begin{array}{c}
    \q_{0}(t) \\ \q_{1}(t)
    \end{array}\right]
    \approx
    \left[\begin{array}{cc}
    \mathbf{V}_{0} & \mathbf{0} \\ \mathbf{0} & \mathbf{V}_{1}
    \end{array}\right]
    \left[\begin{array}{c}
    \qhat_{0}(t) \\ \qhat_{1}(t)
    \end{array}\right].
\end{align*}
$$

In this case, Galerkin projection produces a Model
$\ddt\qhat(t) = \Ahat\qhat(t)$ as before, but now with

$$
\begin{align*}
    \Ahat
    = \left[\begin{array}{cc}
    \mathbf{V}_{0} & \mathbf{0} \\ \mathbf{0} & \mathbf{V}_{1}
    \end{array}\right]\trp
    \left[\begin{array}{cc}
    \mathbf{0} & \A_{0,1} \\ \A_{1,0} & \mathbf{0}
    \end{array}\right]
    \left[\begin{array}{cc}
    \mathbf{V}_{0} & \mathbf{0} \\ \mathbf{0} & \mathbf{V}_{1}
    \end{array}\right]
    =
    \left[\begin{array}{cc}
    \mathbf{0} & \mathbf{V}_{0}\trp\A_{0,1}\mathbf{V}_{1}
    \\
    \mathbf{V}_{1}\trp\A_{1,0}\mathbf{V}_{0} & \mathbf{0}
    \end{array}\right],
\end{align*}
$$

which has the same block structure as $\A$.
:::
-->

## Nonparametric Models

A _nonparametric_ model is comprised exclusively of [nonparametric operators](sec-operators-nonparametric) (see [parametric models](sec-models-parametric)).

### API Summary

#### Initialization and Model Structure

Nonparametric model classes are initialized with a single argument, `operators`, that must be a list of nonparametric {mod}`opinf.operators_new` objects.
The right-hand side of the model is defined to be the sum of the action of the operators on the model state and the (optional) input.
For example, a {class}`ContinuousModel` represents a system of ODEs

$$
\begin{align*}
   \ddt\qhat(t)
   = \fhat(\qhat,\u)
   = \sum_{\ell=1}^{n_\textrm{terms}}
   \Ophat_{\ell}(\qhat(t),\u(t))
\end{align*}
$$

where each $\Ophat_{\ell}$ is a nonparametric operator.

#### Model Calibration

The individual operators given to the constructor may or may not have their entries set.
The `fit()` method uses Operator Inference to calibrate the operators without entries through a regression problem, see [Calibrating Operator Entries](sec-operators-calibration).

#### Model Evaluation

Once the model operators are calibrated, nonparametric models may use the following methods.

- `rhs()`: Compute the right-hand side of the model, i.e., $\Ophat(\qhat, \u)$.
- `jacobian()`: Construct the state Jacobian of the right-hand side of the model, i.e, $\ddqhat\Ophat(\qhat,\u)$.
- `predict()`: Solve the model with given initial conditions and/or inputs.

#### Object Persistence

Models can be saved to disk in [HDF5 format](https://www.h5py.org/) via the `save()` method.
Every model has a class method `load()` for loading an operator from the HDF5 file previously produced by `save()`.

### Nonparametric Model Classes

<!--
These classes represent models without a block sparsity structure.
Use [multilithic models](sec-models-multilithic) to encode more specific system structure.
-->

```{eval-rst}
.. currentmodule:: opinf.models

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    ContinuousModel
    DiscreteModel
    SteadyModel
```

:::{tip}
The `operators` constructor argument for these classes can also be a string that indicates which type of operator to use.

| Character | {mod}`opinf.operators_new` class |
| :-------- | :------------------------------- |
| `'c'` | {class}`opinf.operators_new.ConstantOperator` |
| `'A'` | {class}`opinf.operators_new.LinearOperator` |
| `'H'` | {class}`opinf.operators_new.QuadraticOperator` |
| `'G'` | {class}`opinf.operators_new.CubicOperator` |
| `'B'` | {class}`opinf.operators_new.InputOperator` |
| `'N'` | {class}`opinf.operators_new.StateInputOperator` |

```python
import opinf

# Initialize the model with a list of operator objects.
model = opinf.models.DiscreteModel(
    operators=[
        opinf.operators.QuadraticOperator(),
        opinf.operators.InputOperator(),
    ]
)

# Equivalently, initialize the model with a string.
model = opinf.models.DiscreteModel(operators="HB")
```

:::

<!--
(sec-models-multilithic)=
## Nonparametric Multilithic Models

- `operators` is a list of lists of nonparametric multilithic operators
- Dimension attribute: `rs` and `r = sum(rs)`
-->

(sec-models-parametric)=
## Parametric Models

A _parametric model_ is a model with at least one [parametric operator](sec-operators-parametric).

:::{admonition} TODO

- `__call__()`/`evaluate()` maps parameter values to a nonparametric model object.
- `operators` can be nonparametric or parametric operators.
- `fit()` takes in parameter values, lists of snapshots, lists of LHS, and lists of inputs.
- `predict()` takes in a parameter value, then whatever else.

:::
