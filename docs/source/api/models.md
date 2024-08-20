# `opinf.models`

```{eval-rst}
.. automodule:: opinf.models

.. currentmodule:: opinf.models

**Nonparametric Models**

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    ContinuousModel
    DiscreteModel

**Parametric Models**

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    ParametricContinuousModel
    ParametricDiscreteModel
    InterpolatedContinuousModel
    InterpolatedDiscreteModel
```

:::{admonition} Overview
:class: note
Model classes represent a set equations describing the dynamics of the model state.
The user specifies the structure of the dynamics by providing a list of [operators](opinf.operators) to the constructor.
Model dynamics are calibrated through a least-squares regression of available state and input data.
Solvers for the least-squares problem are specified in the constructor.

```python
import opinf

# Specify the model structure through a list of operators.
model = opinf.models.ContinuousModel(
    operators=[
        opinf.operators.LinearOperator(),
        opinf.operators.InputOperator(),
    ],
    solver=opinf.lstsq.L2Solver(1e-6),
)

# Calibrate the model through Operator Inference.
model.fit(state_snapshots, state_time_derivatives, corresponding_inputs)

# Solve the model.
result = model.predict(initial_condition, time_domain, input_function)
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

## Nonparametric Models

A _nonparametric_ model is comprised exclusively of [nonparametric operators](sec-operators-nonparametric).

```{eval-rst}
.. currentmodule:: opinf.models

.. autosummary::
    :nosignatures:

    ContinuousModel
    DiscreteModel
```

Nonparametric model classes are initialized with a single argument, `operators`, that must be a list of nonparametric {mod}`opinf.operators` objects.
The right-hand side of the model is defined to be the sum of the action of the operators on the model state and the input (if present).
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

:::{tip}
The `operators` constructor argument for these classes can also be a string that indicates which type of operator to use.

| Character | {mod}`opinf.operators` class |
| :-------- | :------------------------------- |
| `'c'` | {class}`opinf.operators.ConstantOperator` |
| `'A'` | {class}`opinf.operators.LinearOperator` |
| `'H'` | {class}`opinf.operators.QuadraticOperator` |
| `'G'` | {class}`opinf.operators.CubicOperator` |
| `'B'` | {class}`opinf.operators.InputOperator` |
| `'N'` | {class}`opinf.operators.StateInputOperator` |

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

The individual operators given to the constructor may or may not have their entries set.
A model's `fit()` method uses Operator Inference to calibrate the operators without entries through a regression problem, see [Learning Operators from Data](sec-operators-calibration).
Once the model operators are calibrated, nonparametric models may use the following methods.

- `rhs()`: Compute the right-hand side of the model, i.e., $\Ophat(\qhat, \u)$.
- `jacobian()`: Construct the state Jacobian of the right-hand side of the model, i.e, $\ddqhat\Ophat(\qhat,\u)$.
- `predict()`: Solve the model with given initial conditions and/or inputs.

(sec-models-parametric)=

## Parametric Models

A _parametric model_ is a model with at least one [parametric operator](sec-operators-parametric).

Parametric models are similar to nonparametric models: they are initialized with a list of operators, use `fit()` to calibrate operator entries, and `predict()` to solve the model.
In addition, parametric models have an `evaluate()` method that returns a nonparametric model at a fixed parameter value.

```{eval-rst}
.. currentmodule:: opinf.models

.. autosummary::
   :nosignatures:

   ParametricContinuousModel
   ParametricDiscreteModel
```

### Interpolated Models

Interpolated models consist exclusively of [interpolated operators](sec-operators-interpolated).

```{eval-rst}
.. currentmodule:: opinf.models

.. autosummary::
    :nosignatures:

    InterpolatedContinuousModel
    InterpolatedDiscreteModel
```

:::{tip}
The `operators` constructor argument for these classes can also be a string that indicates which type of operator to use.

| Character | {mod}`opinf.operators` class |
| :-------- | :------------------------------- |
| `'c'` | {class}`opinf.operators.InterpolatedConstantOperator` |
| `'A'` | {class}`opinf.operators.InterpolatedLinearOperator` |
| `'H'` | {class}`opinf.operators.InterpolatedQuadraticOperator` |
| `'G'` | {class}`opinf.operators.InterpolatedCubicOperator` |
| `'B'` | {class}`opinf.operators.InterpolatedInputOperator` |
| `'N'` | {class}`opinf.operators.InterpolatedStateInputOperator` |

```python
import opinf

# Initialize the model with a list of operator objects.
model = opinf.models.InterpolatedContinuousModel(
    operators=[
        opinf.operators.InterpolatedCubicOperator(),
        opinf.operators.InterpolatedStateInputOperator(),
    ]
)

# Equivalently, initialize the model with a string.
model = opinf.models.InterpolatedContinuousModel(operators="GN")
```

:::

<!--
These classes represent models without a block sparsity structure.
Use [multilithic models](sec-models-multilithic) to encode more specific system structure.
-->

<!--
(sec-models-multilithic)=
## Nonparametric Multilithic Models

- `operators` is a list of lists of nonparametric multilithic operators
- Dimension attribute: `rs` and `r = sum(rs)`
-->

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
