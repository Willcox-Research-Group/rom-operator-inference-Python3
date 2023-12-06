# `opinf.models`

:::{eval-rst}
.. automodule:: opinf.models
:::

This page discusses the different kinds of dynamical systems models, how to specify model structure, and using Operator Inference to calibrate the model to data.

:::{admonition} Overview
:class: note
Every model has describes a set equations describing the dynamics of the model state.
The user specifies the structure of the dynamics by providing a list of [operators](opinf.operators_new) to a model class.
Operators are calibrated through a least-squares regression of available state and input data.

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

The models defined in this module can be classified in two ways.

1. **Continuous vs Discrete:** _Continuous-time_ models are for systems of ordinary differential equations (or spatially discretized partial differential equations), and _discrete-time_ models are for discrete dynamical systems.
2. **Monolithic vs Multilithic:** A _monolithic_ model defines a single set of equations for the state variable, while a _multilithic_ model defines specific equations for individual parts of the state variable.
3. **Parametric vs Nonparametric:** In a _parametric_ model, the dynamics depend on one or more external parameters; a _nonparametric_ model has no external parameter dependence.

### Continuous-time Models

Continuous-time models are for systems of ordinary differential equations (ODEs), for example those resulting from spatially discretizing partial differential equations.
The state $\q(t)\in\RR^{n}$ and the input $\u(t)\in\RR^{m}$ are time-dependent.

::::{tab-set}
:::{tab-item} Nonparametric Problem
$$
\frac{\text{d}}{\text{d}t}\q(t)
= \mathbf{F}(\q(t), \u(t))
$$
:::

:::{tab-item} Parametric Problem
$$
\frac{\text{d}}{\text{d}t}\q(t;\mu)
= \mathbf{F}(\q(t;\mu), \u(t); \mu)
$$
:::
::::

The reduced-order dynamics are a system of ODEs for the reduced state $\qhat(t)$.
In the multilithic case, the reduced state is decomposed into chunks,

$$
\qhat(t)
= \left[\begin{array}{c}
\qhat_{0}(t)
\\ \vdots \\
\qhat_{d-1}(t)
\end{array}\right],
$$

and a set of ODEs is defined for each $\qhat_{\ell}(t)$, $\ell=0,\ldots,d-1$.

| Model Type | `opinf` Class | Reduced-order Dynamics |
| :------- | :------------ | :--------------------- |
| Monolithic Nonparametric  | {class}`ContinuousModel` | $\frac{\text{d}}{\text{d}t}\qhat(t) = \widehat{\mathbf{F}}(\qhat(t), \u(t))$ |
| Monolithic Parametric     | **`ContinuousPModel`** | $\frac{\text{d}}{\text{d}t}\qhat(t;\mu) = \widehat{\mathbf{F}}(\qhat(t;\mu), \u(t); \mu)$ |
| Multilithic Nonparametric | **`ContinuousModelMulti`** | $\frac{\text{d}}{\text{d}t}\qhat_{\ell}(t) = \widehat{\mathbf{F}}_{\ell}(\qhat(t), \u(t)),\quad\ell=1,\ldots,d-1$ |
| Multilithic Parametric | **`ContinuousPModelMulti`** | $\frac{\text{d}}{\text{d}t}\qhat_{\ell}(t;\mu) = \widehat{\mathbf{F}}_{\ell}(\qhat(t;\mu), \u(t); \mu),\quad\ell=1,\ldots,d-1$ |

:::{dropdown} Multilithic System Example: Linear Hamiltonian System
Consider the system of ODEs given by

$$
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
$$

where $\q_{0}(t),\q_{1}(t)\in\RR^{n/2}$, $\A_{0,1},\A_{1,0}\in\RR^{n/2\times n/2}$, and

$$
\q(t) = \left[\begin{array}{c}
\q_{0}(t) \\ \q_{1}(t)
\end{array}\right]\in\RR^{n},
\qquad
\A = \left[\begin{array}{cc}
\mathbf{0} & \A_{0,1} \\ \A_{1,0} & \mathbf{0}
\end{array}\right]\in\RR^{n\times n}.
$$

If a monolithic dimensionality reduction technique is used, the structure of the system is lost:
approximating $\q(t) \approx \Vr\qhat$ where $\qhat(t)\in\RR^{r}$ and $\Vr\in\RR^{n\times r}$ has orthogonal columns,
Galerkin projection leads to the model

$$
\frac{\text{d}}{\text{d}t}\qhat(t)
= \Ahat\qhat(t),
\qquad
\Ahat = \Vr\trp\A\Vr.
$$

In most cases, $\Ahat$ will be dense and not have the block structure of $\A$.
Alternatively, consider the multilithic approximation $\q_{0}(t) \approx \mathbf{V}_{0}\qhat_{0}$ and $\q_{1}(t) \approx \mathbf{V}_{1}\qhat_{1}$ where $\qhat_{0},\qhat_{1}\in\RR^{r/2}$ and $\mathbf{V}_{0},\mathbf{V}_{1}\in\RR^{n/2\times r/2}$, i.e.,

$$
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
$$

In this case, Galerkin projection produces a Model
$
\frac{\text{d}}{\text{d}t}\qhat(t)
= \Ahat\qhat(t)
$ as before, but now with

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

### Discrete-time models

Discrete-time models are for discrete dynamical systems, where values of the state $\q\in\RR^{n}$ and the input $\u\in\RR^{m}$ are given at discrete iterates, denoted with the superscripted $\q^{(j)}$, $\u^{(j)}$.
The full-order model is an updated formula for $\q^{(j+1)}$ in terms of $\q^{(j)}$ and $\u^{(j)}$.

::::{tab-set}
:::{tab-item} Nonparametric Problem
$$
\q^{(j+1)}
= \mathbf{F}(\q^{(j)}, \u^{(j)})
$$
:::

:::{tab-item} Parametric Problem
$$
\q^{(j+1)}(\mu)
= \mathbf{F}(\q^{(j)}(\mu), \u^{(j)}; \mu)
$$
:::
::::

The reduced-order dynamics are a discrete dynamical system for the reduced state $\qhat$.
In the multilithic case, the reduced state is decomposed as $\qhat = [~\qhat\trp~~\qhat_{1}\trp~~\cdots~~\qhat_{d-1}\trp~]\trp$ and an update formula is defined for each $\qhat_{\ell}$, $\ell=0,\ldots,d-1$.

| Model Type | `opinf` Class | Reduced-order Dynamics |
| :------- | :------------ | :--------------------- |
| Monolithic Nonparametric | [**`DiscreteModel`**](opinf.DiscreteModel) | $\qhat^{(j+1)} = \widehat{\mathbf{F}}(\qhat^{(j)}, \u^{(j)})$ |
| Monolithic Parametric    | **`DiscretePModel`** | $\qhat^{(j+1)}(\mu) = \widehat{\mathbf{F}}(\qhat^{(j)}(\mu), \u^{(j)}; \mu)$ |
| Multilithic Nonparametric | **`DiscreteModelMulti`** | $\qhat_{\ell}^{(j+1)} = \widehat{\mathbf{F}}_{\ell}(\qhat^{(j)}, \u^{(j)}),\quad\ell=1,\ldots,d-1$ |
| Multilithic Parametric | **`DiscretePModelMulti`** | $\qhat_{\ell}(\mu)^{(j+1)} = \widehat{\mathbf{F}}_{\ell}(\qhat(\mu)^{(j)}, \u^{(j)}; \mu),\quad\ell=1,\ldots,d-1$ |

<!-- TODO: Steady-state Problems -->

## Model Classes

::::{margin}
:::{tip}
The API for these classes adopts some principles from the [scikit-learn](https://scikit-learn.org/stable/index.html) [API](https://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects): there are `fit()` and `predict()` methods, hyperparameters are set in the constructor, estimated attributes end with underscore, and so on.
:::
::::

- `operators`
- Dimensions: `r`, `m`
- `fit()`
- `predict()`
- `save()` and `load()`

### Nonparametric Monolithic Models

- `operators` is a single list of nonparametric monolithic operators (or strings for shorthand)
- Dimension attribute: `r`
- Shortcut properties for accessing operators: `c_`, `A_`, `H_`, `G_`, `B_`, `N_`.

### Nonparametric Multilithic Models

- `operators` is a list of lists of nonparametric multilithic operators
- Dimension attribute: `rs` and `r = sum(rs)`

### Parametric Models

- `__call__()` maps parameter values to a nonparametric model object.
- `operators` can be nonparametric or parametric operators.
- `fit()` takes in parameter values, lists of snapshots, lists of LHS, and lists of inputs.
- `predict()` takes in a parameter value, then whatever else.

## Nonparametric Models

:::{adomonition} TODO

- `operators`
- Dimensions: `r`, `m`
- `fit()`
- `predict()`
- `save()` and `load()`
:::

### Nonparametric Model Classes

```{eval-rst}
.. currentmodule:: opinf.models

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    ContinuousModel
    DiscreteModel
    SteadyModel
```

## Parametric Models
