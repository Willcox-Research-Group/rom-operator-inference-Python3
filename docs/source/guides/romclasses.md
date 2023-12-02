# Reduced-order Model Design

The `opinf` module provides several classes for representing reduced-order models (ROMs).
This page discusses the different kinds of ROMs and how to build a ROM by specifying a model structure and using Operator Inference to calibrate the model to data.

:::{admonition} Overview
:class: note
Every ROM has two main components: a low-dimensional approximation of the full state (see [Dimensionality Reduction](sec-guide-dimensionality)), and a set equations describing the dynamics of the reduced state.
The user specifies the structure of the reduced dynamics by providing a list of [operators](sec-operator-classes) to a ROM class.
Operators are calibrated through a least-squares regression of available state and input data.

```python
import opinf

# Specify the ROM structure through a list of operators.
rom = opinf.ContinuousROM(
    basis=basis,
    operators=[
        opinf.operators.LinearOperator(),
        opinf.operators.InputOperator(),
    ]
)

# Calibrate the operator entries through Operator Inference.
rom.fit(state_snapshots, state_time_derivatives, corresponding_inputs)

# Compute solutions of the ROM.
result = rom.predict(initial_condition, time_domain, input_function)
```

:::

:::{admonition} Notation
:class: attention
On this page, we use $\mathbf{F}$ to denote the function governing the dynamics of the full-order state $\q\in\RR^{n}$.
Likewise, the function $\widehat{\mathbf{F}}$ determines the dynamics of the reduced-order state $\qhat\in\RR^{r}$, where $r \ll n$.
Inputs are written as $\u\in\RR^{m}$.
For parametric problems, we use $\mu \in \RR^{p}$ to denote the free parameters.
:::

## Types of Reduced-order Models

ROM classes are included in the main [**opinf**](sec-main) namespace.
The type of ROM class to use depends on three factors:

1. **Continuous vs Discrete:** _Continuous-time_ ROMs are for systems of ordinary differential equations (or spatially discretized partial differential equations), and _discrete-time_ ROMs are for discrete dynamical systems.
2. **Monolithic vs Multilithic:** A _monolithic_ ROM defines a single set of equations for the reduced state variable, while a _multilithic_ ROM defines specific equations for individual parts of the reduced state variable.
<!-- See [Dimensionality Reduction / Multilithic](TODO). -->
3. **Parametric vs Nonparametric:** In a _parametric_ ROM, the dynamics depend on one or more external parameters; a _nonparametric_ ROM has no external parameter dependence.

### Continuous-time ROMs

Continuous-time ROMs are for systems of ordinary differential equations (ODEs), for example those resulting from spatially discretizing partial differential equations.
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

| ROM Type | `opinf` Class | Reduced-order Dynamics |
| :------- | :------------ | :--------------------- |
| Monolithic Nonparametric  | [**`ContinuousROM`**](opinf.ContinuousROM) | $\frac{\text{d}}{\text{d}t}\qhat(t) = \widehat{\mathbf{F}}(\qhat(t), \u(t))$ |
| Monolithic Parametric     | **`ContinuousPROM`** | $\frac{\text{d}}{\text{d}t}\qhat(t;\mu) = \widehat{\mathbf{F}}(\qhat(t;\mu), \u(t); \mu)$ |
| Multilithic Nonparametric | **`ContinuousROMMulti`** | $\frac{\text{d}}{\text{d}t}\qhat_{\ell}(t) = \widehat{\mathbf{F}}_{\ell}(\qhat(t), \u(t)),\quad\ell=1,\ldots,d-1$ |
| Multilithic Parametric | **`ContinuousPROMMulti`** | $\frac{\text{d}}{\text{d}t}\qhat_{\ell}(t;\mu) = \widehat{\mathbf{F}}_{\ell}(\qhat(t;\mu), \u(t); \mu),\quad\ell=1,\ldots,d-1$ |

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
Galerkin projection leads to the ROM

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

In this case, Galerkin projection produces a ROM
$
\frac{\text{d}}{\text{d}t}\qhat(t)
= \Ahat\qhat(t)
$ as before, but now with

$$
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
$$

which has the same block structure as $\A$.
:::

### Discrete-time ROMs

Discrete-time ROMs are for discrete dynamical systems, where values of the state $\q\in\RR^{n}$ and the input $\u\in\RR^{m}$ are given at discrete iterates, denoted with the superscripted $\q^{(j)}$, $\u^{(j)}$.
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

| ROM Type | `opinf` Class | Reduced-order Dynamics |
| :------- | :------------ | :--------------------- |
| Monolithic Nonparametric | [**`DiscreteROM`**](opinf.DiscreteROM) | $\qhat^{(j+1)} = \widehat{\mathbf{F}}(\qhat^{(j)}, \u^{(j)})$ |
| Monolithic Parametric    | **`DiscretePROM`** | $\qhat^{(j+1)}(\mu) = \widehat{\mathbf{F}}(\qhat^{(j)}(\mu), \u^{(j)}; \mu)$ |
| Multilithic Nonparametric | **`DiscreteROMMulti`** | $\qhat_{\ell}^{(j+1)} = \widehat{\mathbf{F}}_{\ell}(\qhat^{(j)}, \u^{(j)}),\quad\ell=1,\ldots,d-1$ |
| Multilithic Parametric | **`DiscretePROMMulti`** | $\qhat_{\ell}(\mu)^{(j+1)} = \widehat{\mathbf{F}}_{\ell}(\qhat(\mu)^{(j)}, \u^{(j)}; \mu),\quad\ell=1,\ldots,d-1$ |

<!-- TODO: Steady-state Problems -->

(sec-operator-classes)=
## Operator Classes

All ROM classes are initialized with a list of `operators` that define the structure of the reduced-order model dynamics, i.e., the function $\widehat{\mathbf{F}}$.
These are defined in the {class}`opinf.operators` submodule.

### Nonparametric Monolithic Operators

:::{warning}
This page is under construction.
:::

- `datablock()` and `column_dimension()` methods

## ROM Classes

::::{margin}
:::{tip}
The API for these classes adopts some principles from the [scikit-learn](https://scikit-learn.org/stable/index.html) [API](https://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects): there are `fit()` and `predict()` methods, hyperparameters are set in the constructor, estimated attributes end with underscore, and so on.
:::
::::

- `basis`, `operators`
- Dimensions: `n`, `m`
- `compress()` and `decompress()`
- `fit()`
- `predict()`
- `save()` and `load()`

### Nonparametric Monolithic ROMs

- `basis` can be any basis object, ndarray (`LinearBasis`), or `None`
- `operators` is a single list of nonparametric monolithic operators (or strings for shorthand)
- Dimension attribute: `r`
- Shortcut properties for accessing operators: `c_`, `A_`, `H_`, `G_`, `B_`, `N_`.

### Nonparametric Multilithic ROMs

- `basis` **must** be multilithic
- `operators` is a list of lists of nonparametric multilithic operators
- Dimension attribute: `rs` and `r = sum(rs)`

### Parametric ROMs

- `__call__()` maps parameter values to a nonparametric ROM object.
- `operators` can be nonparametric or parametric operators.
- `fit()` takes in parameter values, lists of snapshots, lists of LHS, and lists of inputs.
- `predict()` takes in a parameter value, then whatever else.

## OLD MATERIAL

In the following discussion we begin with the non-parametric ROM classes; parametric classes are considered in [Parametric ROMs](subsec-parametric-roms).

(subsec-romclass-constructor)=

### Defining Model Structure

ROM classes are instantiated with a single argument, `modelform`, which is a string denoting the structure of the right-hand side function $\widehat{\mathbf{F}}$.
Each character in the string corresponds to a single term in the model.

| Character | Name | Continuous Term | Discrete Term |
| :-------- | :--- | :-------------- | :------------ |
| `c` | Constant | $\chat$ | $\chat$ |
| `A` | Linear | $\Ahat\qhat(t)$ | $\Ahat\qhat_{j}$ |
| `H` | Quadratic | $\Hhat[\qhat(t) \otimes \qhat(t)]$ | $\Hhat[\qhat_{j} \otimes \qhat_{j}]$ |
| `G` | Cubic | $\widehat{\mathbf{G}}[\qhat(t) \otimes \qhat(t) \otimes \qhat(t)]$ | $\widehat{\mathbf{G}}[\qhat_{j} \otimes \qhat_{j} \otimes \qhat_{j}]$ |
| `B` | Input | $\Bhat\u(t)$ | $\Bhat\u_{j}$ |

<!-- | `C` | Output | $\mathbf{y}(t)=\widehat{C}\qhat(t)$ | $\mathbf{y}_{k}=\hat{C}\qhat_{k}$ | -->

The full model form is specified as a single string.

| `modelform` | Continuous ROM Structure | Discrete ROM Structure |
| :---------- | :----------------------- | ---------------------- |
|  `"A"`      | $\frac{\text{d}}{\text{d}t}\qhat(t) = \Ahat\qhat(t)$ | $\qhat_{j+1} = \Ahat\qhat_{j}$ |
|  `"cA"`     | $\frac{\text{d}}{\text{d}t}\qhat(t) = \chat + \Ahat\qhat(t)$ | $\qhat_{j+1} = \chat + \Ahat\qhat_{j}$ |
|  `"AB"`   | $\frac{\text{d}}{\text{d}t}\qhat(t) = \Ahat\qhat(t) + \Bhat\u(t)$ | $\qhat_{j+1} = \Ahat\qhat_{j} + \Bhat\u_{j}$ |
|  `"HB"`     | $\frac{\text{d}}{\text{d}t}\qhat(t) = \Hhat[\qhat(t)\otimes\qhat(t)] + \Bhat\u(t)$ | $\qhat_{j+1} = \Hhat[\qhat_{j}\otimes\qhat_{j}] + \Bhat\u_{j}$ |

<!-- | Steady ROM Structure |
| $\widehat{\mathbf{g}} = \Ahat\qhat$ |
| $\widehat{\mathbf{g}} = \chat + \Ahat\qhat$ |
| $\widehat{\mathbf{g}} = \Ahat\qhat + \Bhat\u$ |
| $\widehat{\mathbf{g}} = \Hhat[\qhat\otimes\qhat] + \Bhat\u$ | -->

## ROM Attributes

All ROM classes have the following attributes.

### Dimensions

These attributes are integers that are initially set to `None`, then inferred from the training inputs during `fit()`.
They cannot be altered manually after calling `fit()`.

| Attribute | Description |
| :-------- | :---------- |
| `n` | Dimension of the high-dimensional training data $\q$. |
| `r` | Dimension of the reduced-order model state $\qhat$. |
| `m` | Dimension of the input $\u$. |

If there is no input (meaning `modelform` does not contain `'B'`), then `m` is set to 0.

### Basis

The `basis` attribute is the mapping between the $n$-dimensional state space of the full-order data and the smaller $r$-dimensional state space of the reduced-order model (e.g., POD basis).
This is the first input to the `fit()` method.
See the [Dimensionality Reduction](sec-guide-dimensionality) guide for details.

### Operators

These attributes are the operators corresponding to the learned parts of the reduced-order model.
The classes are defined in the [**operators**](opinf.operators) submodule.

<!-- TODO: Operator Class with links to API docs -->

| Attribute | Evaluation mapping | Jacobian mapping |
| :-------- | :----------------- | :--------------- |
| `c_` | $\qhat \mapsto \chat$ | $\qhat \mapsto \mathbf{0}$ |
| `A_` | $\qhat \mapsto \Ahat\qhat$ | $\qhat \mapsto \Ahat$ |
| `H_` | $\qhat \mapsto \Hhat[\qhat\otimes\qhat]$ | $\qhat \mapsto \Hhat[(\I\otimes\qhat) + (\qhat\otimes\I)]$ |
| `G_` | $\qhat \mapsto \widehat{\mathbf{G}}[\qhat\otimes\qhat\otimes\qhat]$ | $\qhat \mapsto \widehat{\mathbf{G}}[(\I\otimes\qhat\otimes\qhat) + \cdots + (\qhat\otimes\qhat\otimes\I)]$ |
| `B_` | $\u \mapsto \Bhat\u$ | $\u \mapsto \Bhat$ |

All operators are set to `None` initially and only changed by `fit()` if the operator is included in the prescribed `modelform` (e.g., if `modelform="AHG"`, then `c_` and `B_` are always `None`).
<!-- Note that Jacobian mapping of the input operation _with respect to the state_ is zero. -->

#### Operator Attributes

The discrete representation of the operator is a NumPy array stored as the `entries` attribute.
This array can also be accessed by slicing the operator object directly.

```python
>>> import numpy as np
>>> import opinf

>>> arr = np.arange(16).reshape(4, 4)
>>> operator = opinf.operators.LinearOperator(arr)

>>> operator.entries
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])

>>> operator[:]
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])

>>> operator.shape
(4, 4)
```

In practice, with a ROM object `rom`, the entries of (e.g.) the linear state matrix $\Ahat$ are accessed with `rom.A_[:]` or `rom.A_.entries`.

#### Operator Methods

The `evaluate()` method computes the action of the operator on the (low-dimensional) state or input.

```python
>>> q_ = np.arange(4)
>>> operator.evaluate(q_)
array([14, 38, 62, 86])

# Equivalent calculation with the raw NumPy array.
>>> arr @ q_
array([14, 38, 62, 86])
```

::::{note}
Nothing special is happening under the hood for constant and linear operators, but the quadratic and cubic operators use a compressed representation to efficiently compute the operator action on the quadratic or cubic Kronecker products $\qhat\otimes\qhat$ or $\qhat\otimes\qhat\otimes\qhat$.

```python
>>> r = 5
>>> arr2 = np.random.random((r, r**2))
>>> quadratic_operator = opinf.operators.QuadraticOperator(arr2)
>>> q_ = np.random.random(r)

>>> np.allclose(quadratic_operator.evaluate(q_), arr2 @ (np.kron(q_, q_)))
True

>>> quadratic_operator.shape
(5, 15)
```

The shape of the quadratic operator `entries` has been reduced from $r \times r^{2}$ to $r \times \frac{r(r + 1)}{2}$ to exploit the structure of the Kronecker products.

:::{dropdown} Details
Let $\qhat = [~\hat{q}_{1}~\cdots~\hat{q}_{r}~]\trp\in\RR^{r}$ and consider the Kronecker product

$$
\qhat\otimes\qhat
= \left[\begin{array}{c}
    \hat{q}_{1}\qhat \\
    \hat{q}_{2}\qhat \\
    \vdots \\
    \hat{q}_{r}\qhat \\
\end{array}\right]
= \left[\begin{array}{c}
    \hat{q}_{1}^{2} \\
    \hat{q}_{1}\hat{q}_{2} \\
    \vdots \\
    \hat{q}_{1}\hat{q}_{r} \\
    \hat{q}_{1}\hat{q}_{2} \\
    \hat{q}_{2}^{2} \\
    \vdots \\
    \hat{q}_{2}\hat{q}_{r} \\
    \vdots \\
    \hat{q}_{r}^{2}
\end{array}\right]
\in \RR^{r^{2}}.
$$

Note that $\qhat\otimes\qhat$ has some redundant entries, for example $\hat{q}_{1}\hat{q}_{2}$ shows up twice. In fact, $\hat{q}_{i}\hat{q}_{j}$ occurs twice for every choice of $i \neq j$.
Thus, $\qhat\otimes\qhat$ can be represented with only $r (r + 1)/2$ degrees of freedom as, for instance,

$$
\left[\begin{array}{c}
    \qhat^{(1)} \\
    \qhat^{(2)} \\
    \vdots \\
    \qhat^{(r)}
\end{array}\right]
= \left[\begin{array}{c}
    \hat{q}_{1}^{2} \\
    \hat{q}_{1}\hat{q}_{2} \\
    \hat{q}_{2}^{2} \\
    \hat{q}_{1}\hat{q}_{3} \\
    \hat{q}_{2}\hat{q}_{3} \\
    \hat{q}_{3}^{2} \\
    \vdots \\
    \hat{q}_{r}^{2}
\end{array}\right]
\in \RR^{r(r + 1)/2},
\qquad
\qhat^{(i)}
= \hat{q}_{i}\left[\begin{array}{c}
    \hat{q}_{1} \\ \vdots \\ \hat{q}_{i}
\end{array}\right]\in\RR^{i}.
$$

This is the same as filling a vector with the upper-triangular entries of the outer product $\qhat\qhat\trp$.
The dimension $r (r + 1)/2$ arises because we choose 2 of r entries _without replacement_, i.e., this is a [multiset coefficient](https://en.wikipedia.org/wiki/Multiset#Counting_multisets):

$$
\left(\!\!{r\choose 2}\!\!\right)
= \binom{r + 2 - 1}{2}
= \binom{r+1}{2}
= \frac{r(r+1)}{2}.
$$

:::
::::

<!-- TODO: Jacobians -->

### Summary

| Attribute | Description |
| :-------- | :---------- |
| `n` | Dimension of the high-dimensional training data $\q$. |
| `r` | Dimension of the reduced-order model state $\qhat$. |
| `m` | Dimension of the input $\u$. |
| `basis` | Mapping between the $n$-dimensional state space of the full-order data and the $r$-dimensional state space of the ROM |
| `c_` | Constant operator $\qhat \mapsto \chat$ |
| `A_` | Linear operator $\qhat \mapsto \Ahat\qhat$ |
| `H_` | Quadratic operator $\qhat \mapsto \Hhat[\qhat\otimes\qhat]$ |
| `G_` | Cubic operator $\qhat \mapsto \widehat{\mathbf{G}}[\qhat\otimes\qhat\otimes\qhat]$ |
| `B_` | Input operator $\u \mapsto \Bhat\u$ |

## ROM Methods

All ROM classes have the following methods.

### Dimensionality Reduction

The `compress()` method maps a state quantity from the high-dimensional space $\RR^{n}$ to the low-dimensional space $\RR^{r}$.
Conversely, `decompress()` maps from $\RR^{r}$ to $\RR^{n}$.
<!-- These methods are not quite inverses: the results of `decompress()` are restricted to the portion of $\RR^{n}$ that can be represented through the underlying basis. -->
These methods wrap the `compress()` and `decompress()` methods of the `basis` attribute.

### Training

::::{margin}
:::{tip}
The `fit()` method accepts `basis=None`, in which case the state arguments for training are assumed to be already reduced to an $r$-dimensional state space (e.g., $\widehat{\Q} = \Vr^{\top}\Q$ instead of $\Q$).
:::
::::

The `fit()` method sets up and solves a [least-squares regression](subsec-opinf-regression) to determine the entries of the operators $\chat$, $\Ahat$, $\Hhat$, $\widehat{\mathbf{G}}$, and/or $\Bhat$.
Common inputs are

- the basis
- state snapshot data
- left-hand side data (time derivatives)
- regularization parameters
<!-- TODO: least squares solver! -->

### Prediction

The `evaluate()` method evaluates the right-hand side of the learned reduced-order model, i.e., it is the mapping

<!-- :::{tip}
The `evaluate()` and `jacobian()` methods are useful for constructing custom solvers for the reduced-order model.
::: -->

$$
(\qhat,\u) \mapsto
\chat
- \Ahat\qhat
- \Hhat[\qhat\otimes\qhat]
- \Bhat\u.
$$

The `predict()` method solves the reduced-order model for given initial conditions and inputs.

### Model Persistence

Some ROM objects can be saved in [HDF5 format](http://docs.h5py.org/en/stable/index.html) with the `save()` method, then recovered later with the `load()` class method.
Such files store metadata for the model class and structure, the reduced-order model operators, and (optionally) the basis.

```python
>>> import opinf

# Assume we have a ROM as an opinf.ContinuousROM object, called `rom`.

>>> rom.save("trained_rom.h5")                                # Save a trained model.
>>> rom2 = opinf.ContinuousROM.load("trained_rom.h5")    # Load a model from file.
>>> rom == rom2
True
```

For ROM classes without a `save()`/`load()` implementation, ROM objects can usually be saved locally via the `pickle` or `joblib` libraries, which is [the approach taken by scikit-learn (`sklearn`)](https://scikit-learn.org/stable/model_persistence.html).

:::{tip}
Files in HDF5 format are slightly more transparent than pickled binaries in the sense that individual parts of the file can be extracted manually without loading the entire file.
Several programming languages support HDF5 format (MATLAB, C, C++, etc.), making HDF5 a good candidate for sharing ROM data with other programs.
:::

### Summary

| Method | Description |
| :----- | :---------- |
| `compress()` | Map high-dimensional states to their low-dimensional coordinates |
| `decompress()` | Use low-dimensional coordinates to construct a high-dimensional state |
| `fit()` | Use training data to infer the operators defining the ROM |
| `evaluate()` | Evaluate the reduced-order model for a given state / input |
| `predict()` | Solve the reduced-order model |
| `save()` | Save the ROM data to an HDF5 file |
| `load()` | Load a ROM from an HDF5 file |

(sec-continuous)=

## Continuous-time ROMs

A continuous-time ROM is a surrogate for a system of ordinary differential equations, written generally as

$$
\frac{\text{d}}{\text{d}t}\qhat(t;\mu)
= \widehat{\mathbf{F}}(t, \qhat(t;\mu), \u(t); \mu).
$$

The following ROM classes target the continuous-time setting.

- [**ContinuousROM**](opinf.ContinuousROM) (nonparametric)
- [**InterpolatedContinuousROM**](opinf.InterpolatedContinuousROM) (parametric via interpolation)

### Time Derivative Data

The OpInf regression problem for the continuous-time setting is {eq}`eq:opinf-lstsq-residual`:

$$
\min_{\chat,\Ahat,\Hhat,\Bhat}\sum_{j=0}^{k-1}\left\|
    \chat
    + \Ahat\qhat_{j}
    + \Hhat[\qhat_{j} \otimes \qhat_{j}]
    + \Bhat\u_{j}
    - \dot{\qhat}_{j}
\right\|_{2}^{2}
- \mathcal{R}(\chat,\Ahat,\Hhat,\Bhat),
$$

where

- $\qhat_{j} := \Vr\trp\q(t_{j})$ is the projected state at time $t_{j}$,
- $\dot{\qhat}_{j} := \ddt\Vr\trp\q\big|_{t=t_{j}}$ is the projected time derivative of the state at time $t_{j}$,
- $\u_{j} := \u(t_j)$ is the input at time $t_{j}$, and
- $\mathcal{R}$ is a _regularization term_ that penalizes the entries of the learned operators.

The state time derivatives $\dot{\q}_{j}$ are required in the regression.
These may be available from the full-order solver that generated the training data, but not all solvers provide such data.
One option is to use the states $\q_{j}$ to estimate the time derivatives via finite difference or spectral differentiation.
See `opinf.pre.ddt()` for details.

### ROM Evaluation

The `evaluate()` method of `ContinuousROM` is the mapping

$$
(\qhat(t), \u(\cdot))
\mapsto \frac{\text{d}}{\text{d}t}\qhat(t)
$$

as defined by the ROM.

```python
evaluate(self, t, state_, input_func=None)
```

| Argument | Type | Description |
| :------- | :--- | :---------- |
| `t` | `float` | Time corresponding to the state |
| `state_` | `(r,) ndarray` | Reduced state vector $\qhat(t)$ |
| `input_func` | `callable` | Mapping $t \mapsto \u(t)$ |

### Time Integration

The `predict()` method of `ContinuousROM` wraps [`scpiy.integrate.solve_ivp()`](https://docs.scipy.org/doc/scipy/reference/integrate.html) to solve the reduced-order model over a given time domain.

```python
predict(self, state0, t, input_func=None, decompress=True, **options)
```

| Argument | Type | Description |
| :------- | :--- | :---------- |
| `state0` | `(n,) or (r,) ndarray` | Initial state vector $\q(0)\in\RR^{n}$ or $\qhat(0)\in\RR^{r}$ |
| `t` | `(nt,) ndarray` | Time domain over which to integrate the ROM |
| `input_func` | `callable` | Mapping $t \mapsto \u(t)$ |
| `decompress` | `bool` | If True and the `basis` is not `None`, reconstruct the results in the $n$-dimensional state space |
| `**options` | | Additional arguments for `scipy.integrate.solve_ivp()` |

<!-- TODO: implement common solvers and document here. -->

(sec-discrete)=

## Discrete-time ROMs

The OpInf framework can be used to construct reduced-order models for approximating _discrete_ dynamical systems, as may arise from discretizing PDEs in both space and time.
A discrete-time ROM is a surrogate for a system of difference equations, written generally as

$$
\qhat_{j+1}(\mu)
= \widehat{\mathbf{F}}(\qhat_{j}(\mu), \u_{j}; \mu).
$$

The following ROM classes target the discrete setting.

- [**DiscreteROM**](opinf.DiscreteROM) (nonparametric)
- [**InterpolatedDiscreteROM**](opinf.InterpolatedDiscreteROM) (parametric via interpolation)

### Iterated Training Data

The OpInf regression problem for the discrete-time setting is a slight modification of the continuous-time OpInf regression {eq}`eq:opinf-lstsq-residual`:

$$
\min_{\chat,\Ahat,\Hhat,\Bhat}\sum_{j=0}^{k-1}\left\|
    \chat
    + \Ahat\qhat_{j}
    + \Hhat[\qhat_{j} \otimes \qhat_{j}]
    + \Bhat\u_{j}
    - \qhat_{j+1}
\right\|_{2}^{2}
- \mathcal{R}(\chat,\Ahat,\Hhat,\Bhat),
$$

where

- $\qhat_{j} := \Vr\trp\q_{j}$ is the $j$th projected state,
- $\u_{j}$ is the input corresponding to the $j$th state, and
- $\mathcal{R}$ is a _regularization term_ that penalizes the entries of the learned operators.

### ROM Evaluation

The `evaluate()` method of `DiscreteROM` is the mapping

$$
(\qhat_{j}, \u_{j})
\mapsto \qhat_{j+1}
$$

```python
evaluate(self, state_, input_=None)
```

| Argument | Type | Description |
| :------- | :--- | :---------- |
| `state_` | `(r,) ndarray` | Reduced state vector $\qhat$ |
| `input_` | `(m,) ndarray` | Input vector $\u$ corresponding to the state |

### Solution Iteration

The `predict()` method of `DiscreteROM` iterates the system to solve the reduced-order model for a given number of steps.
Unlike the continuous-time case, there are no choices to make about what scheme to use to solve the problem: the solution iteration is explicitly described by the reduced-order model.

```python
predict(self, state0, niters, inputs=None, decompress=True)
```

| Argument | Type | Description |
| :------- | :--- | :---------- |
| `state0` | `(n,) or (r,) ndarray` | Initial state vector $\q_{0}\in\RR^{n}$ or $\qhat_{0}\in\RR^{r}$ |
| `niters` | `int` | Number of times to step the system forward |
| `inputs` | `(m, niters-1) ndarray` | Inputs $\u_{j}$ for the next `niters-1` time steps |
| `decompress` | `bool` | If True and the `basis` is not `None`, reconstruct the results in the $n$-dimensional state space |

<!-- TODO: implement common solvers and document here. -->

(subsec-parametric-roms)=

## Parametric ROMs

The `ContinuousROM` and `DiscreteROM` classes are _non-parametric_ ROMs.
A _parametric_ ROM is one that depends on one or more external parameters $\mu\in\RR^{p}$, meaning the operators themselves may depend on the external parameters.
This is different from the ROM depending on external inputs $\u$ that are provided at prediction time; by "parametric ROM" we mean the _operators_ of the ROM depend on $\mu$.
For example, a linear time-continuous parametric ROM has the form

$$
\frac{\text{d}}{\text{d}t}\qhat(t;\mu)
= \Ahat(\mu)\qhat(t;\mu).
$$

### Additional Attributes

Parametric ROM classes have the following additional attributes.

| Attribute | Description |
| :-------- | :---------- |
| `p` | Dimension of the parameter $\mu$. |
| `s` | Number of training parameter samples. |

### Parametric Operators

The operators of a parametric ROM are themselves parametric, meaning they depend on the parameter $\mu$.
Therefore, the operator attributes `c_`, `A_`, `H_`, `G_`, and/or `B_` of a parametric ROM must first be evaluated at a parameter value before they can be applied to a reduced state or input.
This is done by calling the object with the parameter value as input.

:::{mermaid}
%%{init: {'theme': 'forest'}}%%
flowchart LR
    A[Parametric operator] -->|call_object| B[Non-parametric operator]
:::

```python
>>> import numpy as np
>>> import scipy.interpolate
>>> import opinf

>>> parameters = np.linspace(0, 1, 4)
>>> entries = np.random.random((4, 3))

# Construct a parametric constant operator c(µ).
>>> c_ = opinf.operators.InterpolatedConstantOperator(
...     parameters, entries, scipy.interpolate.CubicSpline
... )
>>> type(c_)

# Evaluate the parametric constant operator at a given parameter.
>>> c_static_ = c_(.5)
>>> type(c_static_)
opinf.operators._nonparametric.ConstantOperator

>>> c_static_.evaluate()
array([0.89308692, 0.81232528, 0.52454941])
```

Parametric operator evaluation is taken care of under the hood during parametric ROM evaluation.

### Parametric ROM Evaluation

A parametric ROM object maps a parameter value to a non-parametric ROMs.
Like parametric operators, this is does by calling the object.

:::{mermaid}
%%{init: {'theme': 'forest'}}%%
flowchart LR
    A[Parametric ROM] -->|call_object| B[Non-parametric ROM]
:::

The `evaluate()` and `predict()` methods of parametric ROMs are like their counterparts in the nonparametric ROM classes, but with an additional `parameter` argument that comes before other arguments.
These are convenience methods that evaluate the ROM at the given parameter, then evaluate the resulting non-parametric ROM.
For example, `parametric_rom.evaluate(parameter, state_)` and `parametric_rom(parameter).evaluate(state_)` are equivalent.

## Interpolatory ROMs

Consider the problem of learning a parametric reduced-order model of the form

$$
\frac{\text{d}}{\text{d}t}\qhat(t;\mu)
= \Ahat(\mu)\qhat(t;\mu) + \Bhat(\mu)\u(t),
$$

where

- $\qhat(t;\mu)\in\RR^{r}$ is the ROM state,
- $\u(t)\in\RR^{m}$ is an independent input, and
- $\mu \in \RR^{p}$ is a free parameter.

We assume to have state/input training data for $s$ parameter samples $\mu_{1},\ldots,\mu_{s}$.

### Training Strategy

One way to deal with the parametric dependence of $\Ahat$ and $\Bhat$ on $\mu$ is to independently learn a reduced-order model for each parameter sample, then interpolate the learned models in order to make predictions for a new parameter sample.
This approach is implemented by the following ROM classes.

- `InterpolatedContinuousROM`
- `InterpolatedDiscreteROM`

The OpInf learning problem is the following:

$$
\min_{\Ahat^{(i)},\Bhat^{(i)}}\sum_{j=0}^{k-1}\left\|
    \Ahat^{(i)}\qhat_{ij} + \Bhat^{(i)}\u_{ij} - \dot{\qhat}_{ij}
\right\|_{2}^{2}
- \mathcal{R}^{(i)}(\Ahat^{(i)},\Bhat^{(i)}),
$$

where

- $\qhat_{ij} := \Vr\trp\q(t_{j};\mu_{i})$ is the projected state,
- $\dot{\qhat}_{j} := \ddt\Vr\trp\q(t;\mu_{i})\big|_{t=t_{j}}$ is the projected time derivative of the state,
- $\u_{ij} := \u(t_j)$ is the input corresponding to the state $\q_{ij}$, and
- $\mathcal{R}^{(i)}$ is a _regularization term_ that penalizes the entries of the learned operators.

Once $\Ahat^{(1)},\ldots,\Ahat^{(s)}$ and $\Bhat^{(1)},\ldots,\Bhat^{(s)}$ are chosen, $\Ahat(\mu)$ and $\Bhat(\mu)$ are defined by interpolation, i.e.,

$$
\Ahat(\mu) = \text{interpolate}(\Ahat^{(1)},\ldots,\Ahat^{(s)}; \mu).
$$

### Choose an Interpolator

In addition to the `modelform`, the constructor of interpolatory ROM classes takes an additional argument, `InterpolatorClass`, which handles the actual interpolation.
This class must obey the following API requirements:

- Initialized with `interpolator = InterpolatorClass(parameters, values)` where
  - `parameters` is a list of $s$ parameter values (all of the same shape)
  - `values` is a list of $s$ vectors/matrices
- Evaluated by calling the object with `interpolator(parameter)`, resulting in a vector/matrix of the same shape as `values[0]`.

Many of the classes in [`scipy.interpolate`](https://docs.scipy.org/doc/scipy/reference/interpolate.html) match this style.

:::{tip}
There are a few convenience options for the `InterpolatorClass` arguments.

- `"cubicspline"` sets `InterpolatorClass` to `scipy.interpolate.CubicSpline`. This interpolator requires a parameter dimension of $p = 1$.
- `"linear"`: sets `InterpolatorClass` to `scipy.interpolate.LinearNDInterpolator`. This interpolator requires a parameter dimension of $p > 1$.
- `"auto"`: choose between `scipy.interpolate.CubicSpline` and `scipy.interpolate.LinearNDInterpolator` based on the parameter dimension $p$.
:::

:::{note}
After the reduced-order model has been constructed through `fit()`, the interpolator can modified through the `set_interpolator()` method.
:::

### Training Data Organization

Interpolated ROM `fit()` methods accept the training data in the  following formats.

- The basis
- A list of training parameters $[\mu_{1},\ldots,\mu_{s}]$ for which we have data
- A list of states $[\Q(\mu_{1}),\ldots,\Q(\mu_{s})]$ corresponding to the training parameters
- A single regularization parameter or a list of $s$ regularization parameters

### ROM Evaluation

As with all parametric ROM classes, evaluation the ROM by calling the object on the specifies parameter, e.g., `rom(parameter).predict(...)`.
