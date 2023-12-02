# `opinf.operators` (new)

```{eval-rst}
.. automodule:: opinf.operators_new
```

## Introduction

Reduced-order models based on Operator Inference are systems of ordinary differential equations (or discrete-time difference equations) that can be written as

$$
\begin{align*}
   \ddt\qhat(t)
   = \sum_{\ell=1}^{n_\textrm{terms}}
   \Ophat_{\ell}(\qhat(t),\u(t))
\end{align*}
$$

where each $\Ophat_{\ell}:\RR^{r}\times\RR^{m}\to\RR^{r}$ is a vector-valued function that is polynomial with respect to the reduced state $\qhat\in\RR^{n}$ and the input $\u\in\RR^{m}$.
Such functions can be represented by a matrix-vector product

$$
\begin{align*}
    \Ophat_{\ell}(\qhat,\u) = \Ohat_{\ell}\d_{\ell}(\qhat,\u)
\end{align*}
$$

for some data-dependent vector $\d_{\ell}:\RR^{r}\times\RR^{m}\to\RR^{d}$ and constant matrix $\Ohat_{\ell}\in\RR^{r\times d}$.
The goal of Operator Inference is to learn---using only data---the _operator entries_ $\Ohat_\ell$ for each operator in the reduced-order model.

The classes in this module represent different types of operators that can used in defining the structure of an Operator Inference reduced-order model.

:::{admonition} Example
:class: tip

To construct a linear time-invariant (LTI) system

$$
\begin{align}
    \ddt\qhat(t)
    = \Ahat\qhat(t) + \Bhat\u(t),
    \qquad
    \Ahat\in\RR^{r \times r},
    ~
    \Bhat\in\RR^{r \times m},
\end{align}
$$ (eq:operators:ltiexample)

we use the following operator classes from {mod}`opinf.operators_new`.

| Class | Definition | Operator entries | data vector |
| :---- | :--------- | :--------------- | :---------- |
| {class}`LinearOperator` | $\Ophat_{1}(\qhat,\u) = \Ahat\q$ | $\Ohat_{1} = \Ahat \in \RR^{r\times r}$ | $\d_{1}(\qhat,\u) = \qhat\in\RR^{r}$ |
| {class}`InputOperator` | $\Ophat_{2}(\qhat,\u) = \Bhat\u$ | $\Ohat_{2} = \Bhat \in \RR^{r\times m}$ | $\d_{2}(\qhat,\u) = \u\in\RR^{m}$ |

An {class}`opinf.roms_new.ContinuousROM` object can be instantiated with a list of operators objects to represent {eq}`eq:operators:ltiexample` as

$$
\begin{align*}
    \ddt\qhat(t)
    = \Ophat_{1}(\qhat(t),\u(t))
    + \Ophat_{2}(\qhat(t),\u(t)).
\end{align*}
$$

```python
import opinf

LTI_ROM = opinf.Continuous_ROM(
    operators=[
        opinf.operators_new.LinearOperator(),
        opinf.operators_new.InputOperator(),
    ]
)
```

:::

## Nonparametric Operators

A _nonparametric_ operator is one where the entries matrix $\Ohat$ is constant (as opposed to [parametric operators](sec-operators-parametric)).

### API Summary

#### Initialization

Every nonparametric operator class can be initialized without arguments.
If the operator entries are known, they can be passed into the constructor or set later with the `set_entries()` method.
The entries are stored as the `entries` attribute and can be accessed with slicing operations `[:]`.

Once the entries are set, the following methods are used to compute the action
of the operator or its derivatives.

- `apply()`: compute the operator action $\Ophat(\qhat, \u)$.
- `jacobian()`: construct the state Jacobian $\frac{\textrm{d}}{\textrm{d}\qhat}\Ophat(\qhat, \u)$.

#### Calibrating Operator Entries

Given a list of operators, the ROM classes defined in {mod}`opinf.roms_new` set up a regression problem to learn the entries of each operator from data.
To facilitate this, each nonparametric operator class has a static method `datablock()` that, given state-input data pairs $\{(\qhat_j,\u_j)\}_{j=0}^{k-1}$, forms the matrix

$$
    \D = \left[\begin{array}{c|c|c|c}
        & & & \\
        \mathbf{d}(\qhat_0,\u_0) & \mathbf{d}(\qhat_1,\u_1) & \cdots & \mathbf{d}(\qhat_{k-1},\u_{k-1})
        \\ & & &
    \end{array}\right]
    \in \RR^{d \times k}
$$

where $\Ophat(\qhat,\u) = \Ohat\d(\qhat,\u)$.
The ROM classes call this method, solve the regression problem, and set the entries of the operators based on the regression solution.

#### Galerkin Projection

For a full-order operator $\Op:\RR^{n}\times\RR^{m}\to\RR^{n}$, the _projection_ of $\Op$ is the operator $\Ophat:\RR^{r}\times\RR^{m}\to\RR^{r}$ defined by

$$
\begin{align*}
    \Ophat(\qhat, \u) = \Wr\trp\Op(\Vr\qhat, u)
\end{align*}
$$

where
$\qhat\in\RR^{r}$ is the reduced-order state,
$\u\in\RR^{m}$ is the input, and
$\Vr\qhat_{r}\in\RR^{n}$ is the reduced-order
approximation of the full-order state,
with trial basis $\Vr\in\RR^{n \times r}$
and test basis $\Wr\in\RR^{n \times r}$.
If $\Wr = \Vr$, the result is called a _Galerkin projection_.
If $\Wr \neq \Vr$, it is called a _Petrov-Galerkin projection_.

Every operator class has a `galerkin()` method that receives trial and
test bases and returns a new object representing the projected operator.

:::{admonition} Example
:class: tip

Consider the bilinear full-order operator
$\Op(\q,\u) = \N[\u\otimes\q]$ where $\N\in\RR^{n \times nm}$.
The Galerkin projection of this operator is the bilinear operator

$$
\begin{align*}
    \Ophat(\qhat,\u)
    = \Wr\trp\N[\u\otimes\Vr\qhat]
    = \Nhat[\u\otimes\qhat]
\end{align*}
$$

where $\Nhat = \Wr\trp\N(\I_m\otimes\Vr) \in \RR^{r\times rm}$.
:::

:::{important}
The goal of Operator Inference is to learn operator entries _without_ using a direct projection because full-order operators are unknown or computationally inaccessible.
However, in some scenarios a subset of the ROM operators are known, in which case only the remaining operators need to be inferred from data.
When a ROM object is instantiated with an operator that already has its entries set, the `galerkin()` method is called to project the operator to the appropriate dimension and that operator is not included in the operator inference.
:::

#### Model Persistence

Operators can be saved to disk in HDF5 format via the `save()` method.
Every operator has a class method `load()` for loading an operator from the HDF5 file previously produced by `save()`.

### Nonparametric Monolithic Operators

These operator classes are used in ROMs where the low-dimensional state approximation is monolithic, meaning $\q \approx \Vr\qhat$ where $\Vr$ does not have a block-diagonal sparsity structure.

```{eval-rst}
.. currentmodule:: opinf.operators_new

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    ConstantOperator
    LinearOperator
    QuadraticOperator
    CubicOperator
    InputOperator
    StateInputOperator
```

### Nonparametric Multilithic Operators

:::{admonition} TODO
Multilithic classes
:::

(sec-operators-parametric)=
## Parametric Operators

Operators are called _parametric_ if the operator entries depend on an independent vector
$\bfmu\in\RR^{p}$, i.e., $\Ophat(\qhat,\u;\bfmu) = \Ohat(\bfmu)\d(\qhat,\u)$ where now $\Ohat:\RR^{p}\to\RR^{r\times d}$.

:::{admonition} Example
:class: tip
Let $\bfmu = [~\mu_{1}~~\mu_{2}~]\trp$.
The linear operator
$\Ophat(\qhat,\u;\bfmu) = (\mu_{1}\Ahat_{1} + \mu_{2}\Ahat_{2})\qhat$
is a parametric operator with parameter-dependent entries $\Ohat(\bfmu) = \mu_{1}\Ahat_{1} + \mu_{2}\Ahat_{2}$.
:::

:::{warning}
The rest of this page is under construction.
:::

:::{admonition} TODO

- Constructor takes in parameter information (and anything needed by the underlying nonparametric class)
- `__call__()` or `evaluate()` maps parameter values to a nonparametric operator
:::

### Interpolated Operators

$$
\begin{align*}
    \Ophat(\qhat,\u;\mu)
    = \text{interpolate}(
    (\bfmu_{1},\Ophat_{1}),\ldots,(\bfmu_{s},\Ophat_{s}); \bfmu)
\end{align*}
$$

:::{admonition} TODO
Constructor takes in `s` (the number of parameter samples) and an interpolator class.
:::

### Affine Operators

$$
\begin{align*}
    \Ophat(\qhat,\u;\bfmu)
    = \sum_{\ell=1}^{n_{\theta}}\theta_{\ell}(\bfmu)\Ophat_{\ell}(\qhat,\u)
\end{align*}
$$

:::{admonition} TODO
Constructor takes in list of the affine coefficient functions.
:::
