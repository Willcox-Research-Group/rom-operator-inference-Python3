# `opinf.operators` (new)

```{eval-rst}
.. automodule:: opinf.operators_new
```

<!--
:::{admonition} Summary
Operators defined in {mod}`opinf.operators` are the building blocks for the dynamical systems models defined in {mod}`opinf.models`.
There are a few different types of operators:

- [Nonparametric operators](sec-operators-nonparametric) do not depend on external parameters, while [parametric operators](sec-operators-parametric) have a dependence on
- Monolithic operators are designed for dense systems; multilithic operators are designed for systems with sparse block structure.
:::
-->

## Introduction

Reduced-order models based on Operator Inference are systems of [ordinary differential equations](opinf.models.ContinuousModel) (or [discrete-time difference equations](opinf.models.DiscreteModel)) that can be written as

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

An {class}`opinf.models.ContinuousModel` object can be instantiated with a list of operators objects to represent {eq}`eq:operators:ltiexample` as

$$
\begin{align*}
    \ddt\qhat(t)
    = \Ophat_{1}(\qhat(t),\u(t))
    + \Ophat_{2}(\qhat(t),\u(t)).
\end{align*}
$$

```python
import opinf

LTI_ROM = opinf.models.ContinuousModel(
    operators=[
        opinf.operators_new.LinearOperator(),
        opinf.operators_new.InputOperator(),
    ]
)
```

:::

(sec-operators-nonparametric)=
## Nonparametric Operators

A _nonparametric_ operator is one where the entries matrix $\Ohat_\ell$ is constant (as opposed to [parametric operators](sec-operators-parametric)).

### API Summary

#### Initialization

Every nonparametric operator class can be initialized without arguments.
If the operator entries are known, they can be passed into the constructor or set later with the `set_entries()` method.
The entries are stored as the `entries` attribute and can be accessed with slicing operations `[:]`.

Once the entries are set, the following methods are used to compute the action
of the operator or its derivatives.

- `apply()`: compute the operator action $\Ophat_\ell(\qhat, \u)$.
- `jacobian()`: construct the state Jacobian $\ddqhat\Ophat_\ell(\qhat, \u)$.

#### Calibrating Operator Entries

Nonparametric operator classes have a static `datablock()` method that, given state-input data pairs $\{(\qhat_j,\u_j)\}_{j=0}^{k-1}$, forms the matrix

$$
    \D_{\ell}\trp = \left[\begin{array}{c|c|c|c}
        & & & \\
        \d_{\ell}(\qhat_0,\u_0) & \d_{\ell}(\qhat_1,\u_1) & \cdots & \d_{\ell}(\qhat_{k-1},\u_{k-1})
        \\ & & &
    \end{array}\right]
    \in \RR^{d \times k}
$$

where the operator is given by $\Ophat_{\ell}(\qhat,\u) = \Ohat_{\ell}\d_{\ell}(\qhat,\u)$.
For a model consisting of multiple operators, e.g.,

$$
\begin{align*}
   \ddt\qhat(t)
   = \sum_{\ell=1}^{n_\textrm{terms}}
   \Ophat_{\ell}(\qhat(t),\u(t))
   = \sum_{\ell=1}^{n_\textrm{terms}}
   \Ohat_{\ell}\d_{\ell}(\qhat(t),\u(t)),
\end{align*}
$$

the Operator Inference regression to learn the operator entries from data is given by

$$
\begin{align*}
    \min_{\Ohat_1,\ldots,\Ohat_{n_\textrm{terms}}}\sum_{j=0}^{k-1}\left\|
        \sum_{\ell=1}^{n_\textrm{terms}}\Ohat_\ell\d_\ell(\qhat_j,\u_j) - \dot{\qhat}_j
    \right\|_2^2
    = \min_{\Ohat}\left\|
        \D\Ohat\trp - [~\dot{\qhat}_0~~\cdots~~\dot{\qhat}_{k-1}~]\trp
    \right\|_F^2,
\end{align*}
$$

where the complete operator matrix $\Ohat$ and data matrix $\D$ are concatenations of the operator and data matrices from each operator:

$$
\begin{align*}
    \Ohat = \left[\begin{array}{ccc}
        & & \\
        \Ohat_1 & \cdots & \Ohat_{n_\textrm{terms}}
        \\ & &
    \end{array}\right],
    \qquad
    \D = \left[\begin{array}{ccc}
        & & \\
        \D_1 & \cdots & \D_{n_\textrm{terms}}
        \\ & &
    \end{array}\right].
\end{align*}
$$

The `fit()` method in an {mod}`opinf.models` class calls the `datablock()` method of each operator to assemble the full data matrix $\D$, solves the regression problem for the full data matrix $\Ohat$, and sets the entries of the $\ell$-th operator to $\Ohat_{\ell}$.

#### Galerkin Projection

Every operator class has a `galerkin()` method that performs intrusive projection.
Consider an operator $\Op:\RR^{n}\times\RR^{m}\to\RR^{n}$, written $\Op(\q,\u)$, where

- $\q\in\RR^n$ is the full-order state, and
- $\u\in\RR^m$ is the input.

Given a *trial basis* $\Vr\in\RR^{n\times r}$ and a *test basis* $\Wr\in\RR^{n\times r}$, the corresponding *intrusive projection* of $\Op$ is the operator $\Ophat:\RR^{r}\times\RR^{m}\to\RR^{r}$ defined by

$$
\begin{align*}
    \Ophat(\qhat, \u) = \Wr\trp\Op(\Vr\qhat, \u)
\end{align*}
$$

where
- $\qhat\in\RR^{r}$ is the reduced-order state, and
- $\u\in\RR^{m}$ is the input (as before).

This approach uses the low-dimensional state approximation $\q = \Vr\qhat$.
If $\Wr = \Vr$, the result is called a *Galerkin projection*.
If $\Wr \neq \Vr$, it is called a *Petrov-Galerkin projection*.

:::{admonition} Example
:class: tip

Consider the bilinear operator
$\Op(\q,\u) = \N[\u\otimes\q]$ where $\N\in\RR^{n \times nm}$.
The intrusive Galerkin projection of $\Op$ is the bilinear operator

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
The goal of Operator Inference is to learn operator entries *without* using intrusive projection because full-order operators are unknown or computationally inaccessible.
However, in some scenarios a subset of the model operators are known, in which case only the remaining operators need to be inferred from data.
:::

#### Model Persistence

Operators can be saved to disk in [HDF5 format](https://www.h5py.org/) via the `save()` method.
Every operator has a class method `load()` for loading an operator from the HDF5 file previously produced by `save()`.

### Nonparametric Operator Classes

<!-- These operator classes are used in models where the state is monolithic, meaning the operators do not enjoy a block sparsity structure. -->

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

<!-- ### Nonparametric Multilithic Operators

:::{admonition} TODO
Multilithic classes
::: -->

(sec-operators-parametric)=
## Parametric Operators

Operators are called _parametric_ if the operator entries depend on an independent parameter vector
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
