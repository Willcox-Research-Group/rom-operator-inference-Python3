# `opinf.operators`

```{eval-rst}
.. automodule:: opinf.operators
```

<!--
:::{admonition} Summary
Operators defined in {mod}`opinf.operators` are the building blocks for the dynamical systems models defined in {mod}`opinf.models`.
There are a few different types of operators:

- [Nonparametric operators](sec-operators-nonparametric) do not depend on external parameters, while [parametric operators](sec-operators-parametric) have a dependence on
- Monolithic operators are designed for dense systems; multilithic operators are designed for systems with sparse block structure.
:::
-->

## Overview

Reduced-order models based on Operator Inference are systems of [ordinary differential equations](opinf.models.ContinuousModel) (or [discrete-time difference equations](opinf.models.DiscreteModel)) that can be written as

$$
\begin{align*}
   \ddt\qhat(t)
   = \sum_{\ell=1}^{n_\textrm{terms}}
   \Ophat_{\ell}(\qhat(t),\u(t))
\end{align*}
$$ (eq:operators:model)

where each $\Ophat_{\ell}:\RR^{r}\times\RR^{m}\to\RR^{r}$ is a vector-valued function that is polynomial with respect to the reduced state $\qhat\in\RR^{n}$ and the input $\u\in\RR^{m}$.
Such functions, which we refer to as *operators*, can be represented by a matrix-vector product

$$
\begin{align*}
    \Ophat_{\ell}(\qhat,\u) = \Ohat_{\ell}\d_{\ell}(\qhat,\u)
\end{align*}
$$

for some data-dependent vector $\d_{\ell}:\RR^{r}\times\RR^{m}\to\RR^{d}$ and constant matrix $\Ohat_{\ell}\in\RR^{r\times d}$.
The goal of Operator Inference is to learn---using only data---the *operator entries* $\Ohat_\ell$ for each operator in the reduced-order model.

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

we use the following operator classes.

| Class | Definition | Operator entries | data vector |
| :---- | :--------- | :--------------- | :---------- |
| {class}`LinearOperator` | $\Ophat_{1}(\qhat,\u) = \Ahat\qhat$ | $\Ohat_{1} = \Ahat \in \RR^{r\times r}$ | $\d_{1}(\qhat,\u) = \qhat\in\RR^{r}$ |
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

LTI_model = opinf.models.ContinuousModel(
    operators=[
        opinf.operators.LinearOperator(),
        opinf.operators.InputOperator(),
    ]
)
```

:::

(sec-operators-nonparametric)=
## Nonparametric Operators

A _nonparametric_ operator $\Ophat_{\ell}(\qhat,\u) = \Ohat_{\ell}\d_{\ell}(\qhat,\u)$ is one where the entries matrix $\Ohat_\ell$ is constant.

```{eval-rst}
.. currentmodule:: opinf.operators

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

<!-- These operator classes are used in models where the state is monolithic, meaning the system does not exhibit a block sparsity structure. -->

<!-- ### Nonparametric Multilithic Operators

:::{admonition} TODO
Multilithic classes
::: -->

Nonparametric operators can be instantiated without arguments.
If the operator entries are known, they can be passed into the constructor or set later with the `set_entries()` method.
The entries are stored as the `entries` attribute and can be accessed with slicing operations `[:]`.

In a reduced-order model, there are two ways to determine the operator entries:

- Learn the entries from data (nonintrusive Operator Inference), or
- Shrink an existing high-dimensional operator (intrusive Galerkin projection).

Once the entries are set, the following methods are used to compute the action
of the operator or its derivatives.

- `apply()`: compute the operator action $\Ophat_\ell(\qhat, \u)$.
- `jacobian()`: construct the state Jacobian $\ddqhat\Ophat_\ell(\qhat, \u)$.

(sec-operators-calibration)=
### Learning Operators from Data

Suppose we have state-input-derivative data triples $\{(\qhat_j,\u_j,\dot{\qhat}_j)\}_{j=0}^{k-1}$ that approximately satisfy the model {eq}`eq:operators:model`, i.e.,

$$
\begin{align*}
    \dot{\qhat}_j
    \approx \Ophat(\qhat_j, \u_j)
    = \sum_{\ell=1}^{n_\textrm{terms}} \Ophat_{\ell}(\qhat_j, \u_j)
    = \sum_{\ell=1}^{n_\textrm{terms}} \Ohat_{\ell}\d_{\ell}(\qhat_j, \u_j).
\end{align*}
$$ (eq:operators:approx)

Operator Inference determines the operator entries $\Ohat_1,\ldots,\Ohat_{n_\textrm{terms}}$ by minimizing the residual of {eq}`eq:operators:approx`:

$$
\begin{align*}
    \min_{\Ohat_1,\ldots,\Ohat_{n_\textrm{terms}}}\sum_{j=0}^{k-1}\left\|
        \sum_{\ell=1}^{n_\textrm{terms}}\Ohat_\ell\d_\ell(\qhat_j,\u_j) - \dot{\qhat}_j
    \right\|_2^2.
\end{align*}
$$

To facilitate this, nonparametric operator classes have a static `datablock()` method that, given the state-input data pairs $\{(\qhat_j,\u_j)\}_{j=0}^{k-1}$, forms the matrix

$$
\begin{align*}
    \D_{\ell}\trp = \left[\begin{array}{c|c|c|c}
        & & & \\
        \d_{\ell}(\qhat_0,\u_0) & \d_{\ell}(\qhat_1,\u_1) & \cdots & \d_{\ell}(\qhat_{k-1},\u_{k-1})
        \\ & & &
    \end{array}\right]
    \in \RR^{d \times k}.
\end{align*}
$$

Then {eq}`eq:operators:approx` can be written in the linear least-squares form

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

Model classes from {mod}`opinf.models` are instantiated with a list of operators.
The model's `fit()` method calls the `datablock()` method of each operator to assemble the full data matrix $\D$, solves the regression problem for the full data matrix $\Ohat$ (see {mod}`opinf.lstsq`), and sets the entries of the $\ell$-th operator to $\Ohat_{\ell}$.

:::{admonition} Example
:class: tip

For the LTI system {eq}`eq:operators:ltiexample`, the operator inference problem is the following regression.

$$
\begin{align*}
    \min_{\Ahat,\Bhat}\sum_{j=0}^{k-1}\left\|
        \Ahat\qhat_j + \Bhat\u_j - \dot{\qhat}_j
    \right\|_2^2
    = \min_{\Ohat}\left\|
        \D\Ohat\trp - [~\dot{\qhat}_0~~\cdots~~\dot{\qhat}_{k-1}~]\trp
    \right\|_F^2,
\end{align*}
$$

with operator matrix $\Ohat=[~\Ahat~~\Bhat~]$
and data matrix $\D = [~\Qhat\trp~~\U\trp~]$
where $\Qhat = [~\qhat_0~~\cdots~~\qhat_{k-1}~]$
and $\U = [~\u_0~~\cdots~~\u_{k-1}~]$.
:::

:::{important}
Only operators whose entries are _not initialized_ (set to `None`) when a model is constructed are learned with Operator Inference when `fit()` is called.
For example, suppose for the LTI system {eq}`eq:operators:ltiexample` an appropriate input matrix $\Bhat$ is known and stored as the variable `B_`.

```python
import opinf

LTI_model = opinf.models.ContinuousModel(
    operators=[
        opinf.operators.LinearOperator(),   # No entries specified.
        opinf.operators.InputOperator(B_),  # Entries set to B_.
    ]
)
```

In this case, `LIT_model.fit()` only determines the entries of the {class}`LinearOperator` object using Operator Inference, with regression problem

$$
\begin{align*}
    &\min_{\Ahat,}\sum_{j=0}^{k-1}\left\|
        \Ahat\qhat_j - (\dot{\qhat}_j - \Bhat\u_j)
    \right\|_2^2
    \\
    &= \min_{\Ohat}\left\|
        \Qhat\trp\Ahat\trp - [~(\dot{\qhat}_0 - \Bhat\u_0)~~\cdots~~(\dot{\qhat}_{k-1} - \Bhat\u_{k-1})~]\trp
    \right\|_F^2.
\end{align*}
$$

:::

### Learning Operators via Projection

The goal of Operator Inference is to learn operator entries from data because full-order operators are unknown or computationally inaccessible.
However, in some scenarios a subset of the full-order model operators are known, in which case the corresponding reduced-order model operators can be determined through *intrusive projection*.
Consider a full-order operator $\Op:\RR^{n}\times\RR^{m}\to\RR^{n}$, written $\Op(\q,\u)$, where

- $\q\in\RR^n$ is the full-order state, and
- $\u\in\RR^m$ is the input.

Given a *trial basis* $\Vr\in\RR^{n\times r}$ and a *test basis* $\Wr\in\RR^{n\times r}$, the corresponding intrusive projection of $\Op$ is the operator $\Ophat:\RR^{r}\times\RR^{m}\to\RR^{r}$ defined by

$$
\begin{align*}
    \Ophat(\qhat, \u) = \Wr\trp\Op(\Vr\qhat, \u)
\end{align*}
$$

where
- $\qhat\in\RR^{r}$ is the reduced-order state, and
- $\u\in\RR^{m}$ is the input (the same as before).

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

Every operator class has a `galerkin()` method that performs intrusive projection.

(sec-operators-parametric)=
## Parametric Operators

Operators are called _parametric_ if the operator entries depend on an independent parameter vector
$\bfmu\in\RR^{p}$, i.e., $\Ophat_{\ell}(\qhat,\u;\bfmu) = \Ohat_{\ell}(\bfmu)\d_{\ell}(\qhat,\u)$ where now $\Ohat:\RR^{p}\to\RR^{r\times d}$.

:::{admonition} Example
:class: tip
Let $\bfmu = [~\mu_{1}~~\mu_{2}~]\trp$.
The linear operator
$\Ophat_1(\qhat,\u;\bfmu) = (\mu_{1}\Ahat_{1} + \mu_{2}\Ahat_{2})\qhat$
is a parametric operator with parameter-dependent entries $\Ohat_1(\bfmu) = \mu_{1}\Ahat_{1} + \mu_{2}\Ahat_{2}$.
:::

(sec-operators-interpolated)=
### Interpolated Operators

These operators handle the parametric dependence on $\bfmu$ by using elementwise interpolation:

$$
\begin{align*}
    \Ohat_{\ell}(\bfmu)
    = \text{interpolate}(
    (\bfmu_{1},\Ohat_{\ell}^{(1)}),\ldots,(\bfmu_{s},\Ohat_{\ell}^{(s)}); \bfmu),
\end{align*}
$$

where $\bfmu_1,\ldots,\bfmu_s$ are training parameter values and $\Ohat_{\ell}^{(i)} = \Ohat_{\ell}(\bfmu_i)$ for $i=1,\ldots,s$.

```{eval-rst}
.. currentmodule:: opinf.operators

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    InterpolatedConstantOperator
    InterpolatedLinearOperator
    InterpolatedQuadraticOperator
    InterpolatedCubicOperator
    InterpolatedInputOperator
    InterpolatedStateInputOperator
```

<!-- ### Affine Operators

$$
\begin{align*}
    \Ophat(\qhat,\u;\bfmu)
    = \sum_{\ell=1}^{n_{\theta}}\theta_{\ell}(\bfmu)\Ophat_{\ell}(\qhat,\u)
\end{align*}
$$

:::{admonition} TODO
Constructor takes in list of the affine coefficient functions.
::: -->

## Utilities

```{eval-rst}
.. currentmodule:: opinf.operators

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    is_nonparametric
    is_parametric
    has_inputs
```
