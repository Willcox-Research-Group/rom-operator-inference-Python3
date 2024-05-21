# `opinf.ddt`

```{eval-rst}
.. automodule:: opinf.ddt

.. currentmodule:: opinf.ddt

**Classes**

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    UniformFiniteDifferencer
    NonuniformFiniteDifferencer

**Finite Difference Schemes for Uniformly Spaced Data**

*Forward Differences*

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    fwd1
    fwd2
    fwd3
    fwd4
    fwd5
    fwd6

*Backward Differences*

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    bwd1
    bwd2
    bwd3
    bwd4
    bwd5
    bwd6

*Central Differences*

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    ctr2
    ctr4
    ctr6

*Mixed Differences*

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    ord2
    ord4
    ord6
    ddt_uniform

**Finite Difference Schemes for Nonuniformly Spaced Data**

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    ddt_nonuniform
    ddt

```

## Time Derivative Estimation

To calibrate time-continuous models, Operator Inference requires the time derivative of the state snapshots.
For example, consider the LTI system

$$
\begin{aligned}
    \ddt\qhat(t)
    = \Ahat\qhat(t) + \Bhat\u(t).
\end{aligned}
$$ (eq:ddt:lti-reduced)

Here, $\qhat(t)\in\RR^{r}$ is the time-dependent ([reduced-order](./basis.ipynb)) state and $\u(t)\in\RR^{m}$ is the time-dependent input.
In order to learn $\Ahat$ and $\Bhat$, Operator Inference solves a regression problem of the form

$$
\begin{aligned}
    \min_{\Ahat,\Bhat}\sum_{j=0}^{k-1}\left\|
    \Ahat\qhat_j + \Bhat\u_j
    - \dot{\qhat}_j
    \right\|_2^2
\end{aligned}
$$

or similar, where each triplet $(\qhat_j, \dot{\qhat}_j, \u_j)$ should correspond to the solution of {eq}`eq:ddt:lti-reduced` at some time $t_j$, $j = 0, \ldots, k - 1$.
In particular, we want

$$
\begin{aligned}
    \dot{\qhat}_j
    \approx \ddt\qhat(t)\big|_{t = t_j}
    = \Ahat\qhat_j + \Bhat\u_j.
\end{aligned}
$$

This module provides tools for estimating the snapshot time derivatives $\dot{\qhat}_0,\ldots,\dot{\qhat}_{k-1}\in\RR^{r}$ from the reduced snapshots $\qhat_0,\ldots,\qhat_{k-1}\in\RR^{r}$.

:::{warning}
In some cases, a full-order model may provide snapshot time derivatives $\dot{\q}_0,\ldots,\dot{\q}_{k-1}\in\RR^{n}$ in addition to state snapshots $\q_0,\ldots,\q_{k-1}\in\RR^{n}$.
If any lifting or preprocessing steps are used on the state snapshots, be careful to use the appropriate transformation for snapshot time derivatives, which may be different than the transformation used on the snapshots themselves.

For example, consider the affine state approximation $\q(t) \approx \Vr\qhat(t) + \bar{\q}$ with an orthonormal basis matrix $\Vr\in\RR^{n\times r}$ and a fixed vector $\bar{\q}\in\RR^{n}$.
In this case,

$$
\begin{aligned}
    \ddt\q(t)
    \approx \ddt\left[\Vr\qhat(t) + \bar{\q}\right]
    = \Vr\ddt\left[\qhat(t)\right].
\end{aligned}
$$

Hence, while the compressed state snapshots are given by $\qhat_j = \Vr\trp(\q_j - \bar{\q})$, the correct compressed snapshot time derivatives are $\dot{\qhat}_j = \Vr\trp\dot{\q}_j$ (without the $\bar{\q}$ shift).

See {meth}`opinf.lift.LifterTemplate.lift_ddts` and {meth}`opinf.pre.TransformerTemplate.transform_ddts`.
:::

## Partial Estimation

Every finite difference scheme has limitations on where the derivative can be estimated.
For example, a [first-order backward scheme](opinf.ddt.bwd1) requires $\qhat(t_{j-1})$ and $\qhat(t_j)$ to estimate $\dot{\qhat}(t_j)$, hence the derivative cannot be estimated at $t = t_0$.
The forward, backward, and central difference functions ({func}`fwd1`, {func}`bwd3`, {func}`ctr6`, etc.) take in a snapshot matrix $\Qhat\in\RR^{r\times k}$ and (optionally) the corresponding input matrix $\U\in\RR^{m\times k}$ and return a subset of the snapshots $\Qhat'\in\mathbb{R}^{r\times k'}$, the corresponding derivatives $\dot{\Qhat}\in\RR^{r\times k'}$, and (optionally) the corresponding inputs $\U'\in\RR^{m \times k'}$.

## Complete Estimation

The functions {func}`ddt_uniform`, {func}`ddt_nonuniform`, {func}`ord2`, {func}`ord4`, and {func}`ord6` mix forward, central, and backward differences to provide derivative estimates

## Custom Estimators

New time derivative estimators can be defined by inheriting from {class}`DerivativeEstimatorTemplate`.
Once implemented, the [`verify()`](opinf.ddt.DerivativeEstimatorTemplate.verify) method may be used to compare the results of [`estimate()`](opinf.ddt.DerivativeEstimatorTemplate.estimate) with true derivatives for a limited number of test cases.
