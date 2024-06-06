# `opinf.lstsq`

```{eval-rst}
.. automodule:: opinf.lstsq
```

<!-- The following [least-squares regression problem](subsec-opinf-regression) is at the heart of Operator Inference:

$$
\min_{\chat,\Ahat,\Hhat,\Bhat}\sum_{j=0}^{k-1}\left\|
    \chat
    + \Ahat\qhat_{j}
    + \Hhat[\qhat_{j} \otimes \qhat_{j}]
    + \Bhat\u_{j}
    - \dot{\qhat}_{j}
\right\|_{2}^{2}
\\
= \min_{\Ohat}\left\|
    \D\Ohat\trp - \mathbf{R}\trp
\right\|_{F}^{2},
$$

where
- $\qhat_{j} = \Vr\trp\q(t_{j})$ is the state at time $t_{j}$ represented in the coordinates of the basis,
- $\dot{\qhat}_{j} = \ddt\Vr\trp\q\big|_{t=t_{j}}$ is the time derivative of the state at time $t_{j}$ in the coordinates of the basis,
- $\u_{j} = \u(t_j)$ is the input at time $t_{j}$,
- $\D$ is the _data matrix_ containing low-dimensional state data,
- $\Ohat$ is the _operator matrix_ of unknown operators to be inferred, and
- $\mathbf{R}$ is the matrix of low-dimensional time derivative data.

We often need to add a _regularization term_ $\mathcal{R}(\Ohat)$ that penalizes the entries of the learned operators.
This promotes stability and accuracy in the learned reduced-order model by preventing overfitting.
The problem stated above then becomes

$$
\min_{\Ohat}\left\|
    \D\Ohat\trp - \mathbf{R}\trp
\right\|_{F}^{2} + \mathcal{R}(\Ohat),
$$

The form of the regularization $\mathcal{R}$ and the numerical method for solving the corresponding least-squares regression are specified by _solver_ objects in `opinf.lstsq`.
For example, `opinf.lstsq.L2Solver` implements the $L_{2}$ scalar regularizer

$$
\mathcal{R}(\Ohat)
= \lambda \|\Ohat\trp\|_{F}^{2},
\qquad \lambda > 0.
$$

Least-squares solver objects are passed to `fit()` using the `solver` keyword argument.
If `fit()` does not receive a `solver` object, no regularization is added ($\mathcal{R}(\Ohat) = \mathbf{0}$) and the regression is solved using [`scipy.linalg.lstsq()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html).

:::{eval-rst}
.. currentmodule:: opinf.lstsq

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    lstsq_size
    PlainSolver
:::

## Tikhonov Regularization

For $\mathcal{R}\equiv 0$ and a few other common choices of $\mathcal{R}$, the OpInf learning problem is _linear_ and can be solved explicitly.

:::{dropdown} $\mathcal{R} \equiv 0$
If there is no regularization, then the solution to the linear least-squares problem is given by the _normal equations_:

$$
\Ohat\trp
= (\D\trp\D)^{-1}\D\trp\mathbf{R}\trp.
$$
:::

:::{dropdown} $\mathcal{R}(\Ohat) = ||\lambda\Ohat||_{F}^{2}$
This choice of regularization is called the $L_{2}$ regularizer, a specific type of Tikhonov regularizer.
The solution is given by the modified normal equations

$$
\Ohat\trp
= (\D\trp\D + \lambda\I)^{-1}\D\trp\mathbf{R}\trp.
$$

Pass a positive scalar ($\lambda$) as the `regularizer` argument in `fit()` to use this regularization.
:::

:::{eval-rst}
.. currentmodule:: opinf.lstsq

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    L2Solver
    L2DecoupledSolver
    TikhonovSolver
    TikhonovDecoupledSolver
    TotalLeastSquaresSolver
:::
-->
