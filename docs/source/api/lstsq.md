# `opinf.lstsq`

```{eval-rst}
.. automodule:: opinf.lstsq

.. currentmodule:: opinf.lstsq

.. autosummary::
   :toctree: _autosummaries
   :nosignatures:

   lstsq_size
   SolverTemplate
   PlainSolver
   L2Solver
   L2DecoupledSolver
   TikhonovSolver
   TikhonovDecoupledSolver
   TotalLeastSquaresSolver
```

EXAMPLE DATA

## Least-squares Operator Inference Problems

The following [least-squares regression problem](subsec-opinf-regression) is at the heart of Operator Inference:

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
    \D\Ohat\trp - \Z\trp
\right\|_{F}^{2},
$$

where

- $\qhat_{j} = \Vr\trp\q(t_{j})$ is the state at time $t_{j}$ represented in the coordinates of the basis,
- $\dot{\qhat}_{j} = \ddt\Vr\trp\q\big|_{t=t_{j}}$ is the time derivative of the state at time $t_{j}$ in the coordinates of the basis,
- $\u_{j} = \u(t_j)$ is the input at time $t_{j}$,
- $\D$ is the *data matrix* containing low-dimensional state data,
- $\Ohat$ is the *operator matrix* of unknown operators to be inferred, and
- $\Z$ is the matrix of low-dimensional time derivative data.

This module defines classes for solving the above problem, or related problems with regularization and/or constraints, given the data matrices $\D$ and $\Z$.

Solver objects are passed to the constructor of {mod}`opinf.models` classes.

## Default Solver

The {class}`PlainSolver` class solves EQREF without any additional terms.

We often need to add a *regularization term* $\mathcal{R}(\Ohat)$ that penalizes the entries of the learned operators.
This promotes stability and accuracy in the learned reduced-order model by preventing overfitting.
The problem stated above then becomes

$$
\begin{aligned}
    \min_{\Ohat}\left\|
        \D\Ohat\trp - \mathbf{R}\trp
    \right\|_{F}^{2} + \mathcal{R}(\Ohat),
\end{aligned}
$$

The form of the regularization $\mathcal{R}$ and the numerical method for solving the corresponding least-squares regression are specified by _solver_ objects in `opinf.lstsq`.
For example, `opinf.lstsq.L2Solver` implements the $L_{2}$ scalar regularizer

$$
\begin{aligned}
    \mathcal{R}(\Ohat)
    = \lambda \|\Ohat\trp\|_{F}^{2},
    \qquad \lambda > 0.
\end{aligned}
$$

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

## Total Least-Squares

If you want to use the total least-squares solver use the following class.

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    TotalLeastSquaresSolver
