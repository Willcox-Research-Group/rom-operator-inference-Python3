(sec-lstsq)=
# Least-Squares Solvers

:::{admonition} Coming Soon!
:class: attention
This page is under construction.
:::

The following [least-squares regression problem](subsec-opinf-regression) is at the heart of Operator Inference:

$$
\min_{\widehat{\mathbf{c}},\widehat{\mathbf{A}},\widehat{\mathbf{H}},\widehat{\mathbf{B}}}\sum_{j=0}^{k-1}\left\|
    \widehat{\mathbf{c}}
    + \widehat{\mathbf{A}}\widehat{\mathbf{q}}_{j}
    + \widehat{\mathbf{H}}[\widehat{\mathbf{q}}_{j} \otimes \widehat{\mathbf{q}}_{j}]
    + \widehat{\mathbf{B}}\mathbf{u}_{j}
    - \dot{\widehat{\mathbf{q}}}_{j}
\right\|_{2}^{2}
\\
= \min_{\widehat{\mathbf{O}}}\left\|
    \mathbf{D}\widehat{\mathbf{O}}^{\mathsf{T}} - \mathbf{R}^{\mathsf{T}}
\right\|_{F}^{2},
$$

where
- $\widehat{\mathbf{q}}_{j} = \mathbf{V}_{r}^{\mathsf{T}}\mathbf{q}(t_{j})$ is the state at time $t_{j}$ represented in the coordinates of the basis,
- $\dot{\widehat{\mathbf{q}}}_{j} = \frac{\textrm{d}}{\textrm{d}t}\mathbf{V}_{r}^{\mathsf{T}}\mathbf{q}\big|_{t=t_{j}}$ is the time derivative of the state at time $t_{j}$ in the coordinates of the basis,
- $\mathbf{u}_{j} = \mathbf{u}(t_j)$ is the input at time $t_{j}$,
- $\mathbf{D}$ is the _data matrix_ containing low-dimensional state data,
- $\widehat{\mathbf{O}}$ is the _operator matrix_ of unknown operators to be inferred, and
- $\mathbf{R}$ is the matrix of low-dimensional time derivative data.

We often need to add a _regularization term_ $\mathcal{R}(\widehat{\mathbf{O}})$ that penalizes the entries of the learned operators.
This promotes stability and accuracy in the learned reduced-order model by preventing overfitting.
The problem stated above then becomes

$$
\min_{\widehat{\mathbf{O}}}\left\|
    \mathbf{D}\widehat{\mathbf{O}}^{\mathsf{T}} - \mathbf{R}^{\mathsf{T}}
\right\|_{F}^{2} + \mathcal{R}(\widehat{\mathbf{O}}),
$$

The form of the regularization $\mathcal{R}$ and the numerical method for solving the corresponding least-squares regression are specified by _solver_ objects in `opinf.lstsq`.
For example, `opinf.lstsq.L2Solver` implements the $L_{2}$ scalar regularizer

$$
\mathcal{R}(\widehat{\mathbf{O}})
= \lambda \|\widehat{\mathbf{O}}^{\mathsf{T}}\|_{F}^{2},
\qquad \lambda > 0.
$$

Least-squares solver objects are passed to `fit()` using the `solver` keyword argument.
If `fit()` does not receive a `solver` object, no regularization is added ($\mathcal{R}(\widehat{\mathbf{O}}) = \mathbf{0}$) and the regression is solved using [`scipy.linalg.lstsq()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html).

## Tikhonov Regularization

For $\mathcal{R}\equiv 0$ and a few other common choices of $\mathcal{R}$, the OpInf learning problem is _linear_ and can be solved explicitly.

:::{dropdown} $\mathcal{R} \equiv 0$
If there is no regularization, then the solution to the linear least-squares problem is given by the _normal equations_:

$$
\widehat{\mathbf{O}}^{\mathsf{T}}
= (\mathbf{D}^{\mathsf{T}}\mathbf{D})^{-1}\mathbf{D}^{\mathsf{T}}\mathbf{R}^{\mathsf{T}}.
$$
:::

:::{dropdown} $\mathcal{R}(\widehat{\mathbf{O}}) = ||\lambda\widehat{\mathbf{O}}||_{F}^{2}$
This choice of regularization is called the $L_{2}$ regularizer, a specific type of Tikhonov regularizer.
The solution is given by the modified normal equations

$$
\widehat{\mathbf{O}}^{\mathsf{T}}
= (\mathbf{D}^{\mathsf{T}}\mathbf{D} + \lambda\mathbf{I})^{-1}\mathbf{D}^{\mathsf{T}}\mathbf{R}^{\mathsf{T}}.
$$

Pass a positive scalar ($\lambda$) as the `regularizer` argument in `fit()` to use this regularization.
:::
