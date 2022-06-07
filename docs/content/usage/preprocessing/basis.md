(sec-basis-computation)=
# Basis Computation

The purpose of learning a reduced-order model is to achieve a computational speedup, which is a result of reducing the dimension of the state $\mathbf{q}(t)\in\mathbb{R}^{n}$ from $n$ to $r \ll n$.
This is accomplished by introducing a low-dimensional approximation, for example

$$
    \mathbf{q}(t)
    \approx \mathbf{V}_{r} \widehat{\mathbf{q}}(t)
    = \sum_{i=1}^{r}\mathbf{v}_{i}\hat{q}_{i}(t),
$$ (eq-basis-basis-def)

where

$$
    \mathbf{V}_{r}
    = \left[\begin{array}{ccc}
        & & \\
        \mathbf{v}_{1} & \cdots & \mathbf{v}_{r}
        \\ & &
    \end{array}\right] \in \mathbb{R}^{n \times r},
    \qquad
    \widehat{\mathbf{q}}
    = \left[\begin{array}{c}
        \hat{q}_{1}(t) \\ \vdots \\ \hat{q}_{r}(t)
    \end{array}\right] \in \mathbb{R}^{r}.
$$

We call $\mathbf{V}_{r} \in \mathbb{R}^{n \times r}$ the _basis matrix_ and typically require that it has orthonormal columns, i.e., $\mathbf{V}^{\mathsf{T}}\mathbf{V} = \mathbf{I} \in \mathbb{R}^{r \times r}$, the identity matrix.
Through {eq}`eq-basis-basis-def`, the basis matrix is the link between the high-dimensional state space of the full-order model and the low-dimensional state space of the reduced-order model.

:::{margin}
```{note}
The matrix $\mathbf{V}_{r}\mathbf{V}_{r}^{\mathsf{T}} \in \mathbb{R}^{n \times n}$ is the orthogonal projector to the $r$-dimensional span of the columns of $\mathbf{V}_{r}$.
This means that $\mathbf{V}_{r}\mathbf{V}_{r}^{\mathsf{T}}\mathbf{q}(t) = \mathbf{V}_{r}\widehat{\mathbf{q}}(t)$ is the best approximation to $\mathbf{q}(t)$ which can be represented as {eq}`eq-opinf-basis-def`.
```
:::

:::{image} ../../../images/basis-projection.svg
:align: center
:width: 80 %
:::

(subsec-pod)=
## Proper Orthogonal Decomposition

Any low-dimensional representation or choice of basis can be used with OpInf, but for most problems we suggest using the [proper orthogonal decomposition](https://en.wikipedia.org/wiki/Proper_orthogonal_decomposition) (POD), which is closely related to the [singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) (SVD) and [principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) (PCA).
The POD basis consists of the first $r < n$ left singular vectors of the state snapshot matrix: if $\mathbf{Q} = \boldsymbol{\Phi}\boldsymbol{\Sigma}\boldsymbol{\Psi}^{\mathsf{T}}$ is the (thin) SVD of $\mathbf{Q}$, then we set $\mathbf{V}_{r} = \boldsymbol{\Phi}_{:,:r}$.

The function `opinf.pre.pod_basis()` computes $\mathbf{V}_{r}$ and the associated singular values, i.e., the diagonal entries of $\boldsymbol{\Sigma}$.

<!-- TODO: example turn this into a notebook: load Q and show what these functions do. Plot some of the basis vectors. -->

### POD Computation Strategies

The `mode` keyword argument of `opinf.pre.pod_basis()` specifies the strategy to use for computing the basis.
- `"dense"` (default): Use [`scipy.linalg.svd()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html) to compute the full SVD. This method may be inefficient for very large snapshot matrices.
- `"sparse"`: Use [`scipy.sparse.linalg.svds()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html) to compute only the first $r$ left singular vectors with [ARPACK](https://www.caam.rice.edu/software/ARPACK/). This method is most effective for [sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html) snapshot matrices.
- `"randomized"`: Use [`sklearn.utils.extmath.randomized_svd()`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html) to compute an approximation of only the first $r$ left singular vectors with a randomized approach {cite}`HMT2011randomizedNLA`. This method gives fast results at the cost of some accuracy, and it is a good choice for large matrices that `scipy.linalg.svd()` cannot handle well.

:::{note}
If `r` is not specified, the number of singular values `opinf.pre.pod_basis()` returns is always $\min\{n,k\}$.
When `r` is specified, the number of returned singular values depends on the `mode`:
- `"dense"` (default): all $\min\{n,k\}$ singular values.
- `"sparse"`: only the first $r$ singular values.
- `"randomized"`: only the first $r$ singular values.
:::

Additional parameters for the SVD solver can be included with the `options` keyword argument.
For example, the following call tells `scipy.linalg.svd()` to use a non-default solver.

```python
>>> opinf.pre.pod_basis(states, r=10, mode="dense",
...                     options=dict(lapack_driver="gesvd"))
```

<!-- TODO: weighted POD (in space and in time). -->

(subsec-basis-size)=
## Choosing the Basis Size

::::{margin}
:::{tip}
Choosing $r$ carefully is important: if $r$ is too small, the basis may not be expressive enough to capture the features of the solution; as $r$ increases, so does the computational cost of solving the reduced-order model.
:::
::::

The dimension $r$ is the number of basis vectors used in the low-dimensional representation {eq}`eq-basis-basis-def`, so it defines the dimension of the reduced-order model.
For POD, we typically choose $r$ based on the decay of the singular values of the snapshot data.
If $\mathbf{Q} = \boldsymbol{\Phi}\boldsymbol{\Sigma}\boldsymbol{\Psi}^{\mathsf{T}}$, the singular values are the diagonal entries of $\boldsymbol{\Sigma}$, i.e., $\sigma_{1},\ldots,\sigma_{\ell} = \textrm{diag}(\boldsymbol{\Sigma})$, $i = 1, 2, \ldots, \ell$, where $\ell = \min\{n,k\}$.
Some common singular value-based criteria are given below.
In these criteria, $0 < \varepsilon \ll 1$ is a small user-defined tolerance (e.g., $10^{-6}$), and $0 < \kappa < 1$ is a user-defined tolerance that is close to 1 (e.g., $0.9999$).

- **Singular value magnitude.** Choose the smallest integer $r$ such that $\sigma_{i} > \varepsilon$ for $i = 1,\ldots, r$, where $\varepsilon$ is a (small) user-determined tolerance.
The function `opinf.pre.svdval_decay()` plots the singular value decay and counts the number of singular values that are larger than a given $\varepsilon$.

- **Cumulative energy.** Choose the smallest integer $r$ such that the $\mathcal{E}_{r}(\mathbf{Q}) > \kappa$, where
$
    \mathcal{E}_{r}(\mathbf{Q})
    = \frac{\sum_{j=1}^{r}\sigma_{j}^{2}}{\sum_{j=1}^{\ell}\sigma_{j}^{2}}.
$
We call $\mathcal{E}_{r}(\mathbf{Q})$ the _cumulative energy_, which represents how much "energy" in the system is captured by the first $r$ POD modes.
The function `opinf.pre.cumulative_energy()` plots the cumulative energy as a function of $r$.

- **Residual energy**. Choose the smallest integer $r$ such that $1 - \mathcal{E}_{r}(\mathbf{Q}) = \frac{\sum_{j =r+1}^{\ell}\sigma_{j}^{2}}{\sum_{j=1}^{\ell}\sigma_{j}^{2}} < \varepsilon$. This is equivalent to the previous strategy with $\varepsilon = 1 - \kappa$.
We call $1 - \mathcal{E}_{r}(\mathbf{Q})$ the _residual energy_, which represents how much "energy" in the system is neglected by discarding all but the first $r$ POD modes.
The function `opinf.pre.residual_energy()` plots the cumulative energy as a function of $r$.

- **Projection error**. Choose the smallest integer $r$ such that $\mathcal{P}_{r}(\mathbf{Q}) < \varepsilon$, where
$
    \mathcal{P}_{r}(\mathbf{Q})
    = \frac{\|\mathbf{Q} - \mathbf{V}_{r}\mathbf{V}_{r}^{\mathsf{T}}\mathbf{Q}\|_{F}}{\|\mathbf{Q}\|_{F}}.
$
We call $\mathcal{P}_{r}(\mathbf{Q})$ the _projection error_, which quantifies how well $\mathbf{Q}$ can be represented in the column span of $\mathbf{V}_{r}$.
The function `opinf.pre.projection_error()` calculates the projection error.

:::{tip}
These criteria are based on all $\ell = \min\{n,k\}$ singular values of $\mathbf{Q}$.
However, if $\sigma_{J} \lessim \epsilon_{\text{machine}}$, then the relative contribution of $\sigma_{j}$ to $\mathcal{E}_{r}(\mathbf{Q})$ is small for any $j \ge J$, i.e., $\sigma_{j}^{2} \approx 0$.
In this case, the cumulative or residual energies can be estimated using only the first $J$ singular values, which is useful when the POD basis must be computed in randomized fashion.
:::

<!-- TODO: demonstrate this. -->

::::{note}
Each of the singular value-based selection criteria are related.
In particular, the squared projection error of the snapshot matrix is equal to the residual energy:

$$
    \mathcal{P}_{r}(\mathbf{Q})^{2}
    =
    \frac{\|\mathbf{Q} - \mathbf{V}_{r}\mathbf{V}_{r}^{\mathsf{T}}\mathbf{Q}\|_{F}^{2}}{\|\mathbf{Q}\|_{F}^{2}}
    = \displaystyle\frac{\sum_{j = r + 1}^{\ell}\sigma_{j}^{2}}{\sum_{j=1}^{\ell}\sigma_{j}^{2}}
    = \mathcal{E}_{r}(\mathbf{Q}).
$$

:::{dropdown} Proof
Let $\mathbf{Q} = \boldsymbol{\Phi}\boldsymbol{\Sigma}\boldsymbol{\Psi}^{\mathsf{T}}$ be the thin singular value decomposition of $\mathbf{Q}\in\mathbb{R}^{n \times k}$,
meaning $\boldsymbol{\Phi}\in\mathbb{R}^{n\times \ell}$, $\boldsymbol{\Sigma}\in\mathbb{R}^{\ell \times \ell}$, and $\boldsymbol{\Psi}\in\mathbb{R}^{k\times \ell}$,
with $\boldsymbol{\Phi}^{\mathsf{T}}\boldsymbol{\Phi} = \boldsymbol{\Psi}^{\mathsf{T}}\boldsymbol{\Psi} = \mathbf{I}$.
Splitting the decomposition into "first $r$ modes" and "remaining modes" gives

$$
    \mathbf{Q}
    = \left[\begin{array}{cc}
        \boldsymbol{\Phi}_{r} & \boldsymbol{\Phi}_{\perp}
    \end{array}\right]
    \left[\begin{array}{cc}
        \boldsymbol{\Sigma}_{r} & \\
        & \boldsymbol{\Sigma}_{\perp}
    \end{array}\right]
    \left[\begin{array}{c}
        \boldsymbol{\Psi}_{r}^{\mathsf{T}} \\
        \boldsymbol{\Psi}_{\perp}^{\mathsf{T}}
    \end{array}\right]
    = \underbrace{\boldsymbol{\Phi}_{r}\boldsymbol{\Sigma}_{r}\boldsymbol{\Psi}_{r}^{\mathsf{T}}}_{\mathbf{Q}_{r}} + \underbrace{\boldsymbol{\Phi}_{\perp}\boldsymbol{\Sigma}_{\perp}\boldsymbol{\Psi}_{\perp}^{\mathsf{T}}}_{\mathbf{Q}_{\perp}},
$$

where

\begin{align*}
    &\boldsymbol{\Phi}_{r}\in\mathbb{R}^{n\times r},
    &
    &\boldsymbol{\Phi}_{\perp}\in\mathbb{R}^{n\times (\ell - r)},
    &
    &\boldsymbol{\Phi}_{r}^{\mathsf{T}}\boldsymbol{\Phi}_{r}
    = \boldsymbol{\Psi}_{r}^{\mathsf{T}}\boldsymbol{\Psi}_{r}
    = \mathbf{I},
    \\
    &\boldsymbol{\Sigma}_{r}\in\mathbb{R}^{r\times r},
    &
    &\boldsymbol{\Sigma}_{\perp}\in\mathbb{R}^{(\ell - r)\times (\ell - r)},
    &
    &\boldsymbol{\Phi}_{\perp}^{\mathsf{T}}\boldsymbol{\Phi}_{\perp}
    = \boldsymbol{\Psi}_{\perp}^{\mathsf{T}}\boldsymbol{\Psi}_{\perp}
    = \mathbf{I},
    \\
    &\boldsymbol{\Psi}_{r}\in\mathbb{R}^{k\times r},
    &
    &\boldsymbol{\Psi}_{\perp}\in\mathbb{R}^{k\times (\ell - r)},
    &
    &\boldsymbol{\Phi}_{r}^{\mathsf{T}}\boldsymbol{\Phi}_{\perp}
    = \boldsymbol{\Psi}_{r}^{\mathsf{T}}\boldsymbol{\Psi}_{\perp}
    = \mathbf{0}.
\end{align*}

We have defined $\mathbf{V}_{r} = \boldsymbol{\Phi}_{r}$.
Using $\mathbf{V}_{r}^{\mathsf{T}}\boldsymbol{\Phi}_{r} = \mathbf{V}_{r}^{\mathsf{T}}\mathbf{V}_{r} = \mathbf{I}$
and $\mathbf{V}_{r}^{\mathsf{T}}\boldsymbol{\Phi}_{\perp} = \boldsymbol{\Phi}_{r}\boldsymbol{\Phi}_{\perp} = \mathbf{0}$,

\begin{align*}
    \mathbf{V}_{r}\mathbf{V}_{r}^{\mathsf{T}}\mathbf{Q}_{r}
    &= \mathbf{V}_{r}\mathbf{V}_{r}^{\mathsf{T}}\boldsymbol{\Phi}_{r}\boldsymbol{\Sigma}_{r}\boldsymbol{\Psi}_{r}^{\mathsf{T}}
    = \mathbf{V}_{r}\boldsymbol{\Sigma}_{r}\boldsymbol{\Psi}_{r}^{\mathsf{T}}
    = \mathbf{Q}_{r},
    \\
    \mathbf{V}_{r}\mathbf{V}_{r}^{\mathsf{T}}\mathbf{Q}_{\perp}
    &= \mathbf{V}_{r}\mathbf{V}_{r}^{\mathsf{T}}\boldsymbol{\Phi}_{\perp}\boldsymbol{\Sigma}_{\perp}\boldsymbol{\Psi}_{\perp}^{\mathsf{T}}
    = \mathbf{0}.
\end{align*}

That is, $\mathbf{V}_{r}\mathbf{V}_{r}^{\mathsf{T}}\mathbf{Q} = \mathbf{V}_{r}\mathbf{V}_{r}^{\mathsf{T}}(\mathbf{Q}_{r} + \mathbf{Q}_{\perp}) = \mathbf{Q}_{r}$.
It follows that

$$
    \mathbf{Q} - \mathbf{V}_{r}\mathbf{V}_{r}^{\mathsf{T}}\mathbf{Q}
    = \mathbf{Q}_{r} + \mathbf{Q}_{\perp} - \mathbf{Q}_{r}
    = \mathbf{Q}_{\perp}
    = \boldsymbol{\Phi}_{\perp}\boldsymbol{\Sigma}_{\perp}\boldsymbol{\Psi}_{\perp}^{\mathsf{T}}.
$$

Since $\boldsymbol{\Phi}_{\perp}$ and $\boldsymbol{\Psi}_{\perp}$ have orthonormal columns,

$$
    \left\|\boldsymbol{\Phi}_{\perp}\boldsymbol{\Sigma}_{\perp}\boldsymbol{\Psi}_{\perp}^{\mathsf{T}}\right\|_{F}^{2}
    = \left\|\boldsymbol{\Sigma}_{\perp}\right\|_{F}^{2}
    = \sum_{j=r + 1}^{\ell}\sigma_{j}^{2}.
$$

Putting it all together,

$$
    \frac{\|\mathbf{Q} - \mathbf{V}_{r}\mathbf{V}_{r}^{\mathsf{T}}\mathbf{Q}\|_{F}^{2}}{\|\mathbf{Q}\|_{F}^{2}}
    = \frac{\|\boldsymbol{\Sigma}_{\perp}\|_{F}^{2}}{\|\boldsymbol{\Sigma}\|_{F}^{2}}
    = \frac{\sum_{j=r + 1}^{\ell}\sigma_{j}^{2}}{\sum_{j=1}^{\ell}\sigma_{j}^{2}}.
$$

:::
::::
