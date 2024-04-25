(sec-guide-dimensionality)=
# Dimensionality Reduction

The purpose of learning a reduced-order model is to achieve a computational speedup, which is a result of reducing the dimension of the state $\q(t)\in\RR^{n}$ from $n$ to $r \ll n$.
This is accomplished by introducing a low-dimensional approximation $\q(t) \approx \boldsymbol{\Gamma}(\qhat(t))$, where $\qhat(t)\in\RR^{r}$.
This page discusses linear choices for $\boldsymbol{\Gamma}$.

:::{warning}
If lifting or data normalization is used to preprocess the raw snapshot data, the dimensionality reduction techniques discussed here should target the processed snapshots, _not_ the raw snapshots.
On this page, $\q(t)$ represents state snapshots that have already been preprocessed.
:::

## Linear Bases

Most often, we use a linear representation $\boldsymbol{\Gamma}(\qhat) = \Vr\qhat$, that is,

$$
    \q(t)
    \approx \Vr \qhat(t)
    = \sum_{i=1}^{r}\v_{i}\hat{q}_{i}(t),
$$ (eq-basis-basis-def)

where

$$
    \Vr
    = \left[\begin{array}{ccc}
        & & \\
        \v_{1} & \cdots & \v_{r}
        \\ & &
    \end{array}\right] \in \RR^{n \times r},
    \qquad
    \qhat
    = \left[\begin{array}{c}
        \hat{q}_{1}(t) \\ \vdots \\ \hat{q}_{r}(t)
    \end{array}\right] \in \RR^{r}.
$$

The matrix $\Vr \in \RR^{n \times r}$ is called the _basis matrix_.
We typically require that it has orthonormal columns, i.e., $\mathbf{V}\trp\mathbf{V} = \I \in \RR^{r \times r}$, the identity matrix.
Through {eq}`eq-basis-basis-def`, the basis matrix is the link between the high-dimensional state space of the full-order model and the low-dimensional state space of the reduced-order model.

:::{image} ../../images/basis-projection.svg
:align: center
:width: 80 %
:::

The [**basis.LinearBasis**](opinf.basis.LinearBasis) class implements this type of basis.

:::::{note}
If $\Vr\in\RR^{n\times r}$ has orthogonal columns, the appropriate compression operator is $\boldsymbol{\Gamma}^{*}(\q) = \Vr\trp\q$.

:::{dropdown} Proof
The optimal compression operator is defined by

$$
    \boldsymbol{\Gamma}^{*}(\q)
    = \underset{\qhat\in\RR^{r}}{\textrm{arg min}}\left\|
        \q - \boldsymbol{\boldsymbol{\Gamma}}(\qhat)
    \right\|
    = \underset{\qhat\in\RR^{r}}{\textrm{arg min}}\left\|
        \q - \Vr\qhat
    \right\|.
$$

This is a linear least-squares problem; the solution is given by the Normal Equations

$$
    \Vr\trp\Vr\qhat
    = \Vr\trp\q,
$$

which simplifies to $\qhat = \Vr\trp\q$ since $\Vr\trp\Vr = \I$.
:::

Because $\boldsymbol{\Gamma}(\boldsymbol{\Gamma}^{*}(\q)) = \Vr\Vr\trp\q$, the matrix $\Vr\Vr\trp \in \RR^{n \times n}$ is called the _orthogonal projector_ onto $\operatorname{range}(\Vr)$ (the $r$-dimensional subspace of $\RR^{n}$ spanned by the columns of $\Vr$).
The projection error of $\q$ induced by $\Vr$ is

$$
    \left\|\q - \Vr\Vr\trp\q\right\|_{2}
    = \left\|(\I - \Vr\Vr\trp)\q\right\|_{2}.
$$

::::{grid}
:gutter: 3
:margin: 2 2 0 0

:::{grid-item-card}
`compress(state)`
^^^
$\q \to \Vr\trp\q$
:::

:::{grid-item-card}
`decompress(state_)`
^^^
$\qhat \to \Vr\qhat$
:::

:::{grid-item-card}
`project(state))`
^^^
$\q\to\Vr\Vr\trp\q$.
:::
::::

Orthogonal bases also enjoy the property that `compress(decompress(q_)) = q_`:

$$
    \boldsymbol{\Gamma}^{*}(\boldsymbol{\Gamma}(\qhat))
    = \Vr\trp\Vr\qhat
    = \qhat.
$$

If $\Vr\in\RR^{n\times r}$ does _not_ have orthogonal columns, the appropriate compression operator is $\boldsymbol{\Gamma}^{*}(\q) = \Vr^{\dagger}\q$, where $\Vr^{\dagger}$ is the [Moore-Penrose pseudoinverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) of $\Vr$.
:::::

(subsec-pod)=
## Proper Orthogonal Decomposition

Any low-dimensional representation or choice of basis can be used with OpInf, but for most problems we suggest using the [proper orthogonal decomposition](https://en.wikipedia.org/wiki/Proper_orthogonal_decomposition) (POD), a linear basis that is closely related to the [singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) (SVD) and [principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) (PCA).
The POD basis consists of the first $r < n$ left singular vectors of the state snapshot matrix: if $\Q = \boldsymbol{\Phi}\boldsymbol{\Sigma}\boldsymbol{\Psi}\trp$ is the (thin) SVD of $\Q$, then we set $\Vr = \boldsymbol{\Phi}_{:,:r}$.
The function [**basis.pod_basis()**](opinf.basis.pod_basis) and the class [**basis.PODBasis**](opinf.basis.PODBasis) implement these approaches.

### Computation Strategies

For moderately sized problems, the full SVD of $\Q$ can be computed.
This method may be inefficient for very large snapshot matrices $\Q$; in such cases, the principal left singular vectors of $\Q$ can be efficiently approximated with a randomized approach {cite}`halko2011rnla`. This method gives fast results at the cost of some accuracy.

::::{margin}
:::{note}
If `r` is not specified, the number of singular values `basis.pod_basis()` returns is always $\min\{n,k\}$.
When `r` is specified, the number of returned singular values depends on the `mode`:
- `"dense"` (default): all $\min\{n,k\}$ singular values.
- `"randomized"`: only the first $r$ singular values.
:::
::::

The `mode` keyword argument specifies the strategy for computing the basis:
- `"dense"` (default): Use [**scipy.linalg.svd()**](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html) to compute the full SVD.
- `"randomized"`: Use [**sklearn.utils.extmath.randomized_svd()**](https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html) to compute the randomized SVD.

Additional parameters for the SVD solver can be included with the `options` keyword argument.
For example, the following call tells [**scipy.linalg.svd()**](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html) to use a non-default solver.

```python
>>> opinf.basis.pod_basis(states, r=10, mode="dense",
...                       options=dict(lapack_driver="gesvd"))
```

(subsec-basis-size)=
### Choosing the Basis Size

::::{margin}
:::{tip}
Choosing $r$ carefully is important: if $r$ is too small, the basis may not be expressive enough to capture the features of the solution, but as $r$ increases, so does the computational cost of solving the reduced-order model.
:::
::::

The dimension $r$ is the number of basis vectors used in the low-dimensional representation {eq}`eq-basis-basis-def`, so it defines the dimension of the reduced-order model.
For POD, we typically choose $r$ based on the decay of the singular values of the snapshot data.
If $\Q = \boldsymbol{\Phi}\boldsymbol{\Sigma}\boldsymbol{\Psi}\trp$, the singular values are the diagonal entries of $\boldsymbol{\Sigma}$, i.e., $\sigma_{1},\ldots,\sigma_{\ell} = \textrm{diag}(\boldsymbol{\Sigma})$, $i = 1, 2, \ldots, \ell$, where $\ell = \min\{n,k\}$.
Some common singular value-based criteria are given below.
In these criteria, $0 < \varepsilon \ll 1$ is a small user-defined tolerance (e.g., $10^{-6}$), and $0 < \kappa < 1$ is a user-defined tolerance that is close to 1 (e.g., $0.9999$).

- **Singular value magnitude.** Choose the smallest integer $r$ such that $\sigma_{i} > \varepsilon$ for $i = 1,\ldots, r$, where $\varepsilon$ is a (small) user-determined tolerance.
- **Cumulative energy.** Choose the smallest integer $r$ such that the $\mathcal{E}_{r}(\Q) > \kappa$, where
$
    \mathcal{E}_{r}(\Q)
    = \frac{\sum_{j=1}^{r}\sigma_{j}^{2}}{\sum_{j=1}^{\ell}\sigma_{j}^{2}}.
$
The value $\mathcal{E}_{r}(\Q)$ is called the _cumulative energy_, which represents how much "energy" in the system is captured by the first $r$ POD modes.
- **Residual energy**. Choose the smallest integer $r$ such that $1 - \mathcal{E}_{r}(\Q) = \frac{\sum_{j =r+1}^{\ell}\sigma_{j}^{2}}{\sum_{j=1}^{\ell}\sigma_{j}^{2}} < \varepsilon$. This is equivalent to the previous strategy with $\varepsilon = 1 - \kappa$.
The value $1 - \mathcal{E}_{r}(\Q)$ is called the _residual energy_, which represents how much "energy" in the system is neglected by discarding all but the first $r$ POD modes.
- **Projection error**. Choose the smallest integer $r$ such that $\mathcal{P}_{r}(\Q) < \varepsilon$, where
$
    \mathcal{P}_{r}(\Q)
    = \frac{\|\Q - \Vr\Vr\trp\Q\|_{F}}{\|\Q\|_{F}}.
$
The value $\mathcal{P}_{r}(\Q)$ is called the _projection error_, which quantifies how well $\Q$ can be represented in the column span of $\Vr$.

::::{margin}
:::{tip}
These criteria are based on all $\ell = \min\{n,k\}$ singular values of $\Q$.
However, if $\sigma_{J} \lesssim \epsilon_{\text{machine}}$, then the relative contribution of $\sigma_{j}$ to $\mathcal{E}_{r}(\Q)$ is small for any $j \ge J$, i.e., $\sigma_{j}^{2} \approx 0$.
In this case, the cumulative or residual energies can be estimated using only the first $J$ singular values.
This is useful, for example, when the POD basis is computed with a randomized algorithm that iteratively computes singular values and vectors.
:::
::::

The following functions help measure these criteria.

| Function | Description |
| :------- | :---------- |
| [**basis.svdval_decay()**](opinf.basis.svdval_decay) | Plot the singular value decay and counts the number of singular values that are larger than a given $\varepsilon$. |
| [**basis.cumulative_energy()**](opinf.basis.cumulative_energy) | Plot the cumulative energy as a function of $r$. |
| [**basis.residual_energy()**](opinf.basis.residual_energy) | Plot the residual energy as a function of $r$. |
<!-- | [**basis.projection_error()**](opinf.basis.projection_error) | Calculate projection error. | -->

::::{note}
Each of the singular value-based selection criteria are related.
In particular, the squared projection error of the snapshot matrix is equal to the residual energy:

$$
    \mathcal{P}_{r}(\Q)^{2}
    =
    \frac{\|\Q - \Vr\Vr\trp\Q\|_{F}^{2}}{\|\Q\|_{F}^{2}}
    = \displaystyle\frac{\sum_{j = r + 1}^{\ell}\sigma_{j}^{2}}{\sum_{j=1}^{\ell}\sigma_{j}^{2}}
    = \mathcal{E}_{r}(\Q).
$$

:::{dropdown} Proof
Let $\Q = \boldsymbol{\Phi}\boldsymbol{\Sigma}\boldsymbol{\Psi}\trp$ be the thin singular value decomposition of $\Q\in\RR^{n \times k}$,
meaning $\boldsymbol{\Phi}\in\RR^{n\times \ell}$, $\boldsymbol{\Sigma}\in\RR^{\ell \times \ell}$, and $\boldsymbol{\Psi}\in\RR^{k\times \ell}$,
with $\boldsymbol{\Phi}\trp\boldsymbol{\Phi} = \boldsymbol{\Psi}\trp\boldsymbol{\Psi} = \I$.
Splitting the decomposition into "first $r$ modes" and "remaining modes" gives

$$
    \Q
    = \left[\begin{array}{cc}
        \boldsymbol{\Phi}_{r} & \boldsymbol{\Phi}_{\perp}
    \end{array}\right]
    \left[\begin{array}{cc}
        \boldsymbol{\Sigma}_{r} & \\
        & \boldsymbol{\Sigma}_{\perp}
    \end{array}\right]
    \left[\begin{array}{c}
        \boldsymbol{\Psi}_{r}\trp \\
        \boldsymbol{\Psi}_{\perp}\trp
    \end{array}\right]
    = \underbrace{\boldsymbol{\Phi}_{r}\boldsymbol{\Sigma}_{r}\boldsymbol{\Psi}_{r}\trp}_{\Q_{r}} + \underbrace{\boldsymbol{\Phi}_{\perp}\boldsymbol{\Sigma}_{\perp}\boldsymbol{\Psi}_{\perp}\trp}_{\Q_{\perp}},
$$

where

\begin{align*}
    &\boldsymbol{\Phi}_{r}\in\RR^{n\times r},
    &
    &\boldsymbol{\Phi}_{\perp}\in\RR^{n\times (\ell - r)},
    &
    &\boldsymbol{\Phi}_{r}\trp\boldsymbol{\Phi}_{r}
    = \boldsymbol{\Psi}_{r}\trp\boldsymbol{\Psi}_{r}
    = \I,
    \\
    &\boldsymbol{\Sigma}_{r}\in\RR^{r\times r},
    &
    &\boldsymbol{\Sigma}_{\perp}\in\RR^{(\ell - r)\times (\ell - r)},
    &
    &\boldsymbol{\Phi}_{\perp}\trp\boldsymbol{\Phi}_{\perp}
    = \boldsymbol{\Psi}_{\perp}\trp\boldsymbol{\Psi}_{\perp}
    = \I,
    \\
    &\boldsymbol{\Psi}_{r}\in\RR^{k\times r},
    &
    &\boldsymbol{\Psi}_{\perp}\in\RR^{k\times (\ell - r)},
    &
    &\boldsymbol{\Phi}_{r}\trp\boldsymbol{\Phi}_{\perp}
    = \boldsymbol{\Psi}_{r}\trp\boldsymbol{\Psi}_{\perp}
    = \mathbf{0}.
\end{align*}

We have defined $\Vr = \boldsymbol{\Phi}_{r}$.
Using $\Vr\trp\boldsymbol{\Phi}_{r} = \Vr\trp\Vr = \I$
and $\Vr\trp\boldsymbol{\Phi}_{\perp} = \boldsymbol{\Phi}_{r}\boldsymbol{\Phi}_{\perp} = \mathbf{0}$,

\begin{align*}
    \Vr\Vr\trp\Q_{r}
    &= \Vr\Vr\trp\boldsymbol{\Phi}_{r}\boldsymbol{\Sigma}_{r}\boldsymbol{\Psi}_{r}\trp
    = \Vr\boldsymbol{\Sigma}_{r}\boldsymbol{\Psi}_{r}\trp
    = \Q_{r},
    \\
    \Vr\Vr\trp\Q_{\perp}
    &= \Vr\Vr\trp\boldsymbol{\Phi}_{\perp}\boldsymbol{\Sigma}_{\perp}\boldsymbol{\Psi}_{\perp}\trp
    = \mathbf{0}.
\end{align*}

That is, $\Vr\Vr\trp\Q = \Vr\Vr\trp(\Q_{r} + \Q_{\perp}) = \Q_{r}$.
It follows that

$$
    \Q - \Vr\Vr\trp\Q
    = \Q_{r} + \Q_{\perp} - \Q_{r}
    = \Q_{\perp}
    = \boldsymbol{\Phi}_{\perp}\boldsymbol{\Sigma}_{\perp}\boldsymbol{\Psi}_{\perp}\trp.
$$

Since $\boldsymbol{\Phi}_{\perp}$ and $\boldsymbol{\Psi}_{\perp}$ have orthonormal columns,

$$
    \left\|\boldsymbol{\Phi}_{\perp}\boldsymbol{\Sigma}_{\perp}\boldsymbol{\Psi}_{\perp}\trp\right\|_{F}^{2}
    = \left\|\boldsymbol{\Sigma}_{\perp}\right\|_{F}^{2}
    = \sum_{j=r + 1}^{\ell}\sigma_{j}^{2}.
$$

Putting it all together,

$$
    \frac{\|\Q - \Vr\Vr\trp\Q\|_{F}^{2}}{\|\Q\|_{F}^{2}}
    = \frac{\|\boldsymbol{\Sigma}_{\perp}\|_{F}^{2}}{\|\boldsymbol{\Sigma}\|_{F}^{2}}
    = \frac{\sum_{j=r + 1}^{\ell}\sigma_{j}^{2}}{\sum_{j=1}^{\ell}\sigma_{j}^{2}}.
$$

:::
::::
