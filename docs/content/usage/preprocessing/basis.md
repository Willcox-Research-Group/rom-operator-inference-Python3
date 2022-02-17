(sec-basis-computation)=
# Basis Computation

The purpose of learning a reduced-order model is to achieve a computational speedup, which is a result of the reducing the dimension of the state $\mathbf{q}(t)\in\mathbb{R}^{n}$ to $r \ll n$.
This is accomplished by introducing the low-dimensional approximation

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

We call $\mathbf{V}_{r} \in \mathbb{R}^{n \times r}$ the _basis matrix_ and typically require that it has orthonormal columns.
The basis matrix is the link between the high-dimensional state space of the full-order model and the low-dimensional state space of the reduced-order model.

:::{margin}
```{note}
The matrix $\mathbf{V}_{r}\mathbf{V}_{r}^{\top} \in \mathbb{R}^{n \times n}$ is the orthogonal projector to the $r$-dimensional span of the columns of $\mathbf{V}_{r}$.
This means that $\mathbf{V}_{r}\mathbf{V}_{r}^{\top}\mathbf{q}(t) = \mathbf{V}_{r}\widehat{\mathbf{q}}(t)$ is the best approximation to $\mathbf{q}(t)$ which can be represented as {eq}`eq-opinf-basis-def`.
```
:::

:::{image} ../../../images/basis-projection.svg
:align: center
:width: 80 %
:::

## Proper Orthogonal Decomposition

Any orthonormal basis may be used for $\mathbf{V}_{r}$, but we advocate using the [proper orthogonal decomposition](https://en.wikipedia.org/wiki/Proper_orthogonal_decomposition) (POD), also referred to as the SVD or PCA.
The POD basis consists of the first $r < n$ left singular vectors of the state snapshot matrix: if $\mathbf{Q} = \boldsymbol{\Phi}\boldsymbol{\Sigma}\boldsymbol{\Psi}^{\top}$ is the (thin) singular-value decomposition of $\mathbf{Q}$, then we set $\mathbf{V}_{r} = \boldsymbol{\Phi}_{:,:r}$.
The function `opinf.pre.pod_basis()` computes $\mathbf{V}_{r}$ this way and returns the associated singular values $\sigma_{1},\ldots,\sigma_{k} = \text{diag}(\boldsymbol{\Sigma})$.

## Choosing the Basis Size

The dimension $r$ is the number of basis vectors used in the low-dimensional representation {eq}`eq-basis-basis-def`.

```{note}
TODO: how the projection error is related to the singular values (survey).
```



## Selecting the Reduced Dimension
