# `opinf.post`

:::{eval-rst}
.. automodule:: opinf.post
:::

<!-- :::{important}
Undo preprocessing before you do postprocessing.
Reduced-order model outputs need to be translated back to the state space of the original system of interest.
Raw -> Shifted -> Scaled -> Projected -> Solve
::: -->


<!--The functions listed below compute the absolute and relative errors in different norms.

| Function | Norm |
| :------- | :--- |
| `post.frobenius_error()` | [Frobenius norm](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm) |
| `post.lp_error()` | [$\ell^{p}$ norm](https://en.wikipedia.org/wiki/Lp_space#The_p-norm_in_finite_dimensions) (columnwise) |
| `post.Lp_error()` | [$L^{p}$ norm](https://en.wikipedia.org/wiki/Lp_space#Lp_spaces) |

## Old API

In the following documentation we denote $q_{ij} = [\mathbf{Q}]$ for the entries of a matrix $\mathbf{Q} \in \mathbb{R}^{n\times k}$ and $q_{i} = [\mathbf{q}]_{i}$ for the entries of a vector $q$.

**`post.frobenius_error(Qtrue, Qapprox)`**: Compute the absolute and relative Frobenius-norm errors between snapshot sets `Qtrue` and `Qapprox`.
The [Frobenius matrix norm](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm) is defined by

$$
    \|\mathbf{Q}\|_{F}
    = \sqrt{\text{trace}(\mathbf{Q}^{\mathsf{T}}\mathbf{Q})}
    = \left(\sum_{i=1}^{n}\sum_{j=1}^{k}|q_{ij}|^2\right)^{1/2}.
$$

**`post.lp_error(Qtrue, Qapprox, p=2, normalize=False)`**: Compute the absolute and relative $\ell^{p}$-norm errors between snapshot sets `Qtrue` and `Qapprox`.
The [$\ell^{p}$ norm](https://en.wikipedia.org/wiki/Lp_space#The_p-norm_in_finite_dimensions) is defined by

\begin{align*}
    \|\mathbf{q}\|_{p}
    = \begin{cases}
    \left(\displaystyle\sum_{i=1}^{n}|q_i|^p\right)^{1/p} & p < \infty,
    \\ & \\
    \underset{i=1,\ldots,n}{\text{sup}}|q_i| & p = \infty.
    \end{cases}
\end{align*}

With $p = 2$ this is the usual _Euclidean norm_.
The errors are calculated for each pair of columns of `Qtrue` and `Qapprox`.
If `normalize=True`, then the _normalized absolute error_ is computed instead of the relative error:

$$
    \text{norm\_abs\_error}_j
    = \frac{\|\mathbf{q}_j - \mathbf{y}_j\|_{p}}{\max_{l=1,\ldots,k}\|\mathbf{q}_l\|_{p}},
    \quad
    j = 1,\ldots,k.
$$

**`post.Lp_error(Qtrue, Qapprox, t=None, p=2)`**: Approximate the absolute and relative $L^{p}$-norm errors between snapshot sets `Qtrue` and `Qapprox` corresponding to times `t`.
The [$L^{p}$ norm](https://en.wikipedia.org/wiki/Lp_space#Lp_spaces) for vector-valued functions is defined by

$$
    \|\mathbf{q}(\cdot)\|_{L^p([a,b])}
    = \begin{cases}
    \left(\displaystyle\int_{a}^{b}\|\mathbf{q}(t)\|_{p}^p\:dt\right)^{1/p} & p < \infty,
    \\ & \\
    \sup_{t\in[a,b]}\|\mathbf{q}(t)\|_{\infty} & p = \infty.
    \end{cases}
$$

For finite _p_, the integrals are approximated by the trapezoidal rule:

$$
    \int_{a}^{b}\|\mathbf{q}(t)\|_{p}^{p}\:dt
    \approx \delta t\left(
        \frac{1}{2}\|\mathbf{q}(t_0)\|_{p}^p
        + \sum_{j=1}^{k-2}\|\mathbf{q}(t_j)\|_{p}^p
        + \frac{1}{2}\|\mathbf{q}(t_{k-1})\|_{p}^p
    \right),
    \\
    a = t_0 < t_1 < \cdots < t_{k-1} = b.
$$

The `t` argument can be omitted if _p_ is infinity (`p = np.inf`). -->
