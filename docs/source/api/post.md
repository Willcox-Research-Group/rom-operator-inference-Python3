# `opinf.post`

```{eval-rst}
.. automodule:: opinf.post
```

## Absolute and Relative Error

Given a norm $\|\cdot\|$, "true" data $\Q$, and an approximation $\breve{\Q}$ to $\Q$, the absolute and relative errors of the approximation $\breve{\Q}$ are defined as

$$
\begin{align*}
    e_{\text{absolute}}
    = \|\Q - \breve{\Q}\|,
    \qquad
    e_{\text{relative}}
    = \frac{e_{\text{absolute}}}{\|\Q\|}
    = \frac{\|\Q - \breve{\Q}\|}{\|\Q\|}.
\end{align*}
$$

In the context of this package, $\Q\in\RR^{n \times k}$ is typically a matrix whose $j$-th column is the true state vector at time $t_{j}$, and the approximation $\breve{\Q}\in\RR^{n\times k}$ is the corresponding matrix of reduced-order model solutions.

## Projection Error

The projection error is defined by the low-dimensional representation of the state, not the solution of a reduced-order model *per se*.
For a true state $\q \in \RR^{n}$, consider the low-dimensional (linear) approximation

$$
\begin{align*}
    \breve{\q} = \Vr\qhat,
\end{align*}
$$

where $\Vr\in\RR^{n\times r}$.
The projection error associated with this approximation is

$$
\begin{align*}
    \|\q - \breve{\q}\|
    = \|\q - \Vr\Vr\trp\q\|.
\end{align*}
$$

The following function computes this projection error, given a basis matrix $\Vr$.

```{eval-rst}
.. currentmodule: opinf.post

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    projection_error
```

Basis classes such as {class}`opinf.basis.PODBasis` also have a `projection_error()` method.

## Reduced-order Model Error

The following functions compute the error between a true state solution $\q(t) \in \RR^{n}$ of the system of interest and an approximation $\breve{\q}(t) \in \RR^{n}$ generated by a reduced-order model.
Each uses a different norm to measure the absolute and relative errors.

```{eval-rst}
.. currentmodule: opinf.post

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    frobenius_error
    lp_error
    Lp_error
```

<!-- :::{important}
Undo preprocessing before you do postprocessing.
Reduced-order model outputs need to be translated back to the state space of the original system of interest.
Raw -> Shifted/Scaled -> Compressed -> Solved
::: -->

<!--The functions listed below compute the absolute and relative errors in different norms.

| Function | Norm |
| :------- | :--- |
| `post.frobenius_error()` | [Frobenius norm](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm) |
| `post.lp_error()` | [$\ell^{p}$ norm](https://en.wikipedia.org/wiki/Lp_space#The_p-norm_in_finite_dimensions) (columnwise) |
| `post.Lp_error()` | [$L^{p}$ norm](https://en.wikipedia.org/wiki/Lp_space#Lp_spaces) |

## Old API

In the following documentation we denote $q_{ij} = [\Q]$ for the entries of a matrix $\Q \in \RR^{n\times k}$ and $q_{i} = [\q]_{i}$ for the entries of a vector $q$.

**`post.frobenius_error(Qtrue, Qapprox)`**: Compute the absolute and relative Frobenius-norm errors between snapshot sets `Qtrue` and `Qapprox`.
The [Frobenius matrix norm](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm) is defined by

$$
    \|\Q\|_{F}
    = \sqrt{\text{trace}(\Q\trp\Q)}
    = \left(\sum_{i=1}^{n}\sum_{j=1}^{k}|q_{ij}|^2\right)^{1/2}.
$$

**`post.lp_error(Qtrue, Qapprox, p=2, normalize=False)`**: Compute the absolute and relative $\ell^{p}$-norm errors between snapshot sets `Qtrue` and `Qapprox`.
The [$\ell^{p}$ norm](https://en.wikipedia.org/wiki/Lp_space#The_p-norm_in_finite_dimensions) is defined by

\begin{align*}
    \|\q\|_{p}
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
    = \frac{\|\q_j - \mathbf{y}_j\|_{p}}{\max_{l=1,\ldots,k}\|\q_l\|_{p}},
    \quad
    j = 1,\ldots,k.
$$

**`post.Lp_error(Qtrue, Qapprox, t=None, p=2)`**: Approximate the absolute and relative $L^{p}$-norm errors between snapshot sets `Qtrue` and `Qapprox` corresponding to times `t`.
The [$L^{p}$ norm](https://en.wikipedia.org/wiki/Lp_space#Lp_spaces) for vector-valued functions is defined by

$$
    \|\q(\cdot)\|_{L^p([a,b])}
    = \begin{cases}
    \left(\displaystyle\int_{a}^{b}\|\q(t)\|_{p}^p\:dt\right)^{1/p} & p < \infty,
    \\ & \\
    \sup_{t\in[a,b]}\|\q(t)\|_{\infty} & p = \infty.
    \end{cases}
$$

For finite _p_, the integrals are approximated by the trapezoidal rule:

$$
    \int_{a}^{b}\|\q(t)\|_{p}^{p}\:dt
    \approx \delta t\left(
        \frac{1}{2}\|\q(t_0)\|_{p}^p
        + \sum_{j=1}^{k-2}\|\q(t_j)\|_{p}^p
        + \frac{1}{2}\|\q(t_{k-1})\|_{p}^p
    \right),
    \\
    a = t_0 < t_1 < \cdots < t_{k-1} = b.
$$

The `t` argument can be omitted if _p_ is infinity (`p = np.inf`). -->
