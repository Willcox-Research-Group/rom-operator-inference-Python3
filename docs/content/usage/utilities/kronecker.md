# Compact Kronecker Representation

For a vector $\mathbf{q}\in\mathbb{R}^{n}$, the [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product) $\mathbf{q} \otimes \mathbf{q} \in \mathbb{R}^{n^2}$ contains redundant terms.
For example, $\mathbf{q}\otimes\mathbf{q}$ contains both $q_{1}q_{2}$ and $q_{2}q_{1}$.
To avoid these redundancies, we introduce a "compact" Kronecker product which only computes the unique terms of the usual Kronecker product:

$$
\begin{align*}
    \widehat{\mathbf{q}}\, \widehat{\otimes}\, \widehat{\mathbf{q}}
    := \left[\begin{array}{c}
        \widehat{\mathbf{q}}^{(1)} \\ \vdots \\ \widehat{\mathbf{q}}^{(r)}
    \end{array}\right]\in\mathbb{R}^{r(r+1)/2},
    \qquad\text{where}\qquad
    \widehat{\mathbf{q}}^{(i)}
    = x_i \left[\begin{array}{c}
        x_1 \\ \vdots \\ x_i
    \end{array}\right]\in\mathbb{R}^{i}.
\end{align*}
$$

The dimension $r(r+1)/2$ arises because we choose $2$ of $r$ entries without replacement, i.e., this is a multiset coefficient:

$$
\begin{align*}
    \left(\!\!{r\choose 2}\!\!\right)
    = \binom{r + 2 - 1}{2}
    = \binom{r+1}{2}
    = \frac{r(r+1)}{2}.
\end{align*}
$$

We similarly define a cubic compact product recursively with the quadratic compact product.

Under the hood, ROM classes infer...TODO
