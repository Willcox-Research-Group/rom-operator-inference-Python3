# ROM Attributes

All ROM classes have the following attributes.

## Dimensions

These attributes are integers that are initially set to `None`, then inferred from the training inputs during `fit()`.
They cannot be altered manually after calling `fit()`.

| Attribute | Description |
| :-------- | :---------- |
| `n` | Dimension of the high-dimensional training data $\mathbf{q}$. |
| `r` | Dimension of the reduced-order model state $\widehat{\mathbf{q}}$. |
| `m` | Dimension of the input $\mathbf{u}$. |

If there is no input (meaning `modelform` does not contain `'B'`), then `m` is set to 0.

## Basis

The `basis` attribute is the mapping between the $n$-dimensional state space of the full-order data and the smaller $r$-dimensional state space of the reduced-order model (e.g., POD basis).
This is the first input to the `fit()` method.
See [Basis Computation](sec-basis-computation) for details.

## Operators

These attributes are the operators corresponding to the learned parts of the reduced-order model.
The classes are defined in `opinf.core.operators`.

<!-- TODO: Operator Class with links to API docs -->

| Attribute | Evaluation mapping | Jacobian mapping |
| :-------- | :----------------- | :--------------- |
| `c_` | $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{c}}$ | $\widehat{\mathbf{q}} \mapsto \mathbf{0}$ |
| `A_` | $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{A}}\widehat{\mathbf{q}}$ | $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{A}}$ |
| `H_` | $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{H}}[\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}]$ | $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{H}}[(\mathbf{I}\otimes\widehat{\mathbf{q}}) + (\widehat{\mathbf{q}}\otimes\mathbf{I})]$ |
| `G_` | $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{G}}[\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}]$ | $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{G}}[(\mathbf{I}\otimes\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}) + \cdots + (\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}\otimes\mathbf{I})]$ |
| `B_` | $\mathbf{u} \mapsto \widehat{\mathbf{B}}\mathbf{u}$ | $\mathbf{u} \mapsto \widehat{\mathbf{B}}$ |

All operators are set to `None` initially and only changed by `fit()` if the operator is included in the prescribed `modelform` (e.g., if `modelform="AHG"`, then `c_` and `B_` are always `None`).
<!-- Note that Jacobian mapping of the input operation _with respect to the state_ is zero. -->

### Operator Attributes

The discrete representation of the operator is a NumPy array stored as the `entries` attribute.
This array can also be accessed by slicing the operator object directly.

```python
>>> import numpy as np
>>> import opinf

>>> arr = np.arange(16).reshape(4, 4)
>>> operator = opinf.core.operators.LinearOperator(arr)

>>> operator.entries
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])

>>> operator[:]
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])

>>> operator.shape
(4, 4)
```

In practice, with a ROM object `rom`, the entries of (e.g.) the linear state matrix $\widehat{\mathbf{A}}$ are accessed with `rom.A_[:]` or `rom.A_.entries`.

### Operator Methods

The `evaluate()` method computes the action of the operator on the (low-dimensional) state or input.

```python
>>> q_ = np.arange(4)
>>> operator.evaluate(q_)
array([14, 38, 62, 86])

# Equivalent calculation with the raw NumPy array.
>>> arr @ q_
array([14, 38, 62, 86])
```

::::{note}
Nothing special is happening under the hood for constant and linear operators, but the quadratic and cubic operators use a compressed representation to efficiently compute the operator action on the quadratic or cubic Kronecker products $\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}$ or $\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}$.

```python
>>> r = 5
>>> arr2 = np.random.random((r, r**2))
>>> quadratic_operator = opinf.core.operators.QuadraticOperator(arr2)
>>> q_ = np.random.random(r)

>>> np.allclose(quadratic_operator.evaluate(q_), arr2 @ (np.kron(q_, q_)))
True

>>> quadratic_operator.shape
(5, 15)
```

The shape of the quadratic operator `entries` has been reduced from $r \times r^{2}$ to $r \times \frac{r(r + 1)}{2}$ to exploit the structure of the Kronecker products.

:::{dropdown} Details
Let $\widehat{\mathbf{q}} = [~\hat{q}_{1}~\cdots~\hat{q}_{r}~]^{\mathsf{T}}\in\mathbb{R}^{r}$ and consider the Kronecker product

$$
\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}
= \left[\begin{array}{c}
    \hat{q}_{1}\widehat{\mathbf{q}} \\
    \hat{q}_{2}\widehat{\mathbf{q}} \\
    \vdots \\
    \hat{q}_{r}\widehat{\mathbf{q}} \\
\end{array}\right]
= \left[\begin{array}{c}
    \hat{q}_{1}^{2} \\
    \hat{q}_{1}\hat{q}_{2} \\
    \vdots \\
    \hat{q}_{1}\hat{q}_{r} \\
    \hat{q}_{1}\hat{q}_{2} \\
    \hat{q}_{2}^{2} \\
    \vdots \\
    \hat{q}_{2}\hat{q}_{r} \\
    \vdots \\
    \hat{q}_{r}^{2}
\end{array}\right]
\in \mathbb{R}^{r^{2}}.
$$

Note that $\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}$ has some redundant entries, for example $\hat{q}_{1}\hat{q}_{2}$ shows up twice. In fact, $\hat{q}_{i}\hat{q}_{j}$ occurs twice for every choice of $i \neq j$.
Thus, $\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}$ can be represented with only $r (r + 1)/2$ degrees of freedom as, for instance,

$$
\left[\begin{array}{c}
    \widehat{\mathbf{q}}^{(1)} \\
    \widehat{\mathbf{q}}^{(2)} \\
    \vdots \\
    \widehat{\mathbf{q}}^{(r)}
\end{array}\right]
= \left[\begin{array}{c}
    \hat{q}_{1}^{2} \\
    \hat{q}_{1}\hat{q}_{2} \\
    \hat{q}_{2}^{2} \\
    \hat{q}_{1}\hat{q}_{3} \\
    \hat{q}_{2}\hat{q}_{3} \\
    \hat{q}_{3}^{2} \\
    \vdots \\
    \hat{q}_{r}^{2}
\end{array}\right]
\in \mathbb{R}^{r(r + 1)/2},
\qquad
\widehat{\mathbf{q}}^{(i)}
= \hat{q}_{i}\left[\begin{array}{c}
    \hat{q}_{1} \\ \vdots \\ \hat{q}_{i}
\end{array}\right]\in\mathbb{R}^{i}.
$$

This is the same as filling a vector with the upper-triangular entries of the outer product $\widehat{\mathbf{q}}\widehat{\mathbf{q}}^{\mathsf{T}}$.
The dimension $r (r + 1)/2$ arises because we choose 2 of r entries _without replacement_, i.e., this is a [multiset coefficient](https://en.wikipedia.org/wiki/Multiset#Counting_multisets):

$$
\left(\!\!{r\choose 2}\!\!\right)
= \binom{r + 2 - 1}{2}
= \binom{r+1}{2}
= \frac{r(r+1)}{2}.
$$

:::
::::

<!-- TODO: Jacobians -->

## Summary

| Attribute | Description |
| :-------- | :---------- |
| `n` | Dimension of the high-dimensional training data $\mathbf{q}$. |
| `r` | Dimension of the reduced-order model state $\widehat{\mathbf{q}}$. |
| `m` | Dimension of the input $\mathbf{u}$. |
| `basis` | Mapping between the $n$-dimensional state space of the full-order data and the $r$-dimensional state space of the ROM |
| `c_` | Constant operator $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{c}}$ |
| `A_` | Linear operator $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{A}}\widehat{\mathbf{q}}$ |
| `H_` | Quadratic operator $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{H}}[\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}]$ |
| `G_` | Cubic operator $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{G}}[\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}]$ |
| `B_` | Input operator $\mathbf{u} \mapsto \widehat{\mathbf{B}}\mathbf{u}$ |
