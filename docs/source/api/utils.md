# `opinf.utils`

```{eval-rst}
.. automodule:: opinf.utils
```

This module contains miscellaneous support functions for the rest of the
package.

## Kronecker Products

The matrix-vector product $\Hhat[\qhat\otimes\qhat]$,
where $\qhat\in\RR^{r}$ and $\Hhat\in\RR^{r\times r^{2}}$,
can be represented more compactly as  $\check{\H}[\qhat\ \widehat{\otimes}\ \qhat]$
where $\check{\H}\in\RR^{r\times r(r+1)/2}$ and
$\widehat{\otimes}$ is a compressed version of the Kronecker product.
Specifically, if
$\qhat = [\hat{q}_{1},\ldots,\hat{q}_{r}]\trp$,
then the full Kronecker product of $\qhat$ with itself is

$$
\begin{align*}
    \qhat\otimes\qhat
    = \left[\begin{array}{c}
        \hat{q}_{1}\qhat
        \\ \vdots \\
        \hat{q}_{r}\qhat
    \end{array}\right]
    =
    \left[\begin{array}{c}
        \hat{q}_{1}^{2} \\
        \hat{q}_{1}\hat{q}_{2} \\
        \vdots \\
        \hat{q}_{1}\hat{q}_{r} \\
        \hat{q}_{1}\hat{q}_{2} \\
        \hat{q}_{2}^{2} \\
        \vdots \\
        \hat{q}_{2}\hat{q}_{r} \\
        \vdots
        \hat{q}_{r}^{2}
    \end{array}\right] \in\RR^{r^{2}}.
\end{align*}
$$

The term $\hat{q}_{1}\hat{q}_{2}$ appears twice in the full Kronecker
product $\qhat\otimes\qhat$.
The compressed Kronecker product is defined here as

$$
\begin{align*}
    \qhat\ \widehat{\otimes}\ \qhat
    = \left[\begin{array}{c}
        \hat{q}_{1}^2
        \\
        \hat{q}_{2}\qhat_{1:2}
        \\ \vdots \\
        \hat{q}_{r}\qhat_{1:r}
    \end{array}\right]
    = \left[\begin{array}{c}
        \hat{q}_{1}^2 \\
        \hat{q}_{1}\hat{q}_{2} \\ \hat{q}_{2}^{2} \\
        \\ \vdots \\ \hline
        \hat{q}_{1}\hat{q}_{r} \\ \hat{q}_{2}\hat{q}_{r}
        \\ \vdots \\ \hat{q}_{r}^{2}
    \end{array}\right]
    \in \RR^{r(r+1)/2},
\end{align*}
$$

where

$$
\begin{align*}
    \qhat_{1:i}
    &= \left[\begin{array}{c}
        \hat{q}_{1} \\ \vdots \\ \hat{q}_{i}
    \end{array}\right]\in\RR^{i}.
\end{align*}
$$

The following functions facilitate compressed Kronecker products of this type.

```{eval-rst}
.. currentmodule:: opinf.utils

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    compress_cubic
    compress_quadratic
    expand_cubic
    expand_quadratic
    kron2c
    kron2c_indices
    kron3c
    kron3c_indices
```

## Load/Save HDF5 Utilities

Many `opinf` classes have `save()` methods that export the object to an HDF5 file and a `load()` class method for importing an object from an HDF5
file.
The following functions facilitate that data transfer.

```{eval-rst}
.. currentmodule:: opinf.utils

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    hdf5_loadhandle
    hdf5_savehandle
```
