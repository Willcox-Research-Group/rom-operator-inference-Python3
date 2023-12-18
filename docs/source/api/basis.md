# `opinf.basis`

```{eval-rst}
.. automodule:: opinf.basis
```

The purpose of learning a reduced-order model is to achieve a computational speedup, which is a result of reducing the dimension of the state $\q(t)\in\RR^{n}$ from $n$ to $r \ll n$.
This is accomplished by introducing a low-dimensional approximation $\q(t) \approx \boldsymbol{\Gamma}(\qhat(t))$, where $\qhat(t)\in\RR^{r}$.
The following tools construct this approximation.

```{eval-rst}
.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    LinearBasis
    LinearBasisMulti
    PODBasis
    PODBasisMulti
    cumulative_energy
    pod_basis
    projection_error
    residual_energy
    svdval_decay
```
