# `opinf.basis`

```{eval-rst}
.. automodule:: opinf.basis
```

```{eval-rst}
.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    pod_basis
    cumulative_energy
    residual_energy
    svdval_decay
    LinearBasis
    PODBasis
    BasisMulti
```

## Low-dimensional Approximations

The purpose of learning a reduced-order model is to achieve a computational speedup, which is a result of reducing the dimension of the state $\q(t)\in\RR^{n}$ from $n$ to $r \ll n$.
This is accomplished by introducing a low-dimensional approximation $\q(t) \approx \boldsymbol{\Gamma}(\qhat(t))$, where $\qhat(t)\in\RR^{r}$.
