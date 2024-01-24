# `opinf.ddt`

```{eval-rst}
.. automodule:: opinf.ddt
```

For time-continuous models, Operator Inference requires the time derivative of the state snapshots.
If they are not available from a full-order solver, the time derivatives can often be estimated from the snapshots.
The following functions implement finite difference estimators for the time derivative of snapshots.

```{eval-rst}
.. currentmodule:: opinf.ddt

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    ddt
    ddt_nonuniform
    ddt_uniform
```
