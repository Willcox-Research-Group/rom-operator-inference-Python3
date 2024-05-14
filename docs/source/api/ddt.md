# `opinf.ddt`

```{eval-rst}
.. automodule:: opinf.ddt

.. currentmodule:: opinf.ddt

**Classes**

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    UniformFiniteDifferencer
    NonuniformFiniteDifferencer

**Finite Difference Schemes** (uniformly spaced time domain)

*Forward Differences*

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    fwd1
    fwd2
    fwd3
    fwd4
    fwd5
    fwd6

*Backward Differences*

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    bwd1
    bwd2
    bwd3
    bwd4
    bwd5
    bwd6

*Central Differences*

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    ctr2
    ctr4
    ctr6

*Mixed Differences*

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    ord2
    ord4
    ord6
    ddt_uniform

**Non-uniform Finite Difference Schemes**

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    ddt_nonuniform
    ddt

```

## Time Derivative Estimation

For time-continuous models, Operator Inference requires the time derivative of the state snapshots.
If they are not available from a full-order solver, the time derivatives can often be estimated from the snapshots.
The following functions implement finite difference estimators for the time derivative of snapshots.
