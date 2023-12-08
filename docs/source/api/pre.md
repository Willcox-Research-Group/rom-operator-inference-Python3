# `opinf.pre`

```{eval-rst}
.. automodule:: opinf.pre
```

See [the preprocessing guide](../guides/preprocessing.ipynb) for discussion
and examples.

## Data Scaling

Raw dynamical systems data often need to be lightly preprocessed before use in Operator Inference.
The following tools enable centering/shifting and scaling/nondimensionalization of snapshot data.

```{eval-rst}
.. currentmodule:: opinf.pre

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    SnapshotTransformer
    SnapshotTransformerMulti
    scale
    shift
```
