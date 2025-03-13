# `opinf.roms`

```{eval-rst}
.. automodule:: opinf.roms
```

## Deterministic Reduced-order Models

These classes are also available in the main [`opinf`](./main.md) namespace.

```{eval-rst}
.. currentmodule:: opinf.roms

.. autosummary::
   :toctree: _autosummaries
   :nosignatures:

   ROM
   ParametricROM
```

## Probabilistic Reduced-order Models

The following classes are used to represent probabilistic models.
A probability distribution is constructed for an operator matrix; an individual draw from this distribution defines a new deterministic model.
See {cite}`guo2022bayesopinf`.

```{eval-rst}
.. currentmodule:: opinf.roms

.. autosummary::
   :toctree: _autosummaries
   :nosignatures:

   OperatorPosterior
   BayesianROM
```
