(sec-whatsnew)=
# What's New

This page lists major changes made between package versions.

## New in Version 0.4.5

- Moved basis classes and dimensionality reduction tools to a new `basis` submodule.
- Moved operator classes from `core.operators` to a new `operators` submodule.
- Renamed the `core` submodule to `roms`.
- Moved time derivative estimation tools to the `utils` module.

## New in Version 0.4.4

- Fixed a bug in `SnapshotTransformer.load()` that treated the `centered_` attribute incorrectly.
- Removed the `transformer` attribute from basis classes.
- Renamed `encode()` to `compress()` and `decode()` to `decompress()`.

## New in Version 0.4.2

- In the `fit()` method in ROM classes, replaced the `regularizer` argument with a `solver` keyword argument. The user should pass in an instantiation of a least-squares solver class from the `lstsq` submodule.
- Hyperparameters for least-squares solver classes in the `lstsq` submodule are now passed to the constructor; `predict()` must not take any arguments.
- Renamed the following least-squares solver classes in the `lstsq` submodule:
    - `SolverL2` -> `L2Solver`
    - `SolverL2Decoupled` -> `L2SolverDecoupled`
    - `SolverTikhonov` -> `TikhonovSolver`
    - `SolverTikhonovDecoupled` -> `TikhonovSolverDecoupled`

Before:
```
>>> rom.fit(basis, states, ddts, inputs, regularizer=1e-2)
```

After:
```
>>> solver = opinf.lstsq.L2Solver(regularizer=1e-2)
>>> rom.fit(basis, states, ddts, inputs, solver=solver)

# The L2 solver is also the default if a float is given:
>>> rom.fit(basis, states, ddts, inputs, solver=1e-2)
```

## Versions 0.4.0 and 0.4.1

This version is a migration of the old `rom_operator_inference` package, version 1.4.1.
See [this page](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/wiki/API-Reference) for the documentation.
