# What's New

:::{attention}
`opinf` is a research code that is still in rapid development.
New versions may introduce substantial new features or API adjustments.
:::

:::{versionchanged} 0.5.0

- Overhauled the `operators` module so that each operator class is responsible for its portion of the Operator Inference data matrix.
  - New `StateInputOperator` for state-input bilinear interactions, $\Nhat[\u\otimes\qhat]$.
  - Operator classes now have `state_dimension` and `input_dimension` properties.
  - Operator classes must implement `datablock()` and `operator_dimension()` methods to facilitate operator inference.
- Renamed `roms` to `models` and updated class names:
  - `ContinuousOpInfROM` to `ContinuousModel`
  - `DiscreteOpInfROM` to `DiscreteModel`
  - `SteadyOpInfROM` to `SteadyModel`
- Model classes now take a list of operators in the constructor. String shortcuts such as `"cAH"` are still valid, similar to the previous `modelform` argument.
- Model classes no longer have a `basis` attribute.
  - The `basis` argument has been removed from the `fit()` method.
  - The `compress()` and `decompress()` methods have been removed from model classes.
  - The dimensions `n` and `r` have been replaced with `state_dimension`; `m` is now `input_dimension`.
- Moved time derivative estimation tools to the new `ddt` submodule.
- Moved Kronecker product utilities to static methods of nonparametric operators.
  - `utils.kron2c()` is now `QuadraticOperator.ckron()`
  - `utils.kron2c_indices()` is now `QuadraticOperator.ckron_indices()`
  - `utils.compress_quadratic()` is now `QuadraticOperator.compress_entries()`
  - `utils.expand_quadratic()` is now `QuadraticOperator.expand_entries()`
  - `utils.kron3c()` is now `CubicOperator.ckron()`
  - `utils.kron3c_indices()` is now `CubicOperator.ckron_indices()`
  - `utils.compress_cubic()` is now `CubicOperator.compress_entries()`
  - `utils.expand_cubic()` is now `CubicOperator.expand_entries()`
:::

:::{versionchanged} 0.4.5

- Moved basis classes and dimensionality reduction tools to a new `basis` submodule.
- Moved operator classes from `core.operators` to a new `operators` submodule.
- Renamed the `core` submodule to `roms`.
- Moved time derivative estimation tools to the `utils` module.
:::

:::{versionchanged} 0.4.4

- Fixed a bug in `SnapshotTransformer.load()` that treated the `centered_` attribute incorrectly.
- Removed the `transformer` attribute from basis classes.
- Renamed `encode()` to `compress()` and `decode()` to `decompress()`.
:::

:::{versionchanged} 0.4.2

- In the `fit()` method in ROM classes, replaced the `regularizer` argument with a `solver` keyword argument. The user should pass in an instantiation of a least-squares solver class from the `lstsq` submodule.
- Hyperparameters for least-squares solver classes in the `lstsq` submodule are now passed to the constructor; `predict()` must not take any arguments.
- Renamed the following least-squares solver classes in the `lstsq` submodule:
  - `SolverL2` -> `L2Solver`
  - `SolverL2Decoupled` -> `L2SolverDecoupled`
  - `SolverTikhonov` -> `TikhonovSolver`
  - `SolverTikhonovDecoupled` -> `TikhonovSolverDecoupled`

Before:

```python
>>> rom.fit(basis, states, ddts, inputs, regularizer=1e-2)
```

After:

```python
>>> solver = opinf.lstsq.L2Solver(regularizer=1e-2)
>>> rom.fit(basis, states, ddts, inputs, solver=solver)

# The L2 solver is also the default if a float is given:
>>> rom.fit(basis, states, ddts, inputs, solver=1e-2)
```

:::

## Versions 0.4.0 and 0.4.1

This version is a migration of the old `rom_operator_inference` package, version 1.4.1.
See [this page](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/wiki/API-Reference) for the documentation.
