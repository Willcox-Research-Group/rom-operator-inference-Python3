# What's New

:::{attention}
`opinf` is a research code that is still in rapid development.
New versions may introduce substantial new features or API adjustments.
:::

## Version 0.5.15

- Improvement to `fit_regselect_*()` so that the regularization does not have to be initialized before fitting the model.
- Time derivative estimators in `opinf.ddt` now have a `mask()` method that map states to the estimation grid.
- Applied the `mask()` in `fit_regselect_continuous()` so that the error computation is correctly aligned with the training snapshots.
- Added regression tests based on the tutorial notebooks.
- Added `pytest` and `pytest-cov` to the dependencies for developers.

## Version 0.5.14

- Catch any errors in `fit_regselect*()` that occur when the model uses `refit()`.
- Tikhonov-type least-squares solvers do not require the regularizer in the constructor but will raise an `AttributeError` in `solve()` (and other methods) if the regularizer is not set.
- `PODBasis.fit(Q)` raises a warning when using the `"method-of-snapshots"`/`"eigh"` strategy if $n < k$ for $\mathbf{Q}\in\mathbb{R}^{n \times k}.$ In this case, calculating the $n \times k$ SVD is likely more efficient than the $k \times k$ eigenvalue problem.
- Added Python 3.13 to list of tests.

## Version 0.5.13

Bayesian operator inference:

- New `roms.BayesianROM` class.
- New supporting class `roms.OperatorPosterior`.
- Updates to relevant least-squares solvers.

These changes implement the framework proposed in {cite}`guo2022bayesopinf`.

## Version 0.5.12

- New `operators.QuarticOperator`, plus unit tests.
- Reorganized some unit tests for models and operators to have OOP structure.
- Bugfix: `fit_regselect_continuous()` now returns `self`

## Version 0.5.11

- New scaling option for `pre.ShiftScaleTransformer` so that training snapshots have at maximum norm 1. Contributed by [@nicolearetz](https://github.com/nicolearetz).
- Small clarifications to `pre.ShiftScaleTransformer` and updates to the `pre` documentation.

## Version 0.5.10

New POD basis solver option `basis.PODBasis(solver="method-of-snapshots")` (or `solver="eigh"`), which solves a symmetric eigenvalue problem instead of computing a (weighted) SVD. This method is more efficient than the SVD for snapshot matrices $\mathbf{Q}\in\mathbb{R}^{n\times k}$ where $n \gg k$ and is significantly more efficient than the SVD when a non-diagonal weight matrix is provided.
Contributed by [@nicolearetz](https://github.com/nicolearetz).

## Version 0.5.9

Automatic regularization selection:

- New methods `fit_regselect_continuous()` and `fit_regselect_discrete()` in the `ROM` and `ParametricROM` classes.
- `utils.gridsearch()` implements grid search followed up by derivative-free log-scale optimization.
- Class method `TikhonovSolver.get_operator_regularizer()` to construct diagonal Tikhonov regularizers where each operator is regularized by a different scalar.

New transformers for custom shifting / scaling:

- `pre.ShiftTransformer`
- `pre.ScaleTransformer`
- `pre.NullTransformer`
- `pre.TransformerPipeline`

Small improvements:

- Executing individual test files now runs the tests contained within.
- `utils.TimedBlock` has a `rebuffer` attribute that, when set to `True`, prevents printing until the end of the block.
- Improved test coverage, fixed some documentation typos, etc.

## Version 0.5.8

Support for affine-parametric problems:

- Affine-parametric operator classes `AffineConstantOperator`, `AffineLinearOperator`, etc.
- Parametric model classes `ParametricContinuousModel`, `ParametricDiscreteModel`.
- `ParametricROM` class.
- Updates to operator / model documentation.

Renamed interpolatory operators / model classes from `Interpolated<Name>` to `Interp<Name>`.
Old names are deprecated but not yet removed.

Miscellaneous:

- Reorganized and expanded tutorials.
- Added and documented `opinf.utils.TimedBlock` context manager for quick timing of code blocks.
- Updated structure for some unit tests.
- Refactored interpolatory operators.
- Standardized string representations, added `[Parametric]ROM.__str__()`.
- Removed some public functions from `operators`, regrouped in `operators._utils`.
- Removed some public functions from `models`, regrouped in `models._utils`.

## Version 0.5.7

Updates to `opinf.lstsq`:

- New `TruncatedSVDSolver` class.
- `predict()` has been renamed `solve()` for `opinf.lstsq` solver classes to not clash with `predict()` from `opinf.roms` / `opinf.models` classes.
- `solve()` always returns a two-dimensional array, even if $r = 1$.

Various small improvements to tests and documentation.

## Version 0.5.6

Added public templates to `opinf.operators`:

- `OperatorTemplate` for general nonparametric operators.
- `OpInfOperator` for nonparametric operators that can be learned through Operator Inference (operator matrix times data vector structure).
- `ParametricOperatorTemplate` for general parametric operators.
- `ParametricOpInfOperator` for parametric operators that can be learned through Operator Inference.

Also added a new `opinf.ddt.InterpolationDerivativeEstimator` class and made various small changes for compatibility with [NumPy 2.0.0](https://numpy.org/doc/stable/release/2.0.0-notes.html).

## Version 0.5.5

Changes to the `opinf.lstsq` API and improvements to the documentation.

- `opinf.model` classes now receive solvers **in the constructor, not in fit()**. This change will be useful for future models that require specific solvers. Updated `ROM` class and tutorials accordingly.
- New `SolverTemplate` class and inheritance guide for creating new solvers.
- Renamed attributes to match OpInf terminology.
  - `A --> data_matrix`, called $\D$ in the docs.
  - `B --> lhs_matrix`, called $\Z$ in the docs. **Warning:** `fit()` receives $\Z$, **not** $\Z\trp$!
  - $X$ is replaced with $\Ohat$. **Warning**: `predict()` returns $\Ohat$, **not** $\Ohat\trp$!
- Renamed two Tikhonov solver classes:
  - `L2SolverDecoupled --> L2DecoupledSolver`
  - `TikhonovSolverDecoupled --> TikhonovDecoupledSolver`
- Tikhonov solvers no longer have a default regularization value of zero.

Before:

```python
>>> model = opinf.models.ContinuousModel("A")
>>> solver = opinf.lstsq.L2Solver(regularizer=1e-2)
>>> model.fit(states, ddts, inputs, solver=solver)
```

After:

```python
>>> solver = opinf.lstsq.L2Solver(regularizer=1e-2)
>>> model = opinf.models.ContinuousModel("A", solver=solver)
>>> model.fit(states, ddts, inputs)
```

## Version 0.5.4

New `opinf.roms` submodule containing an `opinf.roms.ROM` class, also available in the main namespace as `opinf.ROM`.
This class wraps a lifter, transformer, basis, time derivative estimator, model, and least-squares solver together for convenience.
Rewrote the first tutorial to use `opinf.ROM`.

## Version 0.5.3

Expanded the `ddt` submodule (but no API changes to existing functions).

- New backward, central, and forward difference schemes up to sixth order.
- New `DerivativeEstimatorTemplate` class for implementing custom derivative estimation strategies.
- New `UniformFiniteDifference` and `NonuniformFiniteDifference` convenience classes for working with `ROM` classes in the future.
- Better documentation of the module.

The documentation was also updated to JupyterBook 1.0.0, a significant improvement to the look and feel.

Tests are now managed with `tox`, the contributor instructions were updated.

## Version 0.5.2

Significant updates to the `pre` and `basis` submodules.

Updates to `pre`:

- New `TransformerTemplate` class for defining custom transformers.
- Renamed `SnapshotTransformer` to `ShiftScaleTransformer`.
- Removed `SnapshotTransformerMulti`.
- New `TransformerMulti` class for joining multiple transformations.
- Renamed some attributes of the transformer classes: `n` -> `full_state_dimension`, `ni` -> `variable_size`, etc.

Updates to `basis`:

- New `BasisTemplate` class for defining custom bases.
- Standardized arguments of `fit()` to always be the snapshots. Hyperparameters must come in the constructor.
- `LinearBasis` now accepts an optional weight matrix.
- `LinearBasis` raises a warning if the basis entries are not orthogonal (w.r.t. the weights).
- Updated `PODBasis` dimensionality properties. Use `set_dimension()` to update the reduced state dimension on the fly.
- Removed `PODBasisMulti`.
- New `BasisMulti` multi class for joining multiple bases.
- Renamed some attributes of the basis classes (matching transformer syntax): `n` -> `full_state_dimension`, `r` -> `reduced_state_dimension`, etc.

Added a new Literature page to the documentation listing Operator Inference publications.

## Version 0.5.1

New `lift` module that defines a template class for implementing lifting transformations.

## Version 0.5.0

- Overhauled the `operators` module so that each operator class is responsible for its portion of the Operator Inference data matrix.
  - New `StateInputOperator` for state-input bilinear interactions, $\Nhat[\u\otimes\qhat]$.
  - Operator classes now have `state_dimension` and `input_dimension` properties.
  - Operator classes must implement `datablock()` and `operator_dimension()` methods to facilitate operator inference.
- Renamed `roms` to `models` and updated class names:
  - `ContinuousOpInfROM` to `ContinuousModel`
  - `DiscreteOpInfROM` to `DiscreteModel`
  - `SteadyOpInfROM` to `SteadyModel`
  - Same for interpolated models.
- Model classes now take a list of operators in the constructor. String shortcuts such as `"cAH"` are still valid, similar to the previous `modelform` argument. The `known_operators` argument has been removed from the `fit()` method.
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

## Version 0.4.5

- Moved basis classes and dimensionality reduction tools to a new `basis` submodule.
- Moved operator classes from `core.operators` to a new `operators` submodule.
- Renamed the `core` submodule to `roms`.
- Moved time derivative estimation tools to the `utils` module.

## Version 0.4.4

- Fixed a bug in `SnapshotTransformer.load()` that treated the `centered_` attribute incorrectly.
- Removed the `transformer` attribute from basis classes.
- Renamed `encode()` to `compress()` and `decode()` to `decompress()`.

## Version 0.4.2

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

## Versions 0.4.0 and 0.4.1

This version is a migration of the old `rom_operator_inference` package, version 1.4.1.
See [this page](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/wiki/API-Reference) for the documentation.
