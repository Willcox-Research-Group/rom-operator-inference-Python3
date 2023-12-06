# Source Code Guide

This page briefly reviews the package anatomy and gives instructions for new additions.
The source code is stored in [`src/opinf/`](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/tree/main/src/opinf) and is divided into the following submodules:

- {mod}`opinf.pre`: pre-processing tools for transforming snapshots before dimension reduction.
- {mod}`opinf.basis`: dimensionality reduction (compression) tools.
- {mod}`opinf.operators`: operators classes (individual terms for reduced-order models).
- {mod}`opinf.lstsq`: solvers for the least-squares regression problem at the heart of Operator Inference.
- {mod}`opinf.roms`: reduced-order model classes.
- {mod}`opinf.post`: post-processing tools (mostly error evaluation).
- {mod}`opinf.utils`: other routines that don't obviously fit into another submodule. These functions are usually not important for casual users, but advanced users and developers may need access to them.

---

## Conventions

See the [Index of Notation](./notation.md) for specific naming conventions for both mathematical exposition and source code variables.

### Object-oriented Philosophy

Python is a highly [object-oriented language](https://docs.python.org/3/tutorial/classes.html), which is advantageous for building mathematical software with abstract objects.
One-off routines may be implemented as standalone functions, but a class structure is preferable when possible.
The package defines the following main class hierarchies:

- [_Transformer classes_](subsec-contrib-transformerclass) are used for normalizing snapshot data (see {mod}`opinf.pre`).
- [_Basis classes_](subsec-contrib-basisclass) represent the mapping between the full-order state space in $\RR^{n}$ and the reduced-order state space in $\RR^{r}$ (see {mod}`opinf.basis`)
- [_ROM classes_](subsec-contrib-romclass) encapsulate an entire reduced-order model, such as $\frac{\text{d}}{\text{d}t}\qhat(t) = \Ahat\qhat(t) + \Bhat\u(t)$ or $\qhat_{j+1} = \Hhat[\qhat_{j} \otimes \qhat_{j}]$. These are the major objects that the user interacts with (see {mod}`opinf.roms`).
- [_Operator classes_](subsec-contrib-opclass) represent a single operator that forms part of a reduced-order model, e.g., $\Ahat\q$ or $\Hhat[\qhat\otimes\qhat]$. Every ROM object has an `operators` attribute that is a list of operator objects (see {mod}`opinf.operators`).
- [_Solver classes_](subsec-contrib-lstsqsolvers) handle the least-squares regression problem at the heart of Operator Inference (see {mod}`opinf.lstsq`).

:::{tip}
As you design new classes, take advantage of intermediate classes and inheritance to avoid duplicating code when appropriate.
For example, the operator classes {class}`opinf.operators.LinearOperator` and {class}`opinf.operators.QuadraticOperator` both inherit from ``operators._base._BaseNonparametricOperator``, which handles tasks that are common to all nonparametric operators.
:::

:::{admonition} Scikit-Learn Style
:class: note

We strive to follow [`scikit-learn` conventions](https://scikit-learn.org/stable/developers/develop.html#api-overview) and other paradigms that are common in machine learning Python libraries.
Specifically, both ROM classes and solver classes follow these rules.

- The `fit()` method receives training data and solves an inference problem.
- Attributes learned during `fit()` end with an underscore, e.g., `solver_`. For consistency, low-dimensional quantities also end with an underscore (e.g., `state` is high dimensional ($\q$) but `state_` is low dimensional ($\qhat$)).
- The `predict()` method receives a scenario (e.g., initial conditions) and uses information learned during `fit()` to make a prediction for that scenario.
:::

### Code Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and other standard Python formatting guidelines.
See [Linting](sec-contrib-linting) and [Formatting](sec-contrib-formatting) for more.

---

## Pre Module

The [**pre**](opinf.pre) submodule contains transformer class definitions and related tools.
Transformers can be _monolithic_ (treating the state as one variable) or _multivariable_ (treating different chunks of the state as separate variables).
Multivariable transformers and bases inherit from `_multivar._MultivarMixin` and should have, as attributes, a list of monolithic counterparts.

Read [Data Preprocessing](sec-guide-preprocessing) before starting work here.

(subsec-contrib-transformerclass)=
### Transformer Classes

All transformer classes must

- Inherit from `pre._base._BaseTransformer`.
- Accept and store any hyperparameters (transformation settings) in the constructor.
- Implement `transform()`, `fit_transform()`, and `inverse_transform()`. Note that `fit()` is already implemented in `_BaseTransformer` and should work as long as `fit_transform()` is implemented.

There are two basic transformers, `SnapshotTransformer` (monolithic) and `SnapshotTransformerMulti` (multivariable).
These classes handle standard shifting / scaling transformations.
Lifting transformations should not be added to this package disguised as snapshot transformers (too problem dependent).

## Basis Module

The [**basis**](opinf.basis) submodule contains basis class definitions and related tools.
Bases can be _monolithic_ (treating the state as one variable) or _multivariable_ (treating different chunks of the state as separate variables).
Multivariable transformers and bases inherit from `_multivar._MultivarMixin` and should have, as attributes, a list of monolithic counterparts.

Read [Dimensionality Reduction](sec-guide-dimensionality) before starting work here.

(subsec-contrib-basisclass)=
### Basis Classes

All basis classes must

- Inherit from `basis._base._BaseBasis`.
- Accept and store any hyperparameters in the constructor.
- Implement `fit()`, `compress()`, and `decompress()`.

:::{versionchanged} 0.4.4
Note that `project()` is **not** the same as `compress()`, in fact `project(q) = decompress(compress(q))`.
This is a name change from previous versions.
:::

Before implementing a new basis class, take a close look at {class}`opinf.basis.LinearBasis` and then {class}`opinf.basis.PODBasis`.

(subsec-contrib-opclass)=
## Operators Module

The first step for implementing a new kind of reduced-order model is correctly implementing the individual operators that define the terms of the model, for example $\Ahat$ and $\Bhat$ of $\frac{\text{d}}{\text{d}t}\qhat(t) = \Ahat\qhat(t) + \Bhat\u(t)$.
Operator classes are defined in the [**operators**](opinf.operators) module.

### Non-parametric Operators

A _non-parametric_ operator has constant entries that do not depend on external parameters.
Classes for representing non-parametric operators are defined in `operators/_nonparametric.py`.
These classes inherit from `operators._base._BaseNonparametricOperator` and must implement its abstract methods.

#### Base Class for Nonparametric Operators

:::{eval-rst}
.. currentmodule:: opinf.operators_new

.. autosummary::
    :toctree: ../api/_autosummaries
    :nosignatures:

    _base._NonparametricOperator
:::

#### Abstract Methods to Implement

#### Operator Class Template

```python
class MyNewOperator(_BaseNonparametricOperator):
    TODO
```

::::{margin}
:::{note}
For nonparametric operators, `apply()` is essentially an alias for `__call__()`, so the action of an operator `H` on a state `q` may be computed as `H.apply(q)` or `H(q)`.
However, parametric operators _do not have_ an `apply()` method, and `__call__()` is implemented to construct a nonparametric operator corresponding to a given parameter value.
:::
::::

#### Parametric Operators

The entries of a _parametric_ operator depend on external parameters, e.g., $\Ahat = \Ahat(\mu)$.
Classes for representing parametric operators are grouped by the parametrization strategy into single files in `operators/`, e.g., `operators/_affine.py`.
These classes should

- Inherit from `operators._base._BaseParametricOperator`.

- Be initialized with whatever parameter data, matrices, and/or functions are needed to represent the parametric operator and validate these inputs (e.g., shape checking on any matrices).

- Call `_BaseParametricROM.__init__(self)` at the beginning of `__init__()`.

- Set the parameter dimension by calling `self._set_parameter_dimension(parameter_values)` in `__init__()`.

- Provide public access to the members that define the parametric structure (`parameter_values`, `coefficient_functions`, `matrices`, etc.).

- Implement `__call__(self, parameter)` so that it returns an non-parametric operator object:

    ```python
        >>> parametric_operator = MyNewParametricOperator(init_args)
        >>> nonparametric_operator = parametric_operator(parameter)
        >>> isinstance(nonparametric_operator, _BaseNonparametricOperator)
        True
    ```

- Define `_OperatorClass` as a class (static) attribute to be the type of non-parametric operator that the parametric operator evaluates to. This is done by setting the variable within the class but outside of any methods:

        ```python
        class MyNewLinearParametricROM(_BaseParametricROM):
            """Don't forget to write descriptive docstrings!"""
            _OperatorClass = LinearOperator     # Set the variable outside of any method.

            def __init__(self, ...):
                """Don't forget to write descriptive docstrings!"""
                _BaseParametricROM.__init__(self)
                # ...
        ```

    Use `self.OperatorClass` to access this within the class (this is a property inherited from `_BaseParametricOperator`).

Here's an example of the inheritance hierarchy for affine-parametric operators, defined in `operators/_affine.py`.

:::{mermaid}
%%{init: {'theme': 'neutral'}}%%
classDiagram
    class _BaseParametricOperator{
        <<abstract>>
    }
    class_BaseAffineOperator{
        <<abstract>>
        matrices, coefficients, shape
        __getitem__(), __eq__()
    }
    class ConstantAffineOperator{
        __init__(), __call__()
    }
    class LinearAffineOperator{
        __init__(), __call__()
    }
    class QuadraticAffineOperator{
        __init__(), __call__()
    }
    class CubicAffineOperator{
        __init__(), __call__()
    }
    class ConstantOperator
    class LinearOperator
    class QuadraticOperator
    class CubicOperator
    _BaseParametricOperator -->_BaseAffineOperator
    _BaseAffineOperator --> ConstantAffineOperator
    _BaseAffineOperator --> LinearAffineOperator
    _BaseAffineOperator --> QuadraticAffineOperator
    _BaseAffineOperator --> CubicAffineOperator
    ConstantAffineOperator ..|> ConstantOperator : operator(parameter)
    LinearAffineOperator ..|> LinearOperator : operator(parameter)
    QuadraticAffineOperator ..|> QuadraticOperator : operator(parameter)
    CubicAffineOperator ..|> CubicOperator : operator(parameter)
:::

:::{tip}
Usually a group of new parametric operator classes can be implemented by writing an intermediate class that does all of the work, then writing child classes for constant, linear, quadratic, and cubic terms which each set `_OperatorClass` appropriately.
For example, here is a simplified version of `_AffineOperator`:

```python
class _AffineOperator(_BaseParametricOperator):
    """Base class for parametric operators with affine structure."""

    def __init__(self, coefficient_functions, matrices):
        """Save the coefficient functions and operator matrices."""
        self.coefficient_functions = coefficient_functions
        self.matrices = matrices

    def __call__(self, parameter):
        """Evaluate the affine operator at the given parameter."""
        entries = sum([thetai(parameter) * Ai
                    for thetai, Ai in zip(self.coefficient_functions,
                                            self.matrices)])
        return self.OperatorClass(entries)
```

Note that `__call__()` uses `self.OperatorClass` to wrap the entries of the evaluated operator.
Now to define constant-affine and linear-affine classes, we simply inherit from `_AffineOperator` and set the `_OperatorClass` appropriately.

```python
class AffineConstantOperator(_AffineOperator):
    """Constant operator with affine parametric structure."""
    _OperatorClass = ConstantOperator


class AffineLinearOperator(_AffineOperator):
    """Linear operator with affine parametric structure."""
    _OperatorClass = LinearOperator
```

This strategy reduces boilerplate and testing code by isolating the heavy lifting to the intermediate class (`_AffineOperator`).
:::

## ROMs Module

The {mod}`opinf.roms` module contains all operator and reduced-order model class definitions.
Read [Reduced-order Model Design](../guides/romclasses.md) before starting work here.

(subsec-contrib-romclass)=
### ROM Classes

The `_BaseROM` class of `roms/_base.py` is the base class for all reduced-order models.
It handles

- Dimensions (`n`, `r`, and `m`).
- The basis $\Vr$ (`basis`).
- Reduced-order operators (`c_`, `A_`, `H_`, `G_`, and `B_`).
- Compression $\q\mapsto\qhat := \Vr^{\top}\q$ (`compress()`) and decompression $\qhat \mapsto \Vr\qhat$ (`decompress()`).
- Evaluation of the right-hand side of the ROM (`evaluate()`) and its Jacobian (`jacobian()`).

Classes that inherit from `_BaseROM` must (eventually) implement the following methods.

- `fit()`: Train the reduced-order model with the specified data.
- `predict()`: Solve the reduced-order model under specified conditions.
- `save()`: Save the reduced-order structure / operators locally in HDF5 format.
- `load()`: Load a previously saved reduced-order model from an HDF5 file. This should be a [`@classmethod`](https://stackoverflow.com/questions/136097/difference-between-staticmethod-and-classmethod).

To write a new ROM class, start with an intermediate base class that inherits from `_BaseROM` and implements `fit()`, `save()`, and `load()`.
See `roms.nonparametric._base._NonparametricROM`, for example.
Then write classes that inherit from your intermediate base class for handling steady-state, discrete-time, and continuous-time problems by implementing `predict()`.

#### Non-parametric ROMs

A _non-parametric_ ROM has exclusively non-parametric operators, i.e., their entries do not depend on external parameters.
Classes for representing non-parametric ROMs are defined in the `roms/nonparametric/` folder, which has the following files:

- `roms/nonparametric/_base.py` defines a base class, `_NonparametricROM`, which implements Operator Inference in the non-parametric setting.
- `roms/nonparametric/_public.py` defines public-facing classes that inherit from `_NonparametricROM`, one for [the discrete setting](sec-discrete) and one for the [continuous setting](sec-continuous).
- `roms/nonparametric/_frozen.py` defines classes that inherit from the public-facing classes but which have their `fit()` method disabled. These classes provide a helpful protection for the parametric case, described later.

The following diagram shows the inheritance hierarchy for the non-parametric ROM classes.

:::{mermaid}
%%{init: {'theme': 'neutral'}}%%
classDiagram
    class _BaseROM{
        <<abstract>>
        modelform
        n, m, r
        basis
        c_, A_, H_, G_, B_
        compress(), decompress()
        evaluate(), jacobian()
    }
    class _NonparametricROM{
        <<abstract>>
        operator_matrix_
        data_matrix_
        fit(), save(), load()
    }
    class SteadyROM{
        evaluate(), fit(), predict()
    }
    class DiscreteROM{
        evaluate(), fit(), predict()
    }
    class ContinuousROM{
        evaluate(), fit(), predict()
    }
    class _FrozenMixin{
        fit()
    }
    class_FrozenSteadyROM
    class _FrozenDiscreteROM
    class_FrozenContinuousROM
    _BaseROM --|>_NonparametricROM
    _NonparametricROM --|> SteadyROM
    _NonparametricROM --|> DiscreteROM
    _NonparametricROM --|> ContinuousROM
    _FrozenMixin --|> _FrozenSteadyROM
    _FrozenMixin --|> _FrozenDiscreteROM
    _FrozenMixin --|> _FrozenContinuousROM
    SteadyROM --|>_FrozenSteadyROM
    DiscreteROM --|> _FrozenDiscreteROM
    ContinuousROM --|>_FrozenContinuousROM
:::

#### Parametric ROMs

::::{margin}
:::{note}
For every parameterization strategy, there should be 1) a file in `operators/` implementing parametric operator classes, and 2) a folder in `roms/` grouping the parametric ROM classes.
For example, the ROM classes defined in `roms/affine/` use the operator classes defined in `operators/_affine.py`.
:::
::::

A _parametric ROM_ has one or more parametric operators.
Classes for representing parametric ROMs should be grouped by parameterization strategy in a new folder within `roms/` (e.g., `roms/affine/`).
These classes should

- Inherit from `roms._base._BaseParametricROM`, which adds an attribute `p` for the parameter space dimension and implements the following methods:
  - `__call__(self, parameter)` results in a parametric ROM whose operators correspond to the given parameter.
  - `evaluate(self, parameter, *args, **kwargs)` evaluates the parametric ROM at the given parameter, then calls the resulting object's `evaluate()` method with the remaining arguments.
  - `predict(self, parameter, *args, **kwargs)` evaluates the parametric ROM at the given parameter, then calls the resulting object's `predict()` method with the remaining arguments.

- Call `_BaseParametricROM.__init__(self)` in the constructor.

- Define `_ModelClass` as a class (static) attribute to be the type of non-parametric ROM that the parametric ROM evaluates to.

        >>> parametric_rom = MyNewParametricROM(init_args).fit(fit_args)
        >>> nonparametric_rom = parametric_rom(parameter)
        >>> isinstance(nonparametric_rom, _NonparametricROM)
        True

    The `_ModelClass` should be one of the "frozen" non-parametric ROM classes in `roms/nonparametric/_frozen.py`, which have their `fit()` method disabled.
    This prevents the user from calling `fit()` on the evaluated parametric ROM (`nonparametric_rom` in the code block above) when they meant to fit the parametric ROM (`parametric_rom`).

<!-- TODO: set dimensions. -->

Here is the inheritance hierarchy for the affine-parametric ROM classes.

:::{mermaid}
%%{init: {'theme': 'neutral'}}%%
classDiagram
    class _BaseROM{
        <<abstract>>
        modelform
        n, m, r
        basis
        c_, A_, H_, G_, B_
        compress(), decompress()
        evaluate(), jacobian()
    }
    class _BaseParametricROM{
        p
        predict(), evaluate(), **call**()
    }
    class_AffineROM{
        <<abstract>>
        fit(), save(), load()
    }
    class AffineSteadyROM{
        fit(), predict(), evaluate()
    }
    class AffineDiscreteROM{
        fit(), predict(), evaluate()
    }
    class AffineContinuousROM{
        fit(), predict(), evaluate()
    }
    class _FrozenSteadyROM
    class_FrozenDiscreteROM
    class _FrozenContinuousROM
    _BaseROM --|> _BaseParametricROM
    _BaseParametricROM --|> _AffineROM
    _AffineROM --|> AffineSteadyROM
    _AffineROM --|> AffineDiscreteROM
    _AffineROM --|> AffineContinuousROM
    AffineSteadyROM ..|> _FrozenSteadyROM : rom(parameter)
    AffineDiscreteROM ..|>_FrozenDiscreteROM : rom(parameter)
    AffineContinuousROM ..|> _FrozenContinuousROM : rom(parameter)
:::

:::{tip}
Similar to parametric operators, a group of new parametric ROM classes can usually be implemented by writing an intermediate class that does most of the work, then writing child classes for the discrete and continuous settings which each set `_ModelClass` appropriately.
For instance, the intermediate class can often implement `fit()`, `save()`, and `load()`, and `_BaseParametricROM` implements `evaluate()`, `predict()`, and `jacobian()` already.
However, it is important to have tailored signatures and detailed docstrings for `fit()` and `predict()`, so these should be carefully defined for every public-facing class.
:::

---

(subsec-contrib-lstsqsolvers)=
## Least-squares Module

The [**lstsq**](opinf.lstsq) module houses classes and tools for solving the least-squares problems that arise in operator inference, namely $\min_{\mathbf{X}}\|\A\mathbf{X} - \B\|$ plus regularization and/or constraints.
Each solver class defines 1) how the problem is posed (e.g., is there regularization and how is it structured, are there constraints and if so what are they, etc.) and 2) how the problem is to be solved (e.g., use the SVD, use the QR decomposition, solve the normal equations, etc.).

The `fit()` method of each ROM class has a `solver` keyword argument which accepts a `lstsq` solver object to apply to the associated operator inference least-squares problem.

### Solver Classes

Least-squares solver classes must do the following.

- Inherit from `lstsq._base._BaseSolver`.
- Accept any hyperparameters (e.g., regularization values) in the constructor.
- Have a `fit(A, B)` method that calls `_BaseSolver.fit(A, B)` and sets up the least-squares problem.
- Have a `predict()` method returns the solution `X` to the least-squares problem $||\mathbf{AX} - \B||$ (+ regularization, etc.).
- Have a `_LSTSQ_LABEL` class attribute that is a string describing the form of the least-squares problem, e.g., `"min_{X} ||AX - B||_F^2 + ||µX||_F^2"` This is used in the string representation of the class.

See `lstsq._tikhonov.L2Solver` for an example. Here is the inheritance hierarchy of the current least-squares solvers.

:::{mermaid}
%%{init: {'theme': 'neutral'}}%%
classDiagram
    class _BaseSolver{
        <<abstract>>
        k, d, r, A, B
        fit(), cond(), misfit()
    }
    class PlainSolver{
        predict()
    }
    class_BaseTikhonovSolver{
        <<abstract>>
        regularizer
    }
    class L2Solver{
        fit(), predict(), regcond(), residual()
    }
    class L2SolverDecoupled{
        fit(), regcond(), residual()
    }
    class TikhonovSolver{
        fit(), predict(), regcond(), residual()
    }
    class TikhonovSolverDecoupled{
        predict(), regcond(), residual()
    }
    _BaseSolver --|> PlainSolver
    _BaseSolver --|> _BaseTikhonovSolver
    _BaseTikhonovSolver --|> L2Solver
    L2Solver --|> L2SolverDecoupled
    _BaseTikhonovSolver --|> TikhonovSolver
    TikhonovSolver --|> TikhonovSolverDecoupled
:::
