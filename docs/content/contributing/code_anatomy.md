(sec-contrib-anatomy)=
# Source Code Guide

This page briefly reviews the package anatomy and gives instructions for new additions.
The source code is stored in [`src/rom_operator_inference/`](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/tree/develop/src/rom_operator_inference) and is divided into the following submodules:
- `core`: operator and reduced-order model classes.
- `lstsq`: solvers for the regression problem at the heart of Operator Inference.
- `pre`: pre-processing tools (basis computation, state transformations, etc.).
- `post`: post-processing tools (mostly error evaluation).
- `utils`: other routines that don't obviously fit into another submodule. These functions are usually not important for casual users, but advanced users and developers may need access to them.

---

## Conventions

See the [Index of Notation](sec-notation) for specific naming conventions for both mathematical exposition and source code variables.

### Object-oriented Philosophy

Python is a highly [object-oriented language](https://docs.python.org/3/tutorial/classes.html), which is advantageous for building mathematical software with abstract objects.
One-off routines (such as computing a POD basis) should be implemented as standalone functions, but a class structure is preferable when possible.
The package has three main class hierarchies:

- [_ROM classes_](subsec-contrib-romclass) encapsulate an entire reduced-order model, such as $\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t) = \widehat{\mathbf{A}}\widehat{\mathbf{q}}(t) + \widehat{\mathbf{B}}\mathbf{u}(t)$ or $\widehat{\mathbf{q}}_{j+1} = \widehat{\mathbf{H}}[\widehat{\mathbf{q}}_{j} \otimes \widehat{\mathbf{q}}_{j}]$. These are the major objects that the user interacts with.
- [_Operator classes_](subsec-contrib-opclass) represent a single operator that forms part of a reduced-order model, e.g., $\widehat{\mathbf{A}}$ or $\widehat{\mathbf{H}}$. Every ROM object has, as attributes, several operator classes.
- [_Solver classes_](subsec-contrib-lstsqsolvers) handle the regression problem at the heart of Operator Inference.

As you design new classes, take advantage of intermediate classes and inheritance to avoid duplicating code.

### Scikit-Learn Style

We strive to follow [`scikit-learn` conventions](https://scikit-learn.org/stable/developers/develop.html#api-overview) and other paradigms that are common in machine learning Python libraries.
Specifically, both ROM classes and solver classes follow these rules.
- The `fit()` method receives training data and solves an inference problem.
- Attributes learned during `fit()` end with an underscore, in particular the reduced-order operators `c_`, `A_`, and so on. For consistency, low-dimensional quantities also end with an underscore (e.g., `state` is high dimensional but `state_` is low dimensional).
- The `predict()` method receives a scenario (e.g., initial conditions) and uses information learned during `fit()` to make a prediction for the specified scenario.

In addition, we follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and other standard Python formatting guidelines.
See [Linting](sec-contrib-linting) for more.

---

## Core Module

The `core` module contains all operator and reduced-order model class definitions.
Read [ROM Classes](sec-romclasses) before starting work here.

(subsec-contrib-opclass)=
### Operator Classes

The first step for implementing a new kind of reduced-order model is correctly implementing the individual operators that define the terms of the model, for example $\widehat{\mathbf{A}}$ and $\widehat{\mathbf{B}}$ of $\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t) = \widehat{\mathbf{A}}\widehat{\mathbf{q}}(t) + \widehat{\mathbf{B}}\mathbf{u}(t)$.
Operator classes are defined in `core/operators`.

#### Non-parametric Operators

A _non-parametric_ operator has constant entries that do not depend on external parameters.
Classes for representing non-parametric operators are defined in `core/operators/_nonparametric.py`.
These classes

- Inherit from `core.operators._base._BaseNonparametricOperator`.

- Are initialized with a NumPy array (the entries of the operator).
    Specifically, `__init__(self, entries)`
    1. calls `self._validate_entries(entries)` to ensure `entries` is a valid NumPy array,
    2. does any shape checking needed on `entries`, and
    3. calls `_BaseNonparametricOperator.__init__(self, entries)`.

- Implements `evaluate(self, state)` as the mapping defined by the operator.
    For the linear term $\widehat{\mathbf{A}}$ this is simply matrix multiplication $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{A}}\widehat{\mathbf{q}}$;
    for the quadratic term $\widehat{\mathbf{H}}$ this is the mapping $\widehat{\mathbf{q}}\mapsto\widehat{\mathbf{H}}[\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}]$, which is done efficiently by computing only the unique terms of the Kronecker product $\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}$.

- Implements `jacobian(self, state)` to evaluate of the Jacobian of the operator, i.e., $\frac{\partial}{\partial \widehat{\mathbf{q}}}$ of the mapping in `evaluate()`.
    For the linear mapping $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{A}}\widehat{\mathbf{q}}$, the Jacobian is $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{A}}$;
    for the quadratic mapping $\widehat{\mathbf{q}}\mapsto\widehat{\mathbf{H}}[\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}]$, the Jacobian is $\widehat{\mathbf{q}}\mapsto\widehat{\mathbf{H}}[(\mathbf{I}\otimes\widehat{\mathbf{q}}) + (\widehat{\mathbf{q}}\otimes\mathbf{I})]$.

::::{margin}
:::{note}
For nonparametric operators, `__call__()` is essentially an alias for `evaluate()`, so the action of an operator `H` on a state `q` may be computed as `H.evaluate(q)` or `H(q)`.
However, parametric operators _do not have_ an `evaluate()` method, and `__call__()` is implemented to construct a nonparametric operator corresponding to a given parameter value.
:::
::::

The following diagram shows the class hierarchy for non-parametric operators.

:::{mermaid}
%%{init: {'theme': 'neutral'}}%%
classDiagram
    class _BaseNonparametricOperator{
        <<abstract>>
        entries, shape
        __getitem__(), __eq__()
    }
    class ConstantOperator{
        __init__(), evaluate(), jacobian()
    }
    class LinearOperator{
        __init__(), evaluate(), jacobian()
    }
    class QuadraticOperator{
        __init__(), evaluate(), jacobian()
    }
    class CubicOperator{
        __init__(), evaluate(), jacobian()
    }
    _BaseNonparametricOperator --> ConstantOperator
    _BaseNonparametricOperator --> LinearOperator
    _BaseNonparametricOperator --> QuadraticOperator
    _BaseNonparametricOperator --> CubicOperator
:::


#### Parametric Operators

The entries of a _parametric_ operator depend on external parameters, e.g., $\widehat{\mathbf{A}} = \widehat{\mathbf{A}}(\mu)$.
Classes for representing parametric operators are grouped by the parametrization strategy into single files in `core/operators/`, e.g., `core/operators/_affine.py`.
These classes should

- Inherit from `core.operators._base._BaseParametricOperator`.

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

Here's an example of the inheritance hierarchy for affine-parametric operators, defined in `core/operators/_affine.py`.

:::{mermaid}
%%{init: {'theme': 'neutral'}}%%
classDiagram
    class _BaseParametricOperator{
        <<abstract>>
    }
    class _BaseAffineOperator{
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
    _BaseParametricOperator --> _BaseAffineOperator
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
For example, here is a barebones version of `_AffineOperator`:
```python
class _AffineOperator(_BaseParametricOperator):
    """Base class for parametric operators with affine structure."""

    def __init__(self, coefficient_functions, matrices):
        """Save the coefficient functions and operator matrices."""
        self.ceofficient_functions = coefficient_functions
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

(subsec-contrib-romclass)=
### ROM Classes

The `_BaseROM` class of `core/_base.py` is the base class for all reduced-order models.
It handles
- Dimensions (`n`, `r`, and `m`).
- The basis $\mathbf{V}_{r}$ (`basis`).
- Reduced-order operators (`c_`, `A_`, `H_`, `G_`, and `B_`).
- Projection $\mathbf{q}\mapsto\widehat{\mathbf{q}} := \mathbf{V}_{r}^{\top}\mathbf{q}$ (`project()`) and reconstruction $\widehat{\mathbf{q}} \mapsto \mathbf{V}_{r}\widehat{\mathbf{q}}$ (`reconstruct()`).
- Evaluation of the right-hand side of the ROM (`evaluate()`) and its Jacobian (`jacobian()`).

Classes that inherit from `_BaseROM` must (eventually) implement the following methods.
- `fit()`: Train the reduced-order model with the specified data.
- `predict()`: Solve the reduced-order model under specified conditions.
- `save()`: Save the reduced-order structure / operators locally in HDF5 format.
- `load()`: Load a previously saved reduced-order model from an HDF5 file. This should be a [`@classmethod`](https://stackoverflow.com/questions/136097/difference-between-staticmethod-and-classmethod).

To write a new ROM class, start with an intermediate base class that inherits from `_BaseROM` and implements `fit()`, `save()`, and `load()`.
See `core.nonparametric._base._NonparametricOpInfROM`, for example.
Then write classes that inherit from your intermediate base class for handling steady-state, discrete-time, and continuous-time problems by implementing `predict()`.

#### Non-parametric ROMs

A _non-parametric_ ROM has exclusively non-parametric operators, i.e., their entries do not depend on external parameters.
Classes for representing non-parametric ROMs are defined in the `core/nonparametric/` folder, which has the following files:
- `core/nonparametric/_base.py` defines a base class, `_NonparametricOpInfROM`, which implements Operator Inference in the non-parametric setting.
- `core/nonparametric/_public.py` defines public-facing classes that inherit from `_NonparametricOpInfROM`, one for [the discrete setting](sec-discrete) and one for the [continuous setting](sec-continuous).
- `core/nonparametric/_frozen.py` defines classes that inherit from the public-facing classes but which have their `fit()` method disabled. These classes provide a helpful protection for the parametric case, described later.

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
        project(), reconstruct()
        evaluate(), jacobian()
    }
    class _NonparametricOpInfROM{
        <<abstract>>
        operator_matrix_
        data_matrix_
        fit(), save(), load()
    }
    class SteadyOpInfROM{
        evaluate(), fit(), predict()
    }
    class DiscreteOpInfROM{
        evaluate(), fit(), predict()
    }
    class ContinuousOpInfROM{
        evaluate(), fit(), predict()
    }
    class _FrozenMixin{
        fit()
    }
    class _FrozenSteadyROM
    class _FrozenDiscreteROM
    class _FrozenContinuousROM
    _BaseROM --|> _NonparametricOpInfROM
    _NonparametricOpInfROM --|> SteadyOpInfROM
    _NonparametricOpInfROM --|> DiscreteOpInfROM
    _NonparametricOpInfROM --|> ContinuousOpInfROM
    _FrozenMixin --|> _FrozenSteadyROM
    _FrozenMixin --|> _FrozenDiscreteROM
    _FrozenMixin --|> _FrozenContinuousROM
    SteadyOpInfROM --|> _FrozenSteadyROM
    DiscreteOpInfROM --|> _FrozenDiscreteROM
    ContinuousOpInfROM --|> _FrozenContinuousROM
:::

#### Parametric ROMs

::::{margin}
:::{note}
For every parameterization strategy, there should be 1) a file in `core/operators/` implementing parametric operator classes, and 2) a folder in `core/` grouping the parametric ROM classes.
For example, the ROM classes defined in `core/affine/` use the operator classes defined in `core/operators/_affine.py`.
:::
::::

A _parametric ROM_ has one or more parametric operators.
Classes for representing parametric ROMs should be grouped by parameterization strategy in a new folder within `core/` (e.g., `core/affine/`).
These classes should

- Inherit from `core._base._BaseParametricROM`, which adds an attribute `p` for the parameter space dimension and implements the following methods:
    - `__call__(self, parameter)` results in a parametric ROM whose operators correspond to the given parameter.
    - `evaluate(self, parameter, *args, **kwargs)` evaluates the parametric ROM at the given parameter, then calls the resulting object's `evaluate()` mthod with the remaining arguments.
    - `predict(self, parameter, *args, **kwargs)` evaluates the parametric ROM at the given parameter, then calls the resulting object's `predict()` method with the remaining arguments.

- Call `_BaseParametricROM.__init__(self)` in the constructor.

- Define `_ModelClass` as a class (static) attribute to be the type of non-parametric ROM that the parametric ROM evaluates to.

        >>> parametric_rom = MyNewParametricROM(init_args).fit(fit_args)
        >>> nonparametric_rom = parametric_rom(parameter)
        >>> isinstance(nonparametric_rom, _NonparametricOpInfROM)
        True

    The `_ModelClass` should be one of the "frozen" non-parametric ROM classes in `core/nonparametric/_frozen.py`, which have their `fit()` method disabled.
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
        project(), reconstruct()
        evaluate(), jacobian()
    }
    class _BaseParametricROM{
        p
        predict(), evaluate(), __call__()
    }
    class _AffineOpInfROM{
        <<abstract>>
        fit(), save(), load()
    }
    class AffineSteadyOpInfROM{
        fit(), predict(), evaluate()
    }
    class AffineDiscreteOpInfROM{
        fit(), predict(), evaluate()
    }
    class AffineContinuousOpInfROM{
        fit(), predict(), evaluate()
    }
    class _FrozenSteadyROM
    class _FrozenDiscreteROM
    class _FrozenContinuousROM
    _BaseROM --|> _BaseParametricROM
    _BaseParametricROM --|> _AffineOpInfROM
    _AffineOpInfROM --|> AffineSteadyOpInfROM
    _AffineOpInfROM --|> AffineDiscreteOpInfROM
    _AffineOpInfROM --|> AffineContinuousOpInfROM
    AffineSteadyOpInfROM ..|> _FrozenSteadyROM : rom(parameter)
    AffineDiscreteOpInfROM ..|> _FrozenDiscreteROM : rom(parameter)
    AffineContinuousOpInfROM ..|> _FrozenContinuousROM : rom(parameter)
:::

:::{tip}
Similar to parametric operators, a group of new parametric ROM classes can usually be implemented by writing an intermediate class that does most of the work, then writing child classes for the discrete and continuous settings which each set `_ModelClass` appropriately.
For instance, the intermediate class can often implement `fit()`, `save()`, and `load()`, and `_BaseParametricROM` implements `evaluate()`, `predict()`, and `jacobian()` already.
However, it is important to have tailored signatures and detailed docstrings for `fit()` and `predict()`, so these should be carefully defined for every public-facing class.
:::


---

(subsec-contrib-lstsqsolvers)=
## Least-squares Module

TODO
