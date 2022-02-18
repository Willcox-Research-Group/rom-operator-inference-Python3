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

The first step for implementing a new kind of reduced-order model is correctly implementing the individual operators that the model consists of, the  $\widehat{\mathbf{A}}$ of $\frac{\text{d}}{\text{d}t}\widehat{\mathbf{A}}\widehat{\mathbf{q}}(t)$ for example.
Operator classes are defined in `core/operators`.

A _non-parametric operator_ has constant entries that do not depend on external parameters.
These classes are defined in `core/operators/_nonparametric.py` and inherit from `core.operators._base._BaseNonparametricOperator`.
They are initialized with a NumPy array (the entries of the operator), which can be accessed as the `entries` attribute.
These classes also handle the mappings defined by the operator: for the linear term $\widehat{\mathbf{A}}$ this is simply matrix multiplication $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{A}}\widehat{\mathbf{q}}$; for the quadratic term $\widehat{\mathbf{H}}$ this is the mapping $\widehat{\mathbf{q}}\mapsto\widehat{\mathbf{H}}[\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}]$, which is done efficiently by computing only the unique terms of the Kronecker product $\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}$.

:::{image} ../../images/opclasses-nonparametric.png
:align: center
:width: 80 %
:::

The entries of a _parametric operator_ depend on external parameters.
Parametric operator classes should inherit from `core.operators._base._BaseParametricOperator` and
1. Be initialized with matrices
2. Do shape checking on the matrices
3. Provide public access to the matrices (`.entries`, `.matrices`, etc.)
4. Implement `__call__(self, parameter)` so that it returns an non-parametric operator object:
```python
>>> parametric_operator = MyNewParametricOperator(init_args)
>>> nonparametric_operator = parametric_operator(parameter)
>>> isinstance(nonparametric_operator, _BaseNonparametricOperator)
True
```

Here's an example of the inheritance hierarchy for affine-parametric operators.

:::{image} ../../images/opclasses-affine.png
:align: center
:width: 80 %
:::

(subsec-contrib-romclass)=
### ROM Classes

The `_BaseROM` class of `core/_base.py` is the base class for all reduced-order models.
It handles
- Dimensions (`n`, `r`, and `m`),
- The basis (`basis`),
- Reduced-order operators (`c_`, `A_`, `H_`, `G_`, and `B_`),
- Projection (`project()`) and reconstruction (`reconstruct()`)
- Evaluation of the right-hand side of the ROM (`evaluate()`) and its Jacobian (`jacobian()`).

Classes that inherit from `_BaseROM` must (eventually) implement the following methods.
- `fit()`: Train the reduced-order model with the specified data.
- `predict()`: Solve the reduced-order model under specified conditions.
- `save()`: Save the reduced-order structure / operators locally in HDF5 format.
- `load()`: Load a previously saved reduced-order model from an HDF5 file. This should be a [`@classmethod`](https://stackoverflow.com/questions/136097/difference-between-staticmethod-and-classmethod).

To write a new ROM class, start with an intermediate base class that inherits from `_BaseROM` and implements `fit()`, `save()`, and `load()`.
See `core.nonparametric._base._NonparametricOpInfROM`, for example.
Then write classes that inherit from your intermediate base class for handling steady-state, discrete-time, and continuous-time problems by implementing `predict()`.

The following is the inheritance hierarchy for the standard operator classes.

:::{image} ../../images/romclasses-nonparametric.png
:align: center
:width: 80 %
:::

A _non-parametric ROM_ has exclusively non-parametric operators, i.e., their entries that do not depend on external parameters.
A _parametric ROM_ has one or more parametric operator.
Parametric ROM classes should inherit from `core._base._BaseParametricROM`, which adds an attribute `p` for the parameter space dimension and implements the following methods:

:::{margin}
```{note}
The `evaluate()` and `predict()` methods are already implemented in `_BaseParametricROM`, but it is recommended that you still implement them in your public-facing classes to tailor the arguments and provide accurate docstrings.
```
:::

- `__call__(self, parameter)` results in a parametric ROM whose operators correspond to the given parameter.
- `evaluate(self, parameter, *args, **kwargs)` evaluates the parametric ROM at the given parameter, then calls the resulting object's `evaluate()` mthod with the remaining arguments.
- `predict(self, parameter, *args, **kwargs)` evaluates the parametric ROM at the given parameter, then calls the resulting object's `predict()` method with the remaining arguments.

```python
>>> parametric_rom = MyNewParametricROM(init_args).fit(fit_args)
>>> nonparametric_rom = parametric_rom(parameter)
>>> isinstance(nonparametric_rom, _NonparametricOpInfROM)
True
```

The type of the parametric ROM evaluation should be one of the following classes in `core/nonparametric/_frozen.py`:
- `_FrozenSteadyROM` for steady-state problems
- `_FrozenDiscreteROM` for discrete-time problems
- `_FrozenContinuousROM` for continuous-time problems

These classes have their `fit()` method disabled.
This prevents the user from calling `fit()` on the evaluated parametric ROM (`nonparametric_rom` in the code block above) when they meant to fit the parametric ROM (`parametric_rom`).

For example, here is the inheritance hierarchy for the affine-parametric ROM classes.

:::{image} ../../images/romclasses-affine.png
:align: center
:width: 80 %
:::


---

(subsec-contrib-lstsqsolvers)=
## Least-squares Module

TODO
