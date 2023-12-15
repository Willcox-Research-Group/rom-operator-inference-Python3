# Source Code Guide

This page details the source code class hierarchy.
The source code is stored in [`src/opinf/`](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/tree/main/src/opinf).

:::{tip}
Recommended reading: the [Scientific Python Development Guide](https://learn.scientific-python.org/development/principles/design/).
:::

## Object-oriented Philosophy

Python is a highly [object-oriented language](https://docs.python.org/3/tutorial/classes.html), which is advantageous for building mathematical software with abstract objects.
One-off routines may be implemented as standalone functions, but in `opinf` a class hierarchy is often advantageous.

The `opinf` package defines the following main class hierarchies:

- [_Transformer classes_](subsec-contrib-transformerclass) are used for normalizing snapshot data (see {mod}`opinf.pre`).
- [_Basis classes_](subsec-contrib-basisclass) represent the mapping between the full-order state space in $\RR^{n}$ and the reduced-order state space in $\RR^{r}$ (see {mod}`opinf.basis`)
- [_Model classes_](subsec-contrib-modelclass) encapsulate an entire reduced-order model, such as $\frac{\text{d}}{\text{d}t}\qhat(t) = \Ahat\qhat(t) + \Bhat\u(t)$ or $\qhat_{j+1} = \Hhat[\qhat_{j} \otimes \qhat_{j}]$. These are the major objects that the user interacts with (see {mod}`opinf.models`).
- [_Operator classes_](subsec-contrib-opclass) represent a single operator that forms part of a reduced-order model, e.g., $\Ahat\q$ or $\Hhat[\qhat\otimes\qhat]$. Every ROM object has an `operators` attribute that is a list of operator objects (see {mod}`opinf.operators`).
- [_Solver classes_](subsec-contrib-lstsqsolvers) handle the least-squares regression problem at the heart of Operator Inference (see {mod}`opinf.lstsq`).

:::{tip}
As you design new classes, take advantage of intermediate classes and inheritance to avoid duplicating code when appropriate.
For example, the operator classes {class}`opinf.operators.LinearOperator` and {class}`opinf.operators.QuadraticOperator` both inherit from {class}`opinf.operators._base._NonparametricOperator`, which handles tasks that are common to all nonparametric operators.
:::

(subsec-contrib-transformerclass)=
## Transformer Classes

```{eval-rst}
.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    opinf.pre._base._BaseTransformer
```

:::{admonition} TODO
Discuss multivariable transformers / bases
:::

Transformer classes are defined in {mod}`opinf.pre`.
All transformer classes must

- Inherit from {class}`opinf.pre._base._BaseTransformer`.
- Accept and store any hyperparameters (transformation settings) in the constructor.
- Implement `transform()`, `fit_transform()`, and `inverse_transform()`. Note that `fit()` is already implemented in `_BaseTransformer` and should work as long as `fit_transform()` is implemented.

(subsec-contrib-basisclass)=
## Basis Classes

```{eval-rst}
.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    opinf.basis._base._BaseBasis
```

Basis classes are defined in {mod}`opinf.basis`.
All basis classes must

- Inherit from {class}`opinf.basis._base._BaseBasis`.
- Accept and store any hyperparameters in the constructor.
- Implement `fit()`, `compress()`, and `decompress()`.

:::{versionchanged} 0.4.4
Note that `project()` is **not** the same as `compress()`, in fact `project(q) = decompress(compress(q))`.
This is a name change from previous versions.
:::

Before implementing a new basis class, take a close look at {class}`opinf.basis.LinearBasis` and then {class}`opinf.basis.PODBasis`.

(subsec-contrib-opclass)=
## Operator Classes

```{eval-rst}
.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    opinf.operators._base._NonparametricOperator
    opinf.operators._base._ParametricOperator
    opinf.operators._interpolate._InterpolatedOperator
```

Operator classes are defined in {mod}`opinf.operators`.
The first step for implementing a new kind of reduced-order model is correctly implementing the individual operators that define the terms of the model, for example $\Ahat$ and $\Bhat$ of $\ddt\qhat(t) = \Ahat\qhat(t) + \Bhat\u(t)$.

All [nonparametric operator](sec-operators-nonparametric) classes must

- Inherit from {class}`opinf.operators._base._NonparametricOperator`.
- TODO

All [parametric operator](sec-operators-parametric) classes must

- Inherit from {class}`opinf.operators._base._ParametricOperator`.

:::{tip}
Usually a group of new parametric operator classes can be implemented by writing an intermediate class that does all of the work, then writing child classes for constant, linear, quadratic, and cubic terms which each set `_OperatorClass` appropriately.
For example, here is a simplified version of `_AffineOperator`:

```python
class _AffineOperator(_BaseParametricOperator):
    """Base class for parametric operators with affine structure."""

    def __init__(self, coefficient_functions, matrices=entries):
        """Save the affine coefficient functions."""
        self.coefficient_functions = coefficient_functions
        if entries is not None:
            self.entries = entries

    def evaluate(self, parameter):
        """Evaluate the affine operator at the given parameter."""
        entries = sum(
            [
                theta(parameter) * A
                for theta, A in zip(self.coefficient_functions, self.entries)
            ]
        )
        return self.OperatorClass(entries)
```

Note that `evaluate()` uses `self.OperatorClass` to wrap the entries of the evaluated operator.
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

(subsec-contrib-modelclass)=
## Model Classes

To write a new ROM class, start with an intermediate base class that inherits from `_BaseROM` and implements `fit()`, `save()`, and `load()`.
See `roms.nonparametric._base._NonparametricROM`, for example.
Then write classes that inherit from your intermediate base class for handling steady-state, discrete-time, and continuous-time problems by implementing `predict()`.

:::{tip}
Similar to parametric operators, a group of new parametric ROM classes can usually be implemented by writing an intermediate class that does most of the work, then writing child classes for the discrete and continuous settings which each set `_ModelClass` appropriately.
<!-- For instance, the intermediate class can often implement `fit()`, `save()`, and `load()`, and `_BaseParametricROM` implements `evaluate()`, `predict()`, and `jacobian()` already.
However, it is important to have tailored signatures and detailed docstrings for `fit()` and `predict()`, so these should be carefully defined for every public-facing class. -->
:::

(subsec-contrib-lstsqsolvers)=
## Least-squares Solvers

```{eval-rst}
.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    opinf.lstsq._base._BaseSolver
    opinf.lstsq._tikhonov._BaseTikhonovSolver
```

Solver classes are defined in {mod}`opinf.lstsq`.
This module houses classes and tools for solving the least-squares problems that arise in operator inference, namely $\min_{\mathbf{X}}\|\A\mathbf{X} - \B\|$ plus regularization and/or constraints.
Each solver class defines 1) how the problem is posed (e.g., is there regularization and how is it structured, are there constraints and if so what are they, etc.) and 2) how the problem is to be solved (e.g., use the SVD, use the QR decomposition, solve the normal equations, etc.).

The `fit()` method of each model class has a `solver` keyword argument which accepts a `lstsq` solver object to apply to the associated operator inference least-squares problem.

Least-squares solver classes must do the following.

- Inherit from {class}`opinf.lstsq._base._BaseSolver`.
- Accept any hyperparameters (e.g., regularization values) in the constructor.
- Have a `fit(A, B)` method that calls `_BaseSolver.fit(A, B)` and sets up the least-squares problem.
- Have a `predict()` method returns the solution `X` to the least-squares problem $||\mathbf{AX} - \B||$ (+ regularization, etc.).
- Have a `_LSTSQ_LABEL` class attribute that is a string describing the form of the least-squares problem, e.g., `"min_{X} ||AX - B||_F^2 + ||µX||_F^2"` This is used in the string representation of the class.

See {class}`opinf.lstsq.L2Solver` for an example.
