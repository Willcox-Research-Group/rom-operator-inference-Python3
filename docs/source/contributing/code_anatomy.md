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

## Base Classes

This section documents private classes that are part of the class hierarchy but are not meant to be instantiated.
These are mostly abstract base classes or mixins.

### `pre`

```{eval-rst}
.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    opinf.pre._base._UnivarMixin
    opinf.pre._base._MultivarMixin
```

### `basis`

TODO

(subsec-contrib-opclass)=
### `operators`

```{eval-rst}
.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    opinf.operators._base._NonparametricOperator
    opinf.operators._base._ParametricOperator
    opinf.operators._interpolate._InterpolatedOperator
```

### `models`

```{eval-rst}
.. autosummary::
   :toctree: _autosummaries
   :nosignatures:

   opinf.models.mono._base._Model
   opinf.models.mono._nonparametric._NonparametricModel
   opinf.models.mono._parametric._ParametricModel
   opinf.models.mono._parametric._InterpolatedModel
```

### `lstsq`

```{eval-rst}
.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    opinf.lstsq._base._BaseSolver
    opinf.lstsq._tikhonov._BaseTikhonovSolver
```

<!-- Solver classes are defined in {mod}`opinf.lstsq`.
This module houses classes and tools for solving the least-squares problems that arise in operator inference, namely $\min_{\mathbf{X}}\|\A\mathbf{X} - \B\|$ plus regularization and/or constraints.
Each solver class defines 1) how the problem is posed (e.g., is there regularization and how is it structured, are there constraints and if so what are they, etc.) and 2) how the problem is to be solved (e.g., use the SVD, use the QR decomposition, solve the normal equations, etc.).

The `fit()` method of each model class has a `solver` keyword argument which accepts a `lstsq` solver object to apply to the associated operator inference least-squares problem.

Least-squares solver classes must do the following.

- Inherit from {class}`opinf.lstsq._base._BaseSolver`.
- Accept any hyperparameters (e.g., regularization values) in the constructor.
- Have a `fit(A, B)` method that calls `_BaseSolver.fit(A, B)` and sets up the least-squares problem.
- Have a `predict()` method returns the solution `X` to the least-squares problem $||\mathbf{AX} - \B||$ (+ regularization, etc.).
- Have a `_LSTSQ_LABEL` class attribute that is a string describing the form of the least-squares problem, e.g., `"min_{X} ||AX - B||_F^2 + ||µX||_F^2"` This is used in the string representation of the class. -->
