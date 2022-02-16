(sec-contrib-anatomy)=
# Source Code Guide

This page is a non-exhaustive overview of the package anatomy.
The source code is stored in [`src/rom_operator_inference/`](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/tree/develop/src/rom_operator_inference) and is divided into the following submodules:
- `core`: operator and reduced-order model classes.
- `lstsq`: solvers for the regression problem at the heart of Operator Inference.
- `pre`: pre-processing tools (basis computation, state transformations, etc.).
- `post`: post-processing tools (mostly error evaluation).
- `utils`: other routines that don't obviously fit into another submodule. These functions are usually not important for casual users, but advanced users and developers may need access to them.

## Conventions

### Object-oriented Philosophy

Python is a highly [object-oriented language](https://docs.python.org/3/tutorial/classes.html), which is advantageous for building mathematical software with abstract objects.
One-off routines (such as computing a POD basis) should be implemented as standalone functions, but a class structure is preferable when possible.
The package has three main class hierarchies:

- [_ROM classes_](subsec-contrib-romclass) encapsulate an entire reduced-order model, such as $\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t) = \widehat{\mathbf{A}}\widehat{\mathbf{q}}(t) + \widehat{\mathbf{B}}\mathbf{u}(t)$ or $\widehat{\mathbf{q}}_{j+1} = \widehat{\mathbf{H}}[\widehat{\mathbf{q}}_{j} \otimes \widehat{\mathbf{q}}_{j}]$. These are the major objects that the user interacts with.
- [_Operator classes_](subsec-contrib-opclass) represent a single operator that forms part of a reduced-order model, e.g., $\widehat{\mathbf{A}}$ or $\widehat{\mathbf{H}}$. Every ROM object has, as attributes, several operator classes.
- [_Solver classes_](subsec-contrib-solverclass) handle the regression problem at the heart of Operator Inference.

### Scikit-Learn Style

We strive to follow [`scikit-learn` conventions](https://scikit-learn.org/stable/developers/develop.html#api-overview) and other paradigms that are common in machine learning Python libraries.
Specifically, both ROM classes and solver classes follow these rules.
- The `fit()` method receives training data and solves an inference problem.
- Attributes learned during `fit()` end with an underscore, in particular the reduced-order operators `c_`, `A_`, and so on. For consistency, low-dimensional quantities also end with an underscore (e.g., `state` is high dimensional but `state_` is low dimensional).
- The `predict()` method receives a scenario (e.g., initial conditions) and uses information learned during `fit()` to make a prediction for the specified scenario.

In addition, we follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and other standard Python formatting guidelines.
See [Linting](sec-contrib-linting) for more.

## Core Module

(subsec-contrib-opclass)=
### Operator Class Hierarchy

The operator classes represent single operators that forms part of a reduced-order model, e.g., $\widehat{\mathbf{A}}$ or $\widehat{\mathbf{H}}(\mu)$.
They are defined in `core/operators`.

#### Non-parametric Operators

These classes, defined in `core/operators/_nonparametric.py`, define operators whose entries are constant, i.e., that do not depend on external parameters.
They are initialized with a NumPy array (the entries of the operator), which can be accessed as the `entries` attribute.
These classes also handle the mappings defined by the operator: for the linear term $\widehat{\mathbf{A}}$ this is simply matrix multiplication $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{A}}\widehat{\mathbf{q}}$; for the quadratic term $\widehat{\mathbf{H}}$ this is the mapping $\widehat{\mathbf{q}}\mapsto\widehat{\mathbf{H}}[\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}]$, which is done efficiently by computing only the unique terms of the Kronecker product $\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}$.

Parametric operators, whose entries depend on external parameters, should be callable and evaluate to a non-parametric operator object.

#### Affine-parametric Operators

An operator $\mathbf{A}(\boldsymbol{\mu}) \in \mathbb{R}^{r \times s}$ depending on the parameter $\boldsymbol{\mu}\in\mathbb{R}^{d_\mu}$ is called _affine_ if it can be written as the sum

\begin{align*}
    \widehat{\mathbf{A}}(\boldsymbol{\mu})
    &= \sum_{p=1}^{\ell}
        \theta^{(p)}(\boldsymbol{\mu})\widehat{\mathbf{A}}^{(p)},
    &
    \theta^{(p)} &:\mathbb{R}^{d_\mu} \to \mathbb{R},
    &
    \widehat{\mathbf{A}}^{(p)} \in \mathbb{R}^{r \times s}.
\end{align*}

TODO

#### Interpolated Operators

(subsec-contrib-romclass)=
### ROM Class Hierarchy

The `_BaseROM` class of `core/_base.py` is the base class for all ROMs of the form

TODO

## Least-squares Module

(subsec-contrib-solverclass)=
### Class Hierarchy
