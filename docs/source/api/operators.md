# `opinf.operators`

```{eval-rst}
.. automodule:: opinf.operators

.. currentmodule:: opinf.operators

**Nonparametric Operators**

.. autosummary::
   :toctree: _autosummaries
   :nosignatures:

   OperatorTemplate
   OpInfOperator
   ConstantOperator
   LinearOperator
   QuadraticOperator
   CubicOperator
   InputOperator
   StateInputOperator

**Parametric Operators**

.. autosummary::
   :toctree: _autosummaries
   :nosignatures:

   ParametricOperatorTemplate
   ParametricOpInfOperator
   InterpolatedConstantOperator
   InterpolatedLinearOperator
   InterpolatedQuadraticOperator
   InterpolatedCubicOperator
   InterpolatedInputOperator
   InterpolatedStateInputOperator
```

:::{admonition} Overview
:class: note

- Operator classes defined in {mod}`opinf.operators` represent the individual terms in a dynamic model equation.
  - [`apply()`](OperatorTemplate.apply) applies the operator to given state and input vectors.
  - [`jacobian()`](OperatorTemplate.apply) constructs the derivative of the operator with respect to the state at given state and input vectors.
- Operators that can be written as the product of a matrix and a known vector-valued function can be calibrated through a data-driven inference problem.
  - [`operator_dimension()`](OpInfOperator.operator_dimension) defines the size of the operator entries matrix for given state and input dimensions.
  - [`datablock()`](OpInfOperator.datablock) uses state and input snapshots to construct a block of the data matrix for the inference problem.
- A list of operator objects is passed to the constructor of {mod}`opinf.models` classes.
- [Nonparametric operators](sec-operators-nonparametric) do not depend on external parameters, while [parametric operators](sec-operators-parametric) have a dependence on one or more external parameters.
:::

<!-- - Monolithic operators are designed for dense systems; multilithic operators are designed for systems with sparse block structure. -->

## Operators

Models based on Operator Inference are systems of [ordinary differential equations](opinf.models.ContinuousModel) (or [discrete-time difference equations](opinf.models.DiscreteModel)) that can be written as a sum of terms,

$$
\begin{aligned}
   \ddt\qhat(t)
   = \sum_{\ell=1}^{n_\textrm{terms}}\Ophat_{\ell}(\qhat(t),\u(t)),
\end{aligned}
$$ (eq:operators:model)

where each $\Ophat_{\ell}:\RR^{r}\times\RR^{m}\to\RR^{r}$ is a vector-valued function of the reduced state $\qhat\in\RR^{r}$ and the input $\u\in\RR^{m}$.
We call these functions *operators* on this page.

Operator Inference calibrates operators that can be written as the product of a matrix $\Ohat\in\RR^{r \times d}$ and a known (possibly nonlinear) vector-valued function $\d_{\ell}:\RR^{r}\times\RR^{m}\to\RR^{d}$,

$$
\begin{aligned}
    \Ophat_{\ell}(\qhat,\u)
    = \Ohat_{\ell}\d_{\ell}(\qhat,\u).
\end{aligned}
$$

We call these ``OpInf operators'' on this page.
The goal of Operator Inference is to learn each *operator entries* matrix $\Ohat_\ell$ for each OpInf operator in the model.

This module defines classes for various types of operators.

:::{admonition} Example
:class: tip

To represent a linear time-invariant (LTI) system

$$
\begin{align}
    \ddt\qhat(t)
    = \Ahat\qhat(t) + \Bhat\u(t),
    \qquad
    \Ahat\in\RR^{r \times r},
    ~
    \Bhat\in\RR^{r \times m},
\end{align}
$$ (eq:operators:ltiexample)

we use the following operator classes.

| Class | Definition | Operator entries | data vector |
| :---- | :--------- | :--------------- | :---------- |
| {class}`LinearOperator` | $\Ophat_{1}(\qhat,\u) = \Ahat\qhat$ | $\Ohat_{1} = \Ahat \in \RR^{r\times r}$ | $\d_{1}(\qhat,\u) = \qhat\in\RR^{r}$ |
| {class}`InputOperator` | $\Ophat_{2}(\qhat,\u) = \Bhat\u$ | $\Ohat_{2} = \Bhat \in \RR^{r\times m}$ | $\d_{2}(\qhat,\u) = \u\in\RR^{m}$ |

An {class}`opinf.models.ContinuousModel` object can be instantiated with a list of operators objects to represent {eq}`eq:operators:ltiexample` as

$$
\begin{aligned}
    \ddt\qhat(t)
    = \Ophat_{1}(\qhat(t),\u(t))
    + \Ophat_{2}(\qhat(t),\u(t)).
\end{aligned}
$$

```python
import opinf

LTI_model = opinf.models.ContinuousModel(
    operators=[
        opinf.operators.LinearOperator(),
        opinf.operators.InputOperator(),
    ]
)
```

:::

(sec-operators-nonparametric)=
## Nonparametric Operators

A _nonparametric_ operator is a function of the state and input only.
For OpInf operators $\Ophat_{\ell}(\qhat,\u) = \Ohat_{\ell}\d_{\ell}(\qhat,\u)$, this means the entries matrix $\Ohat_\ell$ is constant.
See [Parametric Operators](sec-operators-parametric) for operators that also depend on one or more external parameters.

Nonparametric operator classes inherit from {class}`OperatorTemplate`.

Nonparametric operators can be instantiated without arguments.
If the operator entries are known, they can be passed into the constructor or set later with [`set_entries()`](OpInfOperator.set_entries).
The entries are stored as the [`entries`](OpInfOperator.entries) attribute and can be accessed with slicing operations `[:]`.

There are two ways to determine the operator entries:

- Learn the entries from data (non-intrusive Operator Inference), or
- Shrink an existing high-dimensional operator (intrusive Galerkin projection).

Once the entries are set, the following methods are used to compute the action
of the operator or its derivatives.

- `apply()`: compute the operator action $\Ophat_\ell(\qhat, \u)$.
- `jacobian()`: construct the state Jacobian $\ddqhat\Ophat_\ell(\qhat, \u)$.

(sec-operators-calibration)=
### Learning Operators from Data

Suppose we have state-input-derivative data triples $\{(\qhat_j,\u_j,\dot{\qhat}_j)\}_{j=0}^{k-1}$ that approximately satisfy the model {eq}`eq:operators:model`, i.e.,

$$
\begin{aligned}
    \dot{\qhat}_j
    \approx \Ophat(\qhat_j, \u_j)
    = \sum_{\ell=1}^{n_\textrm{terms}} \Ophat_{\ell}(\qhat_j, \u_j)
    = \sum_{\ell=1}^{n_\textrm{terms}} \Ohat_{\ell}\d_{\ell}(\qhat_j, \u_j).
\end{aligned}
$$ (eq:operators:approx)

Operator Inference determines the operator entries $\Ohat_1,\ldots,\Ohat_{n_\textrm{terms}}$ by minimizing the residual of {eq}`eq:operators:approx`:

$$
\begin{aligned}
    \min_{\Ohat_1,\ldots,\Ohat_{n_\textrm{terms}}}\sum_{j=0}^{k-1}\left\|
        \sum_{\ell=1}^{n_\textrm{terms}}\Ohat_\ell\d_\ell(\qhat_j,\u_j) - \dot{\qhat}_j
    \right\|_2^2.
\end{aligned}
$$

To facilitate this, nonparametric operator classes have a static `datablock()` method that, given the state-input data pairs $\{(\qhat_j,\u_j)\}_{j=0}^{k-1}$, forms the matrix

$$
\begin{aligned}
    \D_{\ell}\trp = \left[\begin{array}{c|c|c|c}
        & & & \\
        \d_{\ell}(\qhat_0,\u_0) & \d_{\ell}(\qhat_1,\u_1) & \cdots & \d_{\ell}(\qhat_{k-1},\u_{k-1})
        \\ & & &
    \end{array}\right]
    \in \RR^{d \times k}.
\end{aligned}
$$

Then {eq}`eq:operators:approx` can be written in the linear least-squares form

$$
\begin{aligned}
    \min_{\Ohat_1,\ldots,\Ohat_{n_\textrm{terms}}}\sum_{j=0}^{k-1}\left\|
        \sum_{\ell=1}^{n_\textrm{terms}}\Ohat_\ell\d_\ell(\qhat_j,\u_j) - \dot{\qhat}_j
    \right\|_2^2
    = \min_{\Ohat}\left\|
        \D\Ohat\trp - [~\dot{\qhat}_0~~\cdots~~\dot{\qhat}_{k-1}~]\trp
    \right\|_F^2,
\end{aligned}
$$

where the complete operator matrix $\Ohat$ and data matrix $\D$ are concatenations of the operator and data matrices from each operator:

$$
\begin{aligned}
    \Ohat = \left[\begin{array}{ccc}
        & & \\
        \Ohat_1 & \cdots & \Ohat_{n_\textrm{terms}}
        \\ & &
    \end{array}\right],
    \qquad
    \D = \left[\begin{array}{ccc}
        & & \\
        \D_1 & \cdots & \D_{n_\textrm{terms}}
        \\ & &
    \end{array}\right].
\end{aligned}
$$

Model classes from {mod}`opinf.models` are instantiated with a list of operators.
The model's `fit()` method calls the `datablock()` method of each operator to assemble the full data matrix $\D$, solves the regression problem for the full data matrix $\Ohat$ (see {mod}`opinf.lstsq`), and sets the entries of the $\ell$-th operator to $\Ohat_{\ell}$.

:::{admonition} Example
:class: tip

For the LTI system {eq}`eq:operators:ltiexample`, the operator inference problem is the following regression.

$$
\begin{aligned}
    \min_{\Ahat,\Bhat}\sum_{j=0}^{k-1}\left\|
        \Ahat\qhat_j + \Bhat\u_j - \dot{\qhat}_j
    \right\|_2^2
    = \min_{\Ohat}\left\|
        \D\Ohat\trp - [~\dot{\qhat}_0~~\cdots~~\dot{\qhat}_{k-1}~]\trp
    \right\|_F^2,
\end{aligned}
$$

with operator matrix $\Ohat=[~\Ahat~~\Bhat~]$
and data matrix $\D = [~\Qhat\trp~~\U\trp~]$
where $\Qhat = [~\qhat_0~~\cdots~~\qhat_{k-1}~]$
and $\U = [~\u_0~~\cdots~~\u_{k-1}~]$.
:::

:::{important}
Only operators whose entries are _not initialized_ (set to `None`) when a model is constructed are learned with Operator Inference when `fit()` is called.
For example, suppose for the LTI system {eq}`eq:operators:ltiexample` an appropriate input matrix $\Bhat$ is known and stored as the variable `B_`.

```python
import opinf

LTI_model = opinf.models.ContinuousModel(
    operators=[
        opinf.operators.LinearOperator(),   # No entries specified.
        opinf.operators.InputOperator(B_),  # Entries set to B_.
    ]
)
```

In this case, `LIT_model.fit()` only determines the entries of the {class}`LinearOperator` object using Operator Inference, with regression problem

$$
\begin{aligned}
    &\min_{\Ahat,}\sum_{j=0}^{k-1}\left\|
        \Ahat\qhat_j - (\dot{\qhat}_j - \Bhat\u_j)
    \right\|_2^2
    \\
    &= \min_{\Ohat}\left\|
        \Qhat\trp\Ahat\trp - [~(\dot{\qhat}_0 - \Bhat\u_0)~~\cdots~~(\dot{\qhat}_{k-1} - \Bhat\u_{k-1})~]\trp
    \right\|_F^2.
\end{aligned}
$$

:::

### Learning Operators via Projection

The goal of Operator Inference is to learn operator entries from data because full-order operators are unknown or computationally inaccessible.
However, in some scenarios a subset of the full-order model operators are known, in which case the corresponding reduced-order model operators can be determined through *intrusive projection*.
Consider a full-order operator $\Op:\RR^{n}\times\RR^{m}\to\RR^{n}$, written $\Op(\q,\u)$, where

- $\q\in\RR^n$ is the full-order state, and
- $\u\in\RR^m$ is the input.

Given a *trial basis* $\Vr\in\RR^{n\times r}$ and a *test basis* $\Wr\in\RR^{n\times r}$, the corresponding intrusive projection of $\Op$ is the operator $\Ophat:\RR^{r}\times\RR^{m}\to\RR^{r}$ defined by

$$
\begin{aligned}
    \Ophat(\qhat, \u) = (\Wr\trp\Vr)^{-1}\Wr\trp\Op(\Vr\qhat, \u)
\end{aligned}
$$

where
- $\qhat\in\RR^{r}$ is the reduced-order state, and
- $\u\in\RR^{m}$ is the input (the same as before).

This approach uses the low-dimensional state approximation $\q = \Vr\qhat$.
If $\Wr = \Vr$, the result is called a *Galerkin projection*.
Note that if $\Vr$ has orthonormal columns, we have in this case the simplification

$$
    \Ophat(\qhat, \u) = \Vr\trp\Op(\Vr\qhat, \u).
$$

If $\Wr \neq \Vr$, the result is called a *Petrov-Galerkin projection*.

:::{admonition} Example
:class: tip

Consider the bilinear operator
$\Op(\q,\u) = \N[\u\otimes\q]$ where $\N\in\RR^{n \times nm}$.
The intrusive Petrov-Galerkin projection of $\Op$ is the bilinear operator

$$
\begin{aligned}
    \Ophat(\qhat,\u)
    = (\Wr\trp\Vr)^{-1}\Wr\trp\N[\u\otimes\Vr\qhat]
    = \Nhat[\u\otimes\qhat]
\end{aligned}
$$

where $\Nhat = (\Wr\trp\Vr)^{-1}\Wr\trp\N(\I_m\otimes\Vr) \in \RR^{r\times rm}$.
The intrusive Galerkin projection has $\Nhat = \Vr\trp\N(\I_m\otimes\Vr)$.
:::

Every operator class has a `galerkin()` method that performs intrusive projection.

### Custom Nonparametric Operators

New nonparametric operators can be defined by inheriting from {class}`OperatorTemplate` or, for operators that can be calibrated through Operator Inference, {class}`OpInfOperator`.
Once implemented, the [`verify()`](OperatorTemplate.verify) method may be used to test for consistency between [`apply()`](OperatorTemplate.apply) and the other methods outlined below.

```python
class MyOperator(opinf.operators.OperatorTemplate):
    """Custom non-OpInf nonparametric operator."""

    # Constructor -------------------------------------------------------------
    def __init__(self, args_and_kwargs):
        """Construct the operator and set the state_dimension."""
        raise NotImplementedError

    # Required properties and methods -----------------------------------------
    @property
    def state_dimension(self):
        """Dimension of the state that the operator acts on."""
        return NotImplemented

    def apply(self, state, input_=None):
        """Apply the operator to the given state / input."""
        raise NotImplementedError

    # Optional methods --------------------------------------------------------
    def jacobian(self, state, input_=None):
        """Construct the state Jacobian of the operator."""
        raise NotImplementedError

    def galerkin(self, Vr: np.ndarray, Wr=None):
        """Get the (Petrov-)Galerkin projection of this operator."""
        raise NotImplementedError

    def save(self, savefile, overwrite=False):
        """Save the operator to an HDF5 file."""
        raise NotImplementedError

    @classmethod
    def load(cls, loadfile):
        """Load an operator from an HDF5 file."""
        raise NotImplementedError

    def copy(self):
        """Make a copy of the operator.
        If not implemented, copy.deepcopy() is used.
        """
        raise NotImplementedError
```

See {class}`OperatorTemplate` for details on the arguments for each method.

```python
class MyOpInfOperator(opinf.operators.OpInfOperator):
    """Custom nonparametric OpInf operator."""

    # Required methods --------------------------------------------------------
    @opinf.utils.requires("entries")
    def apply(self, state=None, input_=None):
        """Apply the operator to the given state / input."""
        raise NotImplementedError

    @staticmethod
    def datablock(states, inputs=None):
        """Return the data matrix block corresponding to the operator."""
        raise NotImplementedError

    @staticmethod
    def operator_dimension(r=None, m=None):
        """Column dimension of the operator entries matrix."""
        raise NotImplementedError

    # Optional methods --------------------------------------------------------
    @opinf.utils.requires("entries")
    def jacobian(self, state, input_=None):
        """Construct the state Jacobian of the operator.
        NOTE: If this method is omitted it is assumed that the Jacobian is
        zero, implying that the operator does not depend on the state.
        """
        raise NotImplementedError

    def galerkin(self, Vr, Wr=None):
        """Get the (Petrov-)Galerkin projection of this operator."""
        raise NotImplementedError
```

See {class}`OpInfOperator` for details on the arguments for each method.

:::{admonition} Developer Notes
:class: note

- If the operator depends on the input $\u$, the class should also inherit from {class}`InputMixin` and set the [`input_dimension`](InputMixin.input_dimension) attribute.
- The [`jacobian()`](OperatorTemplate.jacobian) method is optional, but {mod}`opinf.models` objects have a `jacobian()` method that calls `jacobian()` for each of its operators. In an {class}`opinf.models.ContinuousModel`, the Jacobian is required for various time integration strategies used in [`predict()`](opinf.models.ContinuousModel.predict).
- The [`galerkin()`](OperatorTemplate.galerkin) method is optional, but {mod}`opinf.models` objects have a `galerkin()` method that calls `galerkin()` for each of its operators.
- The [`save()`](OperatorTemplate.save) and [`load()`](OperatorTemplate.load) methods should be implemented using {func}`opinf.utils.hdf5_savehandle()` and {func}`opinf.utils.hdf5_loadhandle()`, respectively.
:::

#### Example: Hadamard Product with a Fixed Vector

Consider the operator $\Ophat_{\ell}(\qhat, \u) = \qhat \ast \hat{\s}$, where $\hat{\s}\in\RR^{r}$ is a constant vector and $\ast$ denotes the [Hadamard (elementwise) product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)) (`*` in NumPy).
To implement an operator for this class, we first calculate its state Jacobian and determine the operator produced by (Petrov--)Galerkin projection.

Let $\qhat = [~\hat{q}_1~~\cdots~~\hat{q}_r~]\trp$ and $\hat{\s} = [~\hat{s}_1~~\cdots~~\hat{s}_r~]\trp$, i.e., the $i$-th entry of $\Ophat_{\ell}(\qhat, \u)$ is $\hat{q}_i\hat{s}_i$.
Then the $(i,j)$-th entry of the Jacobian is

$$
\begin{aligned}
    \frac{\partial}{\partial \hat{\q}_j}\left[\hat{q}_i\hat{s}_i\right]
    = \begin{cases}
        \hat{s}_i & \textrm{if}~i = j,
        \\
        0 & \textrm{else}.
    \end{cases}
\end{aligned}
$$

That is, $\ddqhat\Ophat_{\ell}(\qhat, \u) = \operatorname{diag}(\hat{\s}).$

Now consider a version of this operator with a large state dimension, $\Op_{\ell}(\q, \u) = \q \ast \s$ for $\q,\s\in\mathbb{R}^{n}$.
For basis matrices $\Vr,\Wr\in\mathbb{R}^{n \times r}$, the Petrov-Galerkin projection of $\Op_{\ell}$ is given by

$$
\begin{aligned}
    \Ophat_{\ell}(\qhat, \u)
    = (\Wr\trp\Vr)^{-1}\Wr\trp\Op_{\ell}(\Vr\qhat, \u)
    = (\Wr\trp\Vr)^{-1}\Wr\trp((\Vr\qhat)\ast\s).
\end{aligned}
$$

It turns out that this product can be written as a matrix-vector product $\Ahat\qhat$ where $\Ahat$ depends on $\Vr$ and $\s$.
Therefore, `galerkin()` should return a {class}`LinearOperator` with entries matrix $\Ahat$.

The following class inherits from {class}`OperatorTemplate`, stores $\hat{\s}$ and sets the state dimension $r$ in the constructor, and implements the methods outlined the inheritance template.

```python
class HadamardOperator(opinf.operators.OperatorTemplate):
    """Custom non-OpInf nonparametric operator for the Hadamard product."""

    # Constructor -------------------------------------------------------------
    def __init__(self, s):
        """Construct the operator and set the state_dimension."""
        self.svector = np.array(s)
        self._jac = np.diag(self.svector)

    # Required properties and methods -----------------------------------------
    @property
    def state_dimension(self):
        """Dimension of the state that the operator acts on."""
        return self.svector.shape[0]

    def apply(self, state, input_=None):
        """Apply the operator to the given state / input."""
        svec = self.svector
        if state.ndim == 2:
            svec = svec.reshape((self.state_dimension, 1))
        return state * svec

    # Optional methods --------------------------------------------------------
    def jacobian(self, state, input_=None):
        """Construct the state Jacobian of the operator."""
        return self._jac

    def galerkin(self, Vr, Wr=None):
        """Get the (Petrov-)Galerkin projection of this operator."""
        if Wr is None:
            Wr = Vr
        n = self.state_dimension
        r = Vr.shape[1]

        M = la.khatri_rao(V.T, np.eye(n)).T
        Ahat = Wr.T @ M.reshape((n, r, n)) @ self.svector
        if not np.allclose((WrTVr := Wr.T @ Vr), np.eye(r)):
            Ahat = la.solve(WrTVr, entries)
        return opinf.operators.LinearOperator(Ahat)

    def save(self, savefile, overwrite=False):
        """Save the operator to an HDF5 file."""
        with utils.hdf5_savehandle(savefile, overwrite) as hf:
            hf.create_dataset("svector", data=self.svector)

    @classmethod
    def load(cls, loadfile):
        """Load an operator from an HDF5 file."""
        with utils.hdf5_loadhandle(loadfile) as hf:
            return cls(hf["svector"][:])

    def copy(self):
        """Make a copy of the operator."""
        return self.__class__(self.svector)
```

(sec-operators-parametric)=
## Parametric Operators

Operators are called _parametric_ if the operator entries depend on an independent parameter vector
$\bfmu\in\RR^{p}$, i.e., $\Ophat_{\ell}(\qhat,\u;\bfmu) = \Ohat_{\ell}(\bfmu)\d_{\ell}(\qhat,\u)$ where now $\Ohat:\RR^{p}\to\RR^{r\times d}$.

:::{admonition} Example
:class: tip
Let $\bfmu = [~\mu_{1}~~\mu_{2}~]\trp$.
The linear operator
$\Ophat_1(\qhat,\u;\bfmu) = (\mu_{1}\Ahat_{1} + \mu_{2}\Ahat_{2})\qhat$
is a parametric operator with parameter-dependent entries $\Ohat_1(\bfmu) = \mu_{1}\Ahat_{1} + \mu_{2}\Ahat_{2}$.
:::

(sec-operators-interpolated)=
### Interpolated Operators

These operators handle the parametric dependence on $\bfmu$ by using elementwise interpolation:

$$
\begin{aligned}
    \Ohat_{\ell}(\bfmu)
    = \text{interpolate}(
    (\bfmu_{1},\Ohat_{\ell}^{(1)}),\ldots,(\bfmu_{s},\Ohat_{\ell}^{(s)}); \bfmu),
\end{aligned}
$$

where $\bfmu_1,\ldots,\bfmu_s$ are training parameter values and $\Ohat_{\ell}^{(i)} = \Ohat_{\ell}(\bfmu_i)$ for $i=1,\ldots,s$.

```{eval-rst}
.. currentmodule:: opinf.operators

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    InterpolatedConstantOperator
    InterpolatedLinearOperator
    InterpolatedQuadraticOperator
    InterpolatedCubicOperator
    InterpolatedInputOperator
    InterpolatedStateInputOperator
```

<!-- ### Affine Operators

$$
\begin{aligned}
    \Ophat(\qhat,\u;\bfmu)
    = \sum_{\ell=1}^{n_{\theta}}\theta_{\ell}(\bfmu)\Ophat_{\ell}(\qhat,\u)
\end{aligned}
$$

:::{admonition} TODO
Constructor takes in list of the affine coefficient functions.
::: -->

## Utilities

```{eval-rst}
.. currentmodule:: opinf.operators

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    is_nonparametric
    is_parametric
    has_inputs
```
