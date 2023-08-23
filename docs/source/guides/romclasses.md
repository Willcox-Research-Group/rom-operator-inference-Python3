(sec-romclasses)=
# ROM Classes

There are several reduced-order model (ROM) classes defined in the main namespace of `opinf`.
Each class corresponds to a specific problem setting: _continuous_ models are ordinary differential equations while _discrete_ models are discrete dynamical systems; in _parametric_ models, the equations may depend on an external parameter.

::::{margin}
:::{tip}
The API for these classes adopts some principles from the [scikit-learn](https://scikit-learn.org/stable/index.html) [API](https://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects): there are `fit()` and `predict()` methods, hyperparameters are set in the constructor, estimated attributes end with underscore, and so on.
:::
::::

| Class Name | Reduced-order Model |
| :--------- | :-----------------: |
| [**ContinuousOpInfROM**](opinf.ContinuousOpInfROM) | $\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t) = \widehat{\mathbf{F}}(t, \widehat{\mathbf{q}}(t), \mathbf{u}(t))$ |
| [**DiscreteOpInfROM**](opinf.DiscreteOpInfROM) | $\widehat{\mathbf{q}}_{j+1} = \widehat{\mathbf{F}}(\widehat{\mathbf{q}}_{j}, \mathbf{u}_{j})$ |
| [**InterpolatedContinuousOpInfROM**](opinf.InterpolatedContinuousOpInfROM) | $\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t;\mu) = \widehat{\mathbf{F}}(t, \widehat{\mathbf{q}}(t;\mu), \mathbf{u}(t); \mu)$ |
| [**InterpolatedDiscreteOpInfROM**](opinf.InterpolatedDiscreteOpInfROM) | $\widehat{\mathbf{q}}_{j+1}(\mu) = \widehat{\mathbf{F}}(\widehat{\mathbf{q}}_{j}(\mu), \mathbf{u}_{j}; \mu)$ |

<!-- | `SteadyOpInfROM` | $\widehat{\mathbf{g}} = \widehat{\mathbf{F}}(\widehat{\mathbf{q}})$ |
| `AffineContinuousOpInfROM` | $\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t;\mu) = \widehat{\mathbf{F}}(t, \widehat{\mathbf{q}}(t;\mu), \mathbf{u}(t); \mu)$ |
| `AffineDiscreteOpInfROM` | $\widehat{\mathbf{q}}_{j+1}(\mu) = \widehat{\mathbf{F}}(\widehat{\mathbf{q}}_{j}(\mu), \mathbf{u}_{j}; \mu)$ | -->

Here $\widehat{\mathbf{q}} \in \mathbb{R}^{n}$ is the reduced-order state, $\mathbf{u} \in \mathbb{R}^{m}$ is the input, and $\mu\in\mathbb{R}^{p}$ is an external parameter (e.g., PDE coefficients).
Our goal is to learn an appropriate representation of $\widehat{\mathbf{F}}$ from data.

In the following discussion we begin with the non-parametric ROM classes; parametric classes are considered in [Parametric ROMs](subsec-parametric-roms).


(subsec-romclass-constructor)=
## Defining Model Structure

ROM classes are instantiated with a single argument, `modelform`, which is a string denoting the structure of the right-hand side function $\widehat{\mathbf{F}}$.
Each character in the string corresponds to a single term in the model.

| Character | Name | Continuous Term | Discrete Term |
| :-------- | :--- | :-------------- | :------------ |
| `c` | Constant | $\widehat{\mathbf{c}}$ | $\widehat{\mathbf{c}}$ |
| `A` | Linear | $\widehat{\mathbf{A}}\widehat{\mathbf{q}}(t)$ | $\widehat{\mathbf{A}}\widehat{\mathbf{q}}_{j}$ |
| `H` | Quadratic | $\widehat{\mathbf{H}}[\widehat{\mathbf{q}}(t) \otimes \widehat{\mathbf{q}}(t)]$ | $\widehat{\mathbf{H}}[\widehat{\mathbf{q}}_{j} \otimes \widehat{\mathbf{q}}_{j}]$ |
| `G` | Cubic | $\widehat{\mathbf{G}}[\widehat{\mathbf{q}}(t) \otimes \widehat{\mathbf{q}}(t) \otimes \widehat{\mathbf{q}}(t)]$ | $\widehat{\mathbf{G}}[\widehat{\mathbf{q}}_{j} \otimes \widehat{\mathbf{q}}_{j} \otimes \widehat{\mathbf{q}}_{j}]$ |
| `B` | Input | $\widehat{\mathbf{B}}\mathbf{u}(t)$ | $\widehat{\mathbf{B}}\mathbf{u}_{j}$ |


<!-- | `C` | Output | $\mathbf{y}(t)=\widehat{C}\widehat{\mathbf{q}}(t)$ | $\mathbf{y}_{k}=\hat{C}\widehat{\mathbf{q}}_{k}$ | -->

The full model form is specified as a single string.

| `modelform` | Continuous ROM Structure | Discrete ROM Structure |
| :---------- | :----------------------- | ---------------------- |
|  `"A"`      | $\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t) = \widehat{\mathbf{A}}\widehat{\mathbf{q}}(t)$ | $\widehat{\mathbf{q}}_{j+1} = \widehat{\mathbf{A}}\widehat{\mathbf{q}}_{j}$ |
|  `"cA"`     | $\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t) = \widehat{\mathbf{c}} + \widehat{\mathbf{A}}\widehat{\mathbf{q}}(t)$ | $\widehat{\mathbf{q}}_{j+1} = \widehat{\mathbf{c}} + \widehat{\mathbf{A}}\widehat{\mathbf{q}}_{j}$ |
|  `"AB"`   | $\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t) = \widehat{\mathbf{A}}\widehat{\mathbf{q}}(t) + \widehat{\mathbf{B}}\mathbf{u}(t)$ | $\widehat{\mathbf{q}}_{j+1} = \widehat{\mathbf{A}}\widehat{\mathbf{q}}_{j} + \widehat{\mathbf{B}}\mathbf{u}_{j}$ |
|  `"HB"`     | $\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t) = \widehat{\mathbf{H}}[\widehat{\mathbf{q}}(t)\otimes\widehat{\mathbf{q}}(t)] + \widehat{\mathbf{B}}\mathbf{u}(t)$ | $\widehat{\mathbf{q}}_{j+1} = \widehat{\mathbf{H}}[\widehat{\mathbf{q}}_{j}\otimes\widehat{\mathbf{q}}_{j}] + \widehat{\mathbf{B}}\mathbf{u}_{j}$ |

<!-- | Steady ROM Structure |
| $\widehat{\mathbf{g}} = \widehat{\mathbf{A}}\widehat{\mathbf{q}}$ |
| $\widehat{\mathbf{g}} = \widehat{\mathbf{c}} + \widehat{\mathbf{A}}\widehat{\mathbf{q}}$ |
| $\widehat{\mathbf{g}} = \widehat{\mathbf{A}}\widehat{\mathbf{q}} + \widehat{\mathbf{B}}\mathbf{u}$ |
| $\widehat{\mathbf{g}} = \widehat{\mathbf{H}}[\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}] + \widehat{\mathbf{B}}\mathbf{u}$ | -->


## ROM Attributes

All ROM classes have the following attributes.

### Dimensions

These attributes are integers that are initially set to `None`, then inferred from the training inputs during `fit()`.
They cannot be altered manually after calling `fit()`.

| Attribute | Description |
| :-------- | :---------- |
| `n` | Dimension of the high-dimensional training data $\mathbf{q}$. |
| `r` | Dimension of the reduced-order model state $\widehat{\mathbf{q}}$. |
| `m` | Dimension of the input $\mathbf{u}$. |

If there is no input (meaning `modelform` does not contain `'B'`), then `m` is set to 0.

### Basis

The `basis` attribute is the mapping between the $n$-dimensional state space of the full-order data and the smaller $r$-dimensional state space of the reduced-order model (e.g., POD basis).
This is the first input to the `fit()` method.
See the [Dimensionality Reduction](sec-guide-dimensionality) guide for details.

### Operators

These attributes are the operators corresponding to the learned parts of the reduced-order model.
The classes are defined in the [**operators**](opinf.operators) submodule.

<!-- TODO: Operator Class with links to API docs -->

| Attribute | Evaluation mapping | Jacobian mapping |
| :-------- | :----------------- | :--------------- |
| `c_` | $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{c}}$ | $\widehat{\mathbf{q}} \mapsto \mathbf{0}$ |
| `A_` | $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{A}}\widehat{\mathbf{q}}$ | $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{A}}$ |
| `H_` | $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{H}}[\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}]$ | $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{H}}[(\mathbf{I}\otimes\widehat{\mathbf{q}}) + (\widehat{\mathbf{q}}\otimes\mathbf{I})]$ |
| `G_` | $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{G}}[\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}]$ | $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{G}}[(\mathbf{I}\otimes\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}) + \cdots + (\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}\otimes\mathbf{I})]$ |
| `B_` | $\mathbf{u} \mapsto \widehat{\mathbf{B}}\mathbf{u}$ | $\mathbf{u} \mapsto \widehat{\mathbf{B}}$ |

All operators are set to `None` initially and only changed by `fit()` if the operator is included in the prescribed `modelform` (e.g., if `modelform="AHG"`, then `c_` and `B_` are always `None`).
<!-- Note that Jacobian mapping of the input operation _with respect to the state_ is zero. -->

#### Operator Attributes

The discrete representation of the operator is a NumPy array stored as the `entries` attribute.
This array can also be accessed by slicing the operator object directly.

```python
>>> import numpy as np
>>> import opinf

>>> arr = np.arange(16).reshape(4, 4)
>>> operator = opinf.operators.LinearOperator(arr)

>>> operator.entries
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])

>>> operator[:]
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])

>>> operator.shape
(4, 4)
```

In practice, with a ROM object `rom`, the entries of (e.g.) the linear state matrix $\widehat{\mathbf{A}}$ are accessed with `rom.A_[:]` or `rom.A_.entries`.

#### Operator Methods

The `evaluate()` method computes the action of the operator on the (low-dimensional) state or input.

```python
>>> q_ = np.arange(4)
>>> operator.evaluate(q_)
array([14, 38, 62, 86])

# Equivalent calculation with the raw NumPy array.
>>> arr @ q_
array([14, 38, 62, 86])
```

::::{note}
Nothing special is happening under the hood for constant and linear operators, but the quadratic and cubic operators use a compressed representation to efficiently compute the operator action on the quadratic or cubic Kronecker products $\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}$ or $\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}$.

```python
>>> r = 5
>>> arr2 = np.random.random((r, r**2))
>>> quadratic_operator = opinf.operators.QuadraticOperator(arr2)
>>> q_ = np.random.random(r)

>>> np.allclose(quadratic_operator.evaluate(q_), arr2 @ (np.kron(q_, q_)))
True

>>> quadratic_operator.shape
(5, 15)
```

The shape of the quadratic operator `entries` has been reduced from $r \times r^{2}$ to $r \times \frac{r(r + 1)}{2}$ to exploit the structure of the Kronecker products.

:::{dropdown} Details
Let $\widehat{\mathbf{q}} = [~\hat{q}_{1}~\cdots~\hat{q}_{r}~]^{\mathsf{T}}\in\mathbb{R}^{r}$ and consider the Kronecker product

$$
\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}
= \left[\begin{array}{c}
    \hat{q}_{1}\widehat{\mathbf{q}} \\
    \hat{q}_{2}\widehat{\mathbf{q}} \\
    \vdots \\
    \hat{q}_{r}\widehat{\mathbf{q}} \\
\end{array}\right]
= \left[\begin{array}{c}
    \hat{q}_{1}^{2} \\
    \hat{q}_{1}\hat{q}_{2} \\
    \vdots \\
    \hat{q}_{1}\hat{q}_{r} \\
    \hat{q}_{1}\hat{q}_{2} \\
    \hat{q}_{2}^{2} \\
    \vdots \\
    \hat{q}_{2}\hat{q}_{r} \\
    \vdots \\
    \hat{q}_{r}^{2}
\end{array}\right]
\in \mathbb{R}^{r^{2}}.
$$

Note that $\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}$ has some redundant entries, for example $\hat{q}_{1}\hat{q}_{2}$ shows up twice. In fact, $\hat{q}_{i}\hat{q}_{j}$ occurs twice for every choice of $i \neq j$.
Thus, $\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}$ can be represented with only $r (r + 1)/2$ degrees of freedom as, for instance,

$$
\left[\begin{array}{c}
    \widehat{\mathbf{q}}^{(1)} \\
    \widehat{\mathbf{q}}^{(2)} \\
    \vdots \\
    \widehat{\mathbf{q}}^{(r)}
\end{array}\right]
= \left[\begin{array}{c}
    \hat{q}_{1}^{2} \\
    \hat{q}_{1}\hat{q}_{2} \\
    \hat{q}_{2}^{2} \\
    \hat{q}_{1}\hat{q}_{3} \\
    \hat{q}_{2}\hat{q}_{3} \\
    \hat{q}_{3}^{2} \\
    \vdots \\
    \hat{q}_{r}^{2}
\end{array}\right]
\in \mathbb{R}^{r(r + 1)/2},
\qquad
\widehat{\mathbf{q}}^{(i)}
= \hat{q}_{i}\left[\begin{array}{c}
    \hat{q}_{1} \\ \vdots \\ \hat{q}_{i}
\end{array}\right]\in\mathbb{R}^{i}.
$$

This is the same as filling a vector with the upper-triangular entries of the outer product $\widehat{\mathbf{q}}\widehat{\mathbf{q}}^{\mathsf{T}}$.
The dimension $r (r + 1)/2$ arises because we choose 2 of r entries _without replacement_, i.e., this is a [multiset coefficient](https://en.wikipedia.org/wiki/Multiset#Counting_multisets):

$$
\left(\!\!{r\choose 2}\!\!\right)
= \binom{r + 2 - 1}{2}
= \binom{r+1}{2}
= \frac{r(r+1)}{2}.
$$

:::
::::

<!-- TODO: Jacobians -->

### Summary

| Attribute | Description |
| :-------- | :---------- |
| `n` | Dimension of the high-dimensional training data $\mathbf{q}$. |
| `r` | Dimension of the reduced-order model state $\widehat{\mathbf{q}}$. |
| `m` | Dimension of the input $\mathbf{u}$. |
| `basis` | Mapping between the $n$-dimensional state space of the full-order data and the $r$-dimensional state space of the ROM |
| `c_` | Constant operator $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{c}}$ |
| `A_` | Linear operator $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{A}}\widehat{\mathbf{q}}$ |
| `H_` | Quadratic operator $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{H}}[\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}]$ |
| `G_` | Cubic operator $\widehat{\mathbf{q}} \mapsto \widehat{\mathbf{G}}[\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}]$ |
| `B_` | Input operator $\mathbf{u} \mapsto \widehat{\mathbf{B}}\mathbf{u}$ |


## ROM Methods

All ROM classes have the following methods.

### Dimensionality Reduction

The `compress()` method maps a state quantity from the high-dimensional space $\mathbb{R}^{n}$ to the low-dimensional space $\mathbb{R}^{r}$.
Conversely, `decompress()` maps from $\mathbb{R}^{r}$ to $\mathbb{R}^{n}$.
<!-- These methods are not quite inverses: the results of `decompress()` are restricted to the portion of $\mathbb{R}^{n}$ that can be represented through the underlying basis. -->
These methods wrap the `compress()` and `decompress()` methods of the `basis` attribute.

### Training

::::{margin}
:::{tip}
The `fit()` method accepts `basis=None`, in which case the state arguments for training are assumed to be already reduced to an $r$-dimensional state space (e.g., $\widehat{\mathbf{Q}} = \mathbf{V}_{r}^{\top}\mathbf{Q}$ instead of $\mathbf{Q}$).
:::
::::

The `fit()` method sets up and solves a [least-squares regression](subsec-opinf-regression) to determine the entries of the operators $\widehat{\mathbf{c}}$, $\widehat{\mathbf{A}}$, $\widehat{\mathbf{H}}$, $\widehat{\mathbf{G}}$, and/or $\widehat{\mathbf{B}}$.
Common inputs are
- the basis
- state snapshot data
- left-hand side data (time derivatives)
- regularization parameters
<!-- TODO: least squares solver! -->

### Prediction

The `evaluate()` method evaluates the right-hand side of the learned reduced-order model, i.e., it is the mapping

<!-- :::{tip}
The `evaluate()` and `jacobian()` methods are useful for constructing custom solvers for the reduced-order model.
::: -->

$$
(\widehat{\mathbf{q}},\mathbf{u}) \mapsto
\widehat{\mathbf{c}}
+ \widehat{\mathbf{A}}\widehat{\mathbf{q}}
+ \widehat{\mathbf{H}}[\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}]
+ \widehat{\mathbf{B}}\mathbf{u}.
$$

The `predict()` method solves the reduced-order model for given initial conditions and inputs.

### Model Persistence

Some ROM objects can be saved in [HDF5 format](http://docs.h5py.org/en/stable/index.html) with the `save()` method, then recovered later with the `load()` class method.
Such files store metadata for the model class and structure, the reduced-order model operators, and (optionally) the basis.

```python
>>> import opinf

# Assume we have a ROM as an opinf.ContinuousOpInfROM object, called `rom`.

>>> rom.save("trained_rom.h5")                                # Save a trained model.
>>> rom2 = opinf.ContinuousOpInfROM.load("trained_rom.h5")    # Load a model from file.
>>> rom == rom2
True
```

For ROM classes without a `save()`/`load()` implementation, ROM objects can usually be saved locally via the `pickle` or `joblib` libraries, which is [the approach taken by scikit-learn (`sklearn`)](https://scikit-learn.org/stable/model_persistence.html).

:::{tip}
Files in HDF5 format are slightly more transparent than pickled binaries in the sense that individual parts of the file can be extracted manually without loading the entire file.
Several programming languages support HDF5 format (MATLAB, C, C++, etc.), making HDF5 a good candidate for sharing ROM data with other programs.
:::

### Summary

| Method | Description |
| :----- | :---------- |
| `compress()` | Map high-dimensional states to their low-dimensional coordinates |
| `decompress()` | Use low-dimensional coordinates to construct a high-dimensional state |
| `fit()` | Use training data to infer the operators defining the ROM |
| `evaluate()` | Evaluate the reduced-order model for a given state / input |
| `predict()` | Solve the reduced-order model |
| `save()` | Save the ROM data to an HDF5 file |
| `load()` | Load a ROM from an HDF5 file |


(sec-continuous)=
## Continuous-time ROMs

A continuous-time ROM is a surrogate for a system of ordinary differential equations, written generally as

$$
\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t;\mu)
= \widehat{\mathbf{F}}(t, \widehat{\mathbf{q}}(t;\mu), \mathbf{u}(t); \mu).
$$

The following ROM classes target the continuous-time setting.
- [**ContinuousOpInfROM**](opinf.ContinuousOpInfROM) (nonparametric)
- [**InterpolatedContinuousOpInfROM**](opinf.InterpolatedContinuousOpInfROM) (parametric via interpolation)

### Time Derivative Data

The OpInf regression problem for the continuous-time setting is {eq}`eq:opinf-lstsq-residual`:

$$
\min_{\widehat{\mathbf{c}},\widehat{\mathbf{A}},\widehat{\mathbf{H}},\widehat{\mathbf{B}}}\sum_{j=0}^{k-1}\left\|
    \widehat{\mathbf{c}}
    + \widehat{\mathbf{A}}\widehat{\mathbf{q}}_{j}
    + \widehat{\mathbf{H}}[\widehat{\mathbf{q}}_{j} \otimes \widehat{\mathbf{q}}_{j}]
    + \widehat{\mathbf{B}}\mathbf{u}_{j}
    - \dot{\widehat{\mathbf{q}}}_{j}
\right\|_{2}^{2}
+ \mathcal{R}(\widehat{\mathbf{c}},\widehat{\mathbf{A}},\widehat{\mathbf{H}},\widehat{\mathbf{B}}),
$$

where
- $\widehat{\mathbf{q}}_{j} := \mathbf{V}_{r}^{\mathsf{T}}\mathbf{q}(t_{j})$ is the projected state at time $t_{j}$,
- $\dot{\widehat{\mathbf{q}}}_{j} := \frac{\textrm{d}}{\textrm{d}t}\mathbf{V}_{r}^{\mathsf{T}}\mathbf{q}\big|_{t=t_{j}}$ is the projected time derivative of the state at time $t_{j}$,
- $\mathbf{u}_{j} := \mathbf{u}(t_j)$ is the input at time $t_{j}$, and
- $\mathcal{R}$ is a _regularization term_ that penalizes the entries of the learned operators.

The state time derivatives $\dot{\mathbf{q}}_{j}$ are required in the regression.
These may be available from the full-order solver that generated the training data, but not all solvers provide such data.
One option is to use the states $\mathbf{q}_{j}$ to estimate the time derivatives via finite difference or spectral differentiation.
See `opinf.pre.ddt()` for details.

### ROM Evaluation

The `evaluate()` method of `ContinuousOpInfROM` is the mapping

$$
(\widehat{\mathbf{q}}(t), \mathbf{u}(\cdot))
\mapsto \frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t)
$$

as defined by the ROM.

```python
evaluate(self, t, state_, input_func=None)
```

| Argument | Type | Description |
| :------- | :--- | :---------- |
| `t` | `float` | Time corresponding to the state |
| `state_` | `(r,) ndarray` | Reduced state vector $\widehat{\mathbf{q}}(t)$ |
| `input_func` | `callable` | Mapping $t \mapsto \mathbf{u}(t)$ |


### Time Integration

The `predict()` method of `ContinuousOpInfROM` wraps [`scpiy.integrate.solve_ivp()`](https://docs.scipy.org/doc/scipy/reference/integrate.html) to solve the reduced-order model over a given time domain.

```python
predict(self, state0, t, input_func=None, decompress=True, **options)
```

| Argument | Type | Description |
| :------- | :--- | :---------- |
| `state0` | `(n,) or (r,) ndarray` | Initial state vector $\mathbf{q}(0)\in\mathbb{R}^{n}$ or $\widehat{\mathbf{q}}(0)\in\mathbb{R}^{r}$ |
| `t` | `(nt,) ndarray` | Time domain over which to integrate the ROM |
| `input_func` | `callable` | Mapping $t \mapsto \mathbf{u}(t)$ |
| `decompress` | `bool` | If True and the `basis` is not `None`, reconstruct the results in the $n$-dimensional state space |
| `**options` | | Additional arguments for `scipy.integrate.solve_ivp()` |

<!-- TODO: implement common solvers and document here. -->


(sec-discrete)=
## Discrete-time ROMs

The OpInf framework can be used to construct reduced-order models for approximating _discrete_ dynamical systems, as may arise from discretizing PDEs in both space and time.
A discrete-time ROM is a surrogate for a system of difference equations, written generally as

$$
\widehat{\mathbf{q}}_{j+1}(\mu)
= \widehat{\mathbf{F}}(\widehat{\mathbf{q}}_{j}(\mu), \mathbf{u}_{j}; \mu).
$$

The following ROM classes target the discrete setting.
- [**DiscreteOpInfROM**](opinf.DiscreteOpInfROM) (nonparametric)
- [**InterpolatedDiscreteOpInfROM**](opinf.InterpolatedDiscreteOpInfROM) (parametric via interpolation)

### Iterated Training Data

The OpInf regression problem for the discrete-time setting is a slight modification of the continuous-time OpInf regression {eq}`eq:opinf-lstsq-residual`:

$$
\min_{\widehat{\mathbf{c}},\widehat{\mathbf{A}},\widehat{\mathbf{H}},\widehat{\mathbf{B}}}\sum_{j=0}^{k-1}\left\|
    \widehat{\mathbf{c}}
    + \widehat{\mathbf{A}}\widehat{\mathbf{q}}_{j}
    + \widehat{\mathbf{H}}[\widehat{\mathbf{q}}_{j} \otimes \widehat{\mathbf{q}}_{j}]
    + \widehat{\mathbf{B}}\mathbf{u}_{j}
    - \widehat{\mathbf{q}}_{j+1}
\right\|_{2}^{2}
+ \mathcal{R}(\widehat{\mathbf{c}},\widehat{\mathbf{A}},\widehat{\mathbf{H}},\widehat{\mathbf{B}}),
$$

where
- $\widehat{\mathbf{q}}_{j} := \mathbf{V}_{r}^{\mathsf{T}}\mathbf{q}_{j}$ is the $j$th projected state,
- $\mathbf{u}_{j}$ is the input corresponding to the $j$th state, and
- $\mathcal{R}$ is a _regularization term_ that penalizes the entries of the learned operators.

### ROM Evaluation

The `evaluate()` method of `DiscreteOpInfROM` is the mapping

$$
(\widehat{\mathbf{q}}_{j}, \mathbf{u}_{j})
\mapsto \widehat{\mathbf{q}}_{j+1}
$$

```python
evaluate(self, state_, input_=None)
```

| Argument | Type | Description |
| :------- | :--- | :---------- |
| `state_` | `(r,) ndarray` | Reduced state vector $\widehat{\mathbf{q}}$ |
| `input_` | `(m,) ndarray` | Input vector $\mathbf{u}$ corresponding to the state |

### Solution Iteration

The `predict()` method of `DiscreteOpInfROM` iterates the system to solve the reduced-order model for a given number of steps.
Unlike the continuous-time case, there are no choices to make about what scheme to use to solve the problem: the solution iteration is explicitly described by the reduced-order model.

```python
predict(self, state0, niters, inputs=None, decompress=True)
```

| Argument | Type | Description |
| :------- | :--- | :---------- |
| `state0` | `(n,) or (r,) ndarray` | Initial state vector $\mathbf{q}_{0}\in\mathbb{R}^{n}$ or $\widehat{\mathbf{q}}_{0}\in\mathbb{R}^{r}$ |
| `niters` | `int` | Number of times to step the system forward |
| `inputs` | `(m, niters-1) ndarray` | Inputs $\mathbf{u}_{j}$ for the next `niters-1` time steps |
| `decompress` | `bool` | If True and the `basis` is not `None`, reconstruct the results in the $n$-dimensional state space |

<!-- TODO: implement common solvers and document here. -->


(subsec-parametric-roms)=
## Parametric ROMs

The `ContinuousOpInfROM` and `DiscreteOpInfROM` classes are _non-parametric_ ROMs.
A _parametric_ ROM is one that depends on one or more external parameters $\mu\in\mathbb{R}^{p}$, meaning the operators themselves may depend on the external parameters.
This is different from the ROM depending on external inputs $\mathbf{u}$ that are provided at prediction time; by "parametric ROM" we mean the _operators_ of the ROM depend on $\mu$.
For example, a linear time-continuous parametric ROM has the form

$$
\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t;\mu)
= \widehat{\mathbf{A}}(\mu)\widehat{\mathbf{q}}(t;\mu).
$$

### Additional Attributes

Parametric ROM classes have the following additional attributes.

| Attribute | Description |
| :-------- | :---------- |
| `p` | Dimension of the parameter $\mu$. |
| `s` | Number of training parameter samples. |

### Parametric Operators

The operators of a parametric ROM are themselves parametric, meaning they depend on the parameter $\mu$.
Therefore, the operator attributes `c_`, `A_`, `H_`, `G_`, and/or `B_` of a parametric ROM must first be evaluated at a parameter value before they can be applied to a reduced state or input.
This is done by calling the object with the parameter value as input.

:::{mermaid}
%%{init: {'theme': 'forest'}}%%
flowchart LR
    A[Parametric operator] -->|call_object| B[Non-parametric operator]
:::

```python
>>> import numpy as np
>>> import scipy.interpolate
>>> import opinf

>>> parameters = np.linspace(0, 1, 4)
>>> entries = np.random.random((4, 3))

# Construct a parametric constant operator c(µ).
>>> c_ = opinf.operators.InterpolatedConstantOperator(
...     parameters, entries, scipy.interpolate.CubicSpline
... )
>>> type(c_)

# Evaluate the parametric constant operator at a given parameter.
>>> c_static_ = c_(.5)
>>> type(c_static_)
opinf.operators._nonparametric.ConstantOperator

>>> c_static_.evaluate()
array([0.89308692, 0.81232528, 0.52454941])
```

Parametric operator evaluation is taken care of under the hood during parametric ROM evaluation.

### Parametric ROM Evaluation

A parametric ROM object maps a parameter value to a non-parametric ROMs.
Like parametric operators, this is does by calling the object.

:::{mermaid}
%%{init: {'theme': 'forest'}}%%
flowchart LR
    A[Parametric ROM] -->|call_object| B[Non-parametric ROM]
:::

The `evaluate()` and `predict()` methods of parametric ROMs are like their counterparts in the nonparametric ROM classes, but with an additional `parameter` argument that comes before other arguments.
These are convenience methods that evaluate the ROM at the given parameter, then evaluate the resulting non-parametric ROM.
For example, `parametric_rom.evaluate(parameter, state_)` and `parametric_rom(parameter).evaluate(state_)` are equivalent.

## Interpolatory ROMs

Consider the problem of learning a parametric reduced-order model of the form

$$
\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t;\mu)
= \widehat{\mathbf{A}}(\mu)\widehat{\mathbf{q}}(t;\mu) + \widehat{\mathbf{B}}(\mu)\mathbf{u}(t),
$$

where
- $\widehat{\mathbf{q}}(t;\mu)\in\mathbb{R}^{r}$ is the ROM state,
- $\mathbf{u}(t)\in\mathbb{R}^{m}$ is an independent input, and
- $\mu \in \mathbb{R}^{p}$ is a free parameter.

We assume to have state/input training data for $s$ parameter samples $\mu_{1},\ldots,\mu_{s}$.

### Training Strategy

One way to deal with the parametric dependence of $\widehat{\mathbf{A}}$ and $\widehat{\mathbf{B}}$ on $\mu$ is to independently learn a reduced-order model for each parameter sample, then interpolate the learned models in order to make predictions for a new parameter sample.
This approach is implemented by the following ROM classes.
- `InterpolatedContinuousOpInfROM`
- `InterpolatedDiscreteOpInfROM`

The OpInf learning problem is the following:

$$
\min_{\widehat{\mathbf{A}}^{(i)},\widehat{\mathbf{B}}^{(i)}}\sum_{j=0}^{k-1}\left\|
    \widehat{\mathbf{A}}^{(i)}\widehat{\mathbf{q}}_{ij} + \widehat{\mathbf{B}}^{(i)}\mathbf{u}_{ij} - \dot{\widehat{\mathbf{q}}}_{ij}
\right\|_{2}^{2}
+ \mathcal{R}^{(i)}(\widehat{\mathbf{A}}^{(i)},\widehat{\mathbf{B}}^{(i)}),
$$

where
- $\widehat{\mathbf{q}}_{ij} := \mathbf{V}_{r}^{\mathsf{T}}\mathbf{q}(t_{j};\mu_{i})$ is the projected state,
- $\dot{\widehat{\mathbf{q}}}_{j} := \frac{\textrm{d}}{\textrm{d}t}\mathbf{V}_{r}^{\mathsf{T}}\mathbf{q}(t;\mu_{i})\big|_{t=t_{j}}$ is the projected time derivative of the state,
- $\mathbf{u}_{ij} := \mathbf{u}(t_j)$ is the input corresponding to the state $\mathbf{q}_{ij}$, and
- $\mathcal{R}^{(i)}$ is a _regularization term_ that penalizes the entries of the learned operators.

Once $\widehat{\mathbf{A}}^{(1)},\ldots,\widehat{\mathbf{A}}^{(s)}$ and $\widehat{\mathbf{B}}^{(1)},\ldots,\widehat{\mathbf{B}}^{(s)}$ are chosen, $\widehat{\mathbf{A}}(\mu)$ and $\widehat{\mathbf{B}}(\mu)$ are defined by interpolation, i.e.,

$$
\widehat{\mathbf{A}}(\mu) = \text{interpolate}(\widehat{\mathbf{A}}^{(1)},\ldots,\widehat{\mathbf{A}}^{(s)}; \mu).
$$

### Choose an Interpolator

In addition to the `modelform`, the constructor of interpolatory ROM classes takes an additional argument, `InterpolatorClass`, which handles the actual interpolation.
This class must obey the following API requirements:
- Initialized with `interpolator = InterpolatorClass(parameters, values)` where
    - `parameters` is a list of $s$ parameter values (all of the same shape)
    - `values` is a list of $s$ vectors/matrices
- Evaluated by calling the object with `interpolator(parameter)`, resulting in a vector/matrix of the same shape as `values[0]`.

Many of the classes in [`scipy.interpolate`](https://docs.scipy.org/doc/scipy/reference/interpolate.html) match this style.

:::{tip}
There are a few convenience options for the `InterpolatorClass` arguments.
- `"cubicspline"` sets `InterpolatorClass` to `scipy.interpolate.CubicSpline`. This interpolator requires a parameter dimension of $p = 1$.
- `"linear"`: sets `InterpolatorClass` to `scipy.interpolate.LinearNDInterpolator`. This interpolator requires a parameter dimension of $p > 1$.
- `"auto"`: choose between `scipy.interpolate.CubicSpline` and `scipy.interpolate.LinearNDInterpolator` based on the parameter dimension $p$.
:::

:::{note}
After the reduced-order model has been constructed through `fit()`, the interpolator can modified through the `set_interpolator()` method.
:::

### Training Data Organization

Interpolated ROM `fit()` methods accept the training data in the  following formats.
- The basis
- A list of training parameters $[\mu_{1},\ldots,\mu_{s}]$ for which we have data
- A list of states $[\mathbf{Q}(\mu_{1}),\ldots,\mathbf{Q}(\mu_{s})]$ corresponding to the training parameters
- A single regularization parameter or a list of $s$ regularization parameters

### ROM Evaluation

As with all parametric ROM classes, evaluation the ROM by calling the object on the specifies parameter, e.g., `rom(parameter).predict(...)`.
