(sec-romclasses)=
# ROM Classes

There are several reduced-order model (ROM) classes defined in the main namespace of `rom_operator_inference`.
Each class corresponds to a specific problem setting.

::::{margin}
:::{tip}
The API for these classes adopts some principles from the [scikit-learn](https://scikit-learn.org/stable/index.html) [API](https://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects): there are `fit()` and `predict()` methods, hyperparameters are set in the constructor, estimated attributes end with underscore, and so on.
:::
::::

| Class Name | Problem Statement |
| :--------- | :---------------: |
| `ContinuousOpInfROM` | $\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t) = \widehat{\mathbf{F}}(t, \widehat{\mathbf{q}}(t), \mathbf{u}(t))$ |
| `DiscreteOpInfROM` | $\widehat{\mathbf{q}}_{j+1} = \widehat{\mathbf{F}}(\widehat{\mathbf{q}}_{j}, \mathbf{u}_{j})$ |
| `InterpolatedContinuousOpInfROM` | $\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t;\mu) = \widehat{\mathbf{F}}(t, \widehat{\mathbf{q}}(t;\mu), \mathbf{u}(t); \mu)$ |
| `InterpolatedDiscreteOpInfROM` | $\widehat{\mathbf{q}}_{j+1}(\mu) = \widehat{\mathbf{F}}(\widehat{\mathbf{q}}_{j}(\mu), \mathbf{u}_{j}; \mu)$ |

<!-- | `SteadyOpInfROM` | $\widehat{\mathbf{g}} = \widehat{\mathbf{F}}(\widehat{\mathbf{q}})$ |
| `AffineContinuousOpInfROM` | $\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t;\mu) = \widehat{\mathbf{F}}(t, \widehat{\mathbf{q}}(t;\mu), \mathbf{u}(t); \mu)$ |
| `AffineDiscreteOpInfROM` | $\widehat{\mathbf{q}}_{j+1}(\mu) = \widehat{\mathbf{F}}(\widehat{\mathbf{q}}_{j}(\mu), \mathbf{u}_{j}; \mu)$ | -->

Here $\widehat{\mathbf{q}} \in \mathbb{R}^{n}$ is the reduced-order state, $\mathbf{u} \in \mathbb{R}^{m}$ is the input, and $\mu\in\mathbb{R}^{p}$ is an external parameter (e.g., PDE coefficients).
Our goal is to learn an appropriate representation of $\widehat{\mathbf{F}}$ from data.

In the following discussion we begin with the non-parametric ROM classes `ContinuousOpInfROM` and `DiscreteOpInfROM`; parametric classes are considered in [Parametric ROMs](subsec-parametric-roms).

(subsec-romclass-constructor)=
## Defining Model Structure

All ROM classes are instantiated with a single argument, `modelform`, which is a string denoting the structure of the right-hand side function $\widehat{\mathbf{F}}$.
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

---

## ROM Attributes

ROM classes have the following attributes.

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
See [Basis Computation](sec-basis-computation) for details.

### Operators

These attributes are the operators corresponding to the learned parts of the reduced-order model.
The classes are defined in `opinf.core.operators`.

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
>>> import rom_operator_inference as opinf

>>> arr = np.arange(16).reshape(4, 4)
>>> operator = opinf.core.operators.LinearOperator(arr)

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
>>> quadratic_operator = opinf.core.operators.QuadraticOperator(arr2)
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

---

## ROM Methods

All ROM classes have the following methods.

### Encoding and Decoding

The `encode()` method maps a state quantity from the high-dimensional space $\mathbb{R}^{n}$ to the low-dimensional space $\mathbb{R}^{r}$.
Conversely, `decode()` maps from $\mathbb{R}^{r}$ to $\mathbb{R}^{n}$.
<!-- These methods are not quite inverses: the results of `decode()` are restricted to the portion of $\mathbb{R}^{n}$ that can be represented through the underlying basis. -->
These methods wrap the `encode()` and `decode()` methods of the `basis` attribute; see [Preprocessing](sec-preprocessing) and [Basis Computation](sec-basis-computation) for more details.

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
>>> import rom_operator_inference as opinf

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
| `project()` | Map high-dimensional states to their low-dimensional coordinates |
| `reconstruct()` | Use low-dimensional coordinates to construct a high-dimensional state |
| `fit()` | Use training data to infer the operators defining the ROM |
| `evaluate()` | Evaluate the reduced-order model for a given state / input |
| `predict()` | Solve the reduced-order model |
| `save()` | Save the ROM data to an HDF5 file |
| `load()` | Load a ROM from an HDF5 file |

(subsec-parametric-roms)=
## Parametric ROMs

The `ContinuousOpInfROM` and `DiscreteOpInfROM` classes detailed above are _non-parametric_ ROMs.
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
>>> import rom_operator_inference as opinf

>>> parameters = np.linspace(0, 1, 4)
>>> entries = np.random.random((4, 3))

# Construct a parametric constant operator c(µ).
>>> c_ = opinf.core.operators.InterpolatedConstantOperator(
...     parameters, entries, scipy.interpolate.CubicSpline
... )
>>> type(c_)

# Evaluate the parametric constant operator at a given parameter.
>>> c_static_ = c_(.5)
>>> type(c_static_)
rom_operator_inference.core.operators._nonparametric.ConstantOperator

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
