(sec-romclasses)=
# ROM Classes

The core of `rom_operator_inference` is highly object oriented and defines several `ROM` classes that serve as the workhorse of the package.
The API for these classes adopts some principles from the [scikit-learn](https://scikit-learn.org/stable/index.html) [API](https://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects): there are `fit()` and `predict()` methods, hyperparameters are set in the constructor, estimated attributes end with underscore, and so on.

Each class corresponds to a specific reduced-order model setting and ROM construction method.

| Class Name | Problem Statement |
| :--------- | :---------------: |
| `ContinuousOpInfROM` | $\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t) = \widehat{\mathbf{f}}(t, \widehat{\mathbf{q}}(t), \mathbf{u}(t))$ |
| `DiscreteOpInfROM` | $\widehat{\mathbf{q}}_{j+1} = \widehat{\mathbf{f}}(\widehat{\mathbf{q}}_{j}, \mathbf{u}_{j})$ |

<!-- | `SteadyOpInfROM` | $\widehat{\mathbf{g}} = \widehat{\mathbf{f}}(\widehat{\mathbf{q}})$ |
| `AffineContinuousOpInfROM` | $\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t;\mu) = \widehat{\mathbf{f}}(t, \widehat{\mathbf{q}}(t;\mu), \mathbf{u}(t); \mu)$ |
| `AffineDiscreteOpInfROM` | $\widehat{\mathbf{q}}_{j+1}(\mu) = \widehat{\mathbf{f}}(\widehat{\mathbf{q}}_{j}(\mu), \mathbf{u}_{j}; \mu)$ |
| `InterpolatedContinuousOpInfROM` | $\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t;\mu) = \widehat{\mathbf{f}}(t, \widehat{\mathbf{q}}(t;\mu), \mathbf{u}(t); \mu)$ |
| `InterpolatedDiscreteOpInfROM` | $\widehat{\mathbf{q}}_{j+1}(\mu) = \widehat{\mathbf{f}}(\widehat{\mathbf{q}}_{j}(\mu), \mathbf{u}_{j}; \mu)$ | -->

Here $\widehat{\mathbf{q}} \in \mathbb{R}^{n}$ is the reduced-order state and $\mathbf{u} \in \mathbb{R}^{m}$ is the (state-independent) input.
<!-- , and $\mu\in\mathbb{R}^{d_\mu}$ represents external parameters. -->


## Constructor: Define Model Structure

All `ROM` classes are instantiated with a single argument, `modelform`, which is a string denoting the structure of the right-hand side of the reduced-order model, $\widehat{\mathbf{f}}$.
Each character in the string corresponds to a single term of the operator, given in the following table.

| Character | Name | Continuous Term | Discrete Term |
| :-------- | :--- | :-------------- | :------------ |
| `c` | Constant | $\widehat{\mathbf{c}}$ | $\widehat{\mathbf{c}}$ |
| `A` | Linear | $\widehat{\mathbf{A}}\widehat{\mathbf{q}}(t)$ | $\widehat{\mathbf{A}}\widehat{\mathbf{q}}_{j}$ |
| `H` | Quadratic | $\widehat{\mathbf{H}}[\widehat{\mathbf{q}}(t) \otimes \widehat{\mathbf{q}}(t)]$ | $\widehat{\mathbf{H}}[\widehat{\mathbf{q}}_{j} \otimes \widehat{\mathbf{q}}_{j}]$ |
| `G` | Cubic | $\widehat{\mathbf{G}}[\widehat{\mathbf{q}}(t) \otimes \widehat{\mathbf{q}}(t) \otimes \widehat{\mathbf{q}}(t)]$ | $\widehat{\mathbf{G}}[\widehat{\mathbf{q}}_{j} \otimes \widehat{\mathbf{q}}_{j} \otimes \widehat{\mathbf{q}}_{j}]$ |
| `B` | Input | $\widehat{\mathbf{B}}\mathbf{u}(t)$ | $\widehat{\mathbf{B}}\mathbf{u}_{j}$ |


<!-- | `C` | Output | $\mathbf{y}(t)=\widehat{C}\widehat{\mathbf{q}}(t)$ | $\mathbf{y}_{k}=\hat{C}\widehat{\mathbf{q}}_{k}$ | -->

These are all input as a single string.
Examples:

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

## Attributes

All `ROM` classes have the following attributes.

### Dimensions

These scalars are set in `fit()` and are inferred from the training inputs.
They cannot be altered by hand post-training.

| Attribute | Description |
| :-------- | :---------- |
| `n` | Dimension of the original, high-dimensional model. |
| `r` | Dimension of the learned reduced-order model. |
| `m` | Dimension of the input **u**, or `None` if `'B'` is not in `modelform`. |

### Basis Matrix

The `basis` attribute is the $n \times r$ basis defining the mapping between the $n$-dimensional space of the full-order model and the reduced $r$-dimensional subspace of the reduced-order model (e.g., POD basis).
This is the first input to the `fit()` method.

:::{tip}
To save memory, ROM classes allow entering `basis=None` in the `fit()` method, which then assumes that state arguments for training are already projected to the $r$-dimensional subspace (e.g., $\widehat{\mathbf{Q}} = \mathbf{V}_{r}^{\top}\mathbf{Q}$ instead of $\mathbf{Q}$).
:::

### Operators

These are the operators corresponding to the learned parts of the reduced-order model.

| Modelform Character | Attribute Name |
| :------------------ | :------------- |
| `"c"` | `c_` |
| `"A"` | `A_` |
| `"H"` | `H_` |
| `"G"` | `G_` |
| `"B"` | `B_` |

All operators are set to `None` initially and only changed by `fit()` if the operator is included in the prescribed `modelform` (e.g., if `modelform="AHG"`, then `c_` and `B_` are always `None`)

---

## Methods

All `ROM` classes have the following methods.
### Training

### Prediction

Reduced model function `evaluate()`, learned in `fit()`: the ROM function, defined by the reduced operators listed above.
For continuous models, `evaluate` has the following signature:
```python
def evaluate(t, q_, input_func):
    """ROM function for continuous models.

    Parameters
    ----------
    t : float
        Time, a scalar.
    q_ : (r,) ndarray
        Reduced state vector.
    input_func : func(float) -> (m,)
        Input function that maps time `t` to an input vector of length m.
    """
```
For discrete models, the signature is the following.
```python
def evaluate(q_, u):
    """ROM function for discrete models.

    Parameters
    ----------
    q_ : (r,) ndarray
        Reduced state vector.
    u : (m,) ndarray
        Input vector of length m corresponding to the state.
    """
```
The input argument `u` is only used if `B` is in `modelform`.

### Model Persistence

Trained ROM objects can be saved in [HDF5 format](http://docs.h5py.org/en/stable/index.html) with the `save()` method, and recovered later with the `load()` class method.
Such files store metadata for the model class and structure, the reduced-order model operators (`c_`, `A_`, etc.), other attributes learned in `fit()`, and (optionally) the basis `Vr`.

TODO: docstring.

```python
>>> import rom_operator_inference as opinf

# Assume we have a ROM as an opinf.ContinuousOpInfROM object, called `rom`.

>>> rom.save("trained_rom.h5")                                # Save a trained model.
>>> rom2 = opinf.ContinuousOpInfROM.load("trained_rom.h5")    # Load a model from file.
>>> rom == rom2
True
```
