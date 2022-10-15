# ROM Methods

All ROM classes have the following methods.

## Encoding and Decoding

The `encode()` method maps a state quantity from the high-dimensional space $\mathbb{R}^{n}$ to the low-dimensional space $\mathbb{R}^{r}$.
Conversely, `decode()` maps from $\mathbb{R}^{r}$ to $\mathbb{R}^{n}$.
<!-- These methods are not quite inverses: the results of `decode()` are restricted to the portion of $\mathbb{R}^{n}$ that can be represented through the underlying basis. -->
These methods wrap the `encode()` and `decode()` methods of the `basis` attribute; see [Preprocessing](sec-preprocessing) and [Basis Computation](sec-basis-computation) for more details.

## Training

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

## Prediction

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

## Model Persistence

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

## Summary

| Method | Description |
| :----- | :---------- |
| `project()` | Map high-dimensional states to their low-dimensional coordinates |
| `reconstruct()` | Use low-dimensional coordinates to construct a high-dimensional state |
| `fit()` | Use training data to infer the operators defining the ROM |
| `evaluate()` | Evaluate the reduced-order model for a given state / input |
| `predict()` | Solve the reduced-order model |
| `save()` | Save the ROM data to an HDF5 file |
| `load()` | Load a ROM from an HDF5 file |
