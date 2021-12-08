# ROM Classes

The core of `rom_operator_inference` is highly object oriented and defines several `ROM` classes that serve as the workhorse of the package.
The API for these classes adopts some principles from the [scikit-learn](https://scikit-learn.org/stable/index.html) [API](https://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects): there are `fit()` and `predict()` methods, hyperparameters are set in the constructor, estimated attributes end with underscore, and so on.

Each class corresponds to a type of full-order model (continuous vs. discrete, non-parametric vs. parametric) and a strategy for constructing the ROM.

| Class Name | Problem Statement | ROM Strategy |
| :--------- | :---------------: | :----------- |
| `InferredContinuousROM` | <img src="./img/documentation/eq01.svg"> | Operator Inference |
| `InferredDiscreteROM` | <img src="./img/documentation/eq02.svg"> | Operator Inference |
| `InterpolatedInferredContinuousROM` | <img src="./img/documentation/eq04.svg"> | Operator Inference |
| `InterpolatedInferredDiscreteROM` | <img src="./img/documentation/eq05.svg"> | Operator Inference |
| `AffineInferredContinuousROM` | <img src="./img/documentation/eq07.svg"> | Operator Inference |
| `AffineInferredDiscreteROM` | <img src="./img/documentation/eq08.svg"> | Operator Inference |
| `IntrusiveContinuousROM` | <img src="./img/documentation/eq01.svg"> | Intrusive Projection |
| `IntrusiveDiscreteROM` | <img src="./img/documentation/eq02.svg"> | Intrusive Projection |
| `AffineIntrusiveContinuousROM` | <img src="./img/documentation/eq07.svg"> | Intrusive Projection |
| `AffineIntrusiveDiscreteROM` | <img src="./img/documentation/eq08.svg"> | Intrusive Projection |


The following function may be helpful for selecting an appropriate class.

**`select_model_class(time, rom_strategy, parametric=False)`**: select the appropriate ROM model class for the situation.
Parameters:
- `time`: Type of full-order model to be reduced, either `"continuous"` or `"discrete"`.
- `rom_strategy`: Whether to use Operator Inference (`"inferred"`) or intrusive projection (`"intrusive"`) to compute the operators of the reduced model.
- `parametric`: Whether or not the model depends on an external parameter, and how to handle the parametric dependence. Options:
    - `False` (default): the problem is nonparametric.
    - `"interpolated"`: construct individual models for each sample parameter and [interpolate them](#interpolatedinferredcontinuousrom) for general parameter inputs. Only valid for `rom_strategy="inferred"`, and only when the parameter is a scalar.
    - `"affine"`: one or more operators in the problem [depends affinely](#affineintrusivecontinuousrom) on the parameter. Only valid for `rom_strategy="intrusive"`.

The return value is the class type for the situation, e.g., `InferredContinuousROM`.

## Constructor: Define Model Structure

All `ROM` classes are instantiated with a single argument, `modelform`, which is a string denoting the structure of the full-order operator **f**.
Each character in the string corresponds to a single term of the operator, given in the following table.

| Character | Name | Continuous Term | Discrete Term |
| :-------- | :--- | :-------------- | :------------ |
| `c` | Constant | <img src="./img/documentation/eq10.svg"> | <img src="./img/documentation/eq10.svg"> |
| `A` | Linear | <img src="./img/documentation/eq11.svg"> | <img src="./img/documentation/eq12.svg"> |
| `H` | Quadratic | <img src="./img/documentation/eq13.svg"> | <img src="./img/documentation/eq14.svg"> |
| `G` | Cubic | <img src="./img/documentation/eq15.svg"> | <img src="./img/documentation/eq16.svg"> |
| `B` | Input | <img src="./img/documentation/eq17.svg"> | <img src="./img/documentation/eq18.svg"> |


<!-- | `C` | Output | <img src="https://latex.codecogs.com/svg.latex?\mathbf{y}(t)=\hat{C}\hat{\mathbf{x}}(t)"> | <img src="https://latex.codecogs.com/svg.latex?\mathbf{y}_{k}=\hat{C}\hat{\mathbf{x}}_{k}"> | -->

These are all input as a single string.
Examples:

| `modelform` | Continuous ROM Structure | Discrete ROM Structure |
| :---------- | :----------------------- | ---------------------- |
|  `"A"`   | <img src="./img/documentation/eq19.svg"> | <img src="./img/documentation/eq20.svg">
|  `"cA"`  | <img src="./img/documentation/eq21.svg"> | <img src="./img/documentation/eq22.svg">
|  `"HB"`  | <img src="./img/documentation/eq23.svg"> | <img src="./img/documentation/eq24.svg">
|  `"cAHB"` | <img src="./img/documentation/eq25.svg"> | <img src="./img/documentation/eq26.svg">

## Attributes

All `ROM` classes have the following attributes.

- Structure of model:
    - `modelform`: set in the [constructor](#constructor).
    - `has_constant`: boolean, whether or not there is a constant term **c**.
    - `has_linear`: boolean, whether or not there is a linear term **Ax**.
    - `has_quadratic`: boolean, whether or not there is a quadratic term **H**(**x**⊗**x**).
    - `has_cubic`: boolean, whether or not there is a cubic term **G**(**x**⊗**x**⊗**x**).
    - `has_inputs`: boolean, whether or not there is an input term **Bu**.
    <!-- - `has_outputs`: boolean, whether or not there is an output **Cx**_. -->

- Dimensions, set in `fit()`:
    - `n`: Dimension of the original model
    - `r`: Dimension of the learned reduced-order model
    - `m`: Dimension of the input **u**, or `None` if `'B'` is not in `modelform`.
    <!-- - `l`: Dimension of the output **y**, or `None` if `has_outputs` is `False`. -->

- Reduced operators `c_`, `A_`, `H_`, `G_`, and `B_`, learned in `fit()`: the [NumPy](https://numpy.org/) arrays corresponding to the learned parts of the reduced-order model.
Set to `None` if the operator is not included in the prescribed `modelform` (e.g., if `modelform="AHG"`, then `c_` and `B_` are `None`).

- Basis matrix `Vr`: the _n_ x _r_ basis defining the mapping between the _n_-dimensional space of the full-order model and the reduced _r_-dimensional subspace of the reduced-order model (e.g., POD basis).
This is the first input to all `fit()` methods.
To save memory, inferred (but not intrusive) ROM classes allow entering `Vr=None`, which then assumes that other inputs for training are already projected to the _r_-dimensional subspace (e.g., **V**<sub>_r_</sub><sup>T</sup>**X** instead of **X**).

- Reduced model function `f_`, learned in `fit()`: the ROM function, defined by the reduced operators listed above.
For continuous models, `f_` has the following signature:
```python
def f_(t, x_, u):
    """ROM function for continuous models.

    Parameters
    ----------
    t : float
        Time, a scalar.

    x_ : (r,) ndarray
        Reduced state vector.

    u : func(float) -> (m,)
        Input function that maps time `t` to an input vector of length m.
    """
```
For discrete models, the signature is the following.
```python
def f_(x_, u):
    """ROM function for discrete models.

    Parameters
    ----------
    x_ : (r,) ndarray
        Reduced state vector.

    u : (m,) ndarray
        Input vector of length m corresponding to the state.
    """
```
The input argument `u` is only used if `B` is in `modelform`.

## Model Persistence

Trained ROM objects can be saved in [HDF5 format](http://docs.h5py.org/en/stable/index.html) with the `save_model()` method, and recovered later with the `load_model()` function.
Such files store metadata for the model class and structure, the reduced-order model operators (`c_`, `A_`, etc.), other attributes learned in `fit()`, and (optionally) the basis `Vr`.

**`load_model(loadfile)`**: Load a serialized model from an HDF5 file, created previously from a ROM object's `save_model()` method.

**`<ROMclass>.save_model(savefile, save_basis=True, overwrite=False)`**: Serialize the learned model, saving it in HDF5 format. The model can then be loaded with `load_model()`. _Currently implemented for nonparametric classes only._ Parameters:
- `savefile`: File to save to. If it does not end with `'.h5'`, this extension will be tacked on to the end.
- `savebasis`: If `True`, save the basis `Vr` as well as the reduced operators. If `False`, only save reduced operators.
- `overwrite`: If `True` and the specified file already exists, overwrite the file. If `False` and the specified file already exists, raise an error.

```python
>>> import rom_operator_inference as roi

# Assume model is a trained roi.InferredContinuousROM object.
>>> model.save_model("trained_rom.h5")          # Save a trained model.
>>> model2 = roi.load_model("trained_rom.h5")   # Load a model from file.
```

## InferredContinuousROM

This class constructs a reduced-order model for the continuous, nonparametric system

<p align="center"><img src="./img/documentation/eq27.svg"></p>

via Operator Inference [\[1\]](#references).
That is, given snapshot data, a basis, and a form for a reduced model, it computes the reduced model operators by solving an ordinary least-squares problem (see [mathematical details](sec-opinf-math)).

**`InferredContinuousROM.fit(Vr, X, Xdot, U=None, P=0)`**: Compute the operators of the reduced-order model via Operator Inference.
- **Parameters**
    - `Vr`: _n_ x _r_ basis for the linear reduced space on which the full-order model will be projected (for example, a POD basis matrix; see [`pre.pod_basis()`](#preprocessing-tools)). Each column is a basis vector. The column space of `Vr` should be a good approximation of the column space of the full-order snapshot matrix `X`. If given as `None`, `X` is assumed to be the projected snapshot matrix **V**<sub>_r_</sub><sup>T</sup>**X** and `Xdot` is assumed to be the projected time derivative matrix.
    - `X`: An _n_ x _k_ snapshot matrix of solutions to the full-order model, or the _r_ x _k_ projected snapshot matrix **V**<sub>_r_</sub><sup>T</sup>**X**. Each column is one snapshot.
    - `Xdot`: _n_ x _k_ snapshot time derivative matrix for the full-order model, or the _r_ x _k_ projected snapshot time derivative matrix. Each column is the time derivative d**x**/dt for the corresponding column of `X`. See the [`pre`](#preprocessing-tools) submodule for some simple derivative approximation tools.
    - `U`: _m_ x _k_ input matrix (or a _k_-vector if _m_ = 1). Each column is the input vector for the corresponding column of `X`. Only required when `'B'` is in `modelform`.
    - `P`: Tikhonov regularization matrix for the least-squares problem; see [`lstsq`](#least-squares-solvers).
- **Returns**
    - Trained `InferredContinuousROM` object.

**`InferredContinuousROM.predict(x0, t, u=None, **options)`**: Simulate the learned reduced-order model with [`scipy.integrate.solve_ivp()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
- **Parameters**
    - `x0`: Initial state vector, either full order (_n_-vector) or projected to reduced order (_r_-vector). If `Vr=None` in `fit()`, this must be the projected initial state **V**<sub>_r_</sub><sup>T</sup>**x**<sub>0</sub>.
    - `t`: Time domain, an _n_<sub>_t_</sub>-vector, over which to integrate the reduced-order model.
    - `u`: Input as a function of time, that is, a function mapping a `float` to an _m_-vector (or to a scalar if _m_ = 1). Alternatively, the _m_ x _n_<sub>_t_</sub> matrix (or _n_<sub>_t_</sub>-vector if _m_ = 1) where column _j_ is the input vector corresponding to time `t[j]`. In this case, **u**(_t_) is approximated by a cubic spline interpolating the given inputs. This argument is only required if `'B'` is in `modelform`.
    - Other keyword arguments for [`scipy.integrate.solve_ivp()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
- **Returns**
    - `X_ROM`: _n_ x _n_<sub>_t_</sub> matrix of approximate solution to the full-order system over `t`, or, if `Vr=None` in `fit()`, the _r_ x _n_<sub>_t_</sub> solution in the reduced-order space. Each column is one snapshot of the solution.


## InferredDiscreteROM

This class constructs a reduced-order model for the discrete, nonparametric system

<p align="center"><img src="./img/documentation/eq02.svg"></p>

via Operator Inference.

**`InferredDiscreteROM.fit(Vr, X, U=None, P=0)`**: Compute the operators of the reduced-order model via Operator Inference.
- **Parameters**
    - `Vr`: _n_ x _r_ basis for the linear reduced space on which the full-order model will be projected. Each column is a basis vector. The column space of `Vr` should be a good approximation of the column space of the full-order snapshot matrix `X`. If given as `None`, `X` is assumed to be the projected snapshot matrix **V**<sub>_r_</sub><sup>T</sup>**X**.
    - `X`: _n_ x _k_ snapshot matrix of solutions to the full-order model, or the _r_ x _k_ projected snapshot matrix **V**<sub>_r_</sub><sup>T</sup>**X**. Each column is one snapshot.
    - `U`: _m_ x _k-1_ input matrix (or a (_k_-1)-vector if _m_ = 1). Each column is the input for the corresponding column of `X`. Only required when `'B'` is in `modelform`.
    - `P`: Tikhonov regularization matrix for the least-squares problem; see [`lstsq`](#least-squares-solvers).
- **Returns**
    - Trained `InferredDiscreteROM` object.

**`InferredDiscreteROM.predict(x0, niters, U=None)`**: Step forward the learned ROM `niters` steps.
- **Parameters**
    - `x0`: Initial state vector, either full order (_n_-vector) or projected to reduced order (_r_-vector). If `Vr=None` in `fit()`, this must be the projected initial state **V**<sub>_r_</sub><sup>T</sup>**x**<sub>0</sub>.
    - `niters`: Number of times to step the system forward.
    - `U`: Inputs for the next `niters`-1 time steps, as an _m_ x `niters`-1 matrix (or an (`niters`-1)-vector if _m_ = 1). This argument is only required if `'B'` is in `modelform`.
- **Returns**
    - `X_ROM`: _n_ x `niters` matrix of approximate solutions to the full-order system, including the initial condition; or, if `Vr=None` in `fit()`, the _r_ x `niters` solution in the reduced-order space. Each column is one iteration of the solution.
