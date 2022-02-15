(sec-api)=
# API Reference (Dummy)

This page contains code documentation for `rom_operator_inference` classes and functions.
The code itself is also internally documented and can be accessed on the fly with dynamic object introspection, e.g.,

```python
>>> import rom_operator_inference as roi
>>> help(roi.InferredContinuousROM)
```
<!-- **TODO**: The complete documentation at _\<insert link to sphinx-generated readthedocs page\>_. -->
<!-- Here we include a short catalog of functions and their inputs. -->

## ROM Classes

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

### Constructor

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

### Attributes

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

### Model Persistence

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

### InferredContinuousROM

This class constructs a reduced-order model for the continuous, nonparametric system

<p align="center"><img src="./img/documentation/eq27.svg"></p>

via Operator Inference [\[1\]](#references).
That is, given snapshot data, a basis, and a form for a reduced model, it computes the reduced model operators by solving an ordinary least-squares problem (see [**Operator-Inference**](sec-opinf-math)).

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


### InferredDiscreteROM

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


### InterpolatedInferredContinuousROM

This class constructs a reduced-order model for the continuous, parametric system
<p align="center"><img src="./img/documentation/eq28.svg"></p>

via Operator Inference.
The strategy is to take snapshot data for several parameter samples and a global basis, compute a reduced model for each parameter sample via Operator Inference, then construct a general parametric model by interpolating the entries of the inferred operators [\[1\]](#references).

**`InterpolatedInferredContinuousROM.fit(Vr, µs, Xs, Xdots, Us=None, P=0)`**: Compute the operators of the reduced-order model via Operator Inference.
- **Parameters**
    - `Vr`: The (global) _n_ x _r_ basis for the linear reduced space on which the full-order model will be projected. Each column is a basis vector. The column space of `Vr` should be a good approximation of the column space of the full-order snapshot matrices `Xs`. If given as `None`, `Xs` is assumed to be the list of the projected snapshot matrices **V**<sub>_r_</sub><sup>T</sup>**X**<sub>_i_</sub> and `Xdots` is assumed to be the list of projected time derivative matrices.
    - `µs`: _s_ parameter values corresponding to the snapshot sets.
    - `Xs`: List of _s_ snapshot matrices, each _n_ x _k_ (full-order solutions) or _r_ x _k_ (projected solutions). The _i_th array `Xs[i]` corresponds to the _i_th parameter, `µs[i]`; each column each of array is one snapshot.
    - `Xdots`: List of _s_ snapshot time derivative matrices, each _n_ x _k_ (full-order velocities) or _r_ x _k_ (projected velocities).  The _i_th array `Xdots[i]` corresponds to the _i_th parameter, `µs[i]`. The _j_th column of the _i_th array, `Xdots[i][:,j]`, is the time derivative d**x**/dt for the corresponding snapshot column `Xs[i][:,j]`.
    - `Us`: List of _s_ input matrices, each _m_ x _k_ (or a _k_-vector if _m_=1). The _i_th array `Us[i]` corresponds to the _i_th parameter, `µs[i]`. The _j_th column of the _i_th array, `Us[i][:,j]`, is the input for the corresponding snapshot `Xs[i][:,j]`. Only required when `'B'` is in `modelform`.
    - `P`: Tikhonov regularization matrix for the least-squares problem; see [`lstsq`](#least-squares-solvers).
- **Returns**
    - Trained `InterpolatedInferredContinuousROM` object.

**`InterpolatedInferredContinuousROM.predict(µ, x0, t, u=None, **options)`**: Simulate the learned reduced-order model with [`scipy.integrate.solve_ivp()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
- **Parameters**
    - `µ`: Parameter value at which to simulate the ROM.
    - `x0`: Initial state vector, either full order (_n_-vector) or projected to reduced order (_r_-vector). If `Vr=None` in `fit()`, this must be the projected initial state **V**<sub>_r_</sub><sup>T</sup>**x**<sub>0</sub>.
    - `t`: Time domain, an _n_<sub>_t_</sub>-vector, over which to integrate the reduced-order model.
    - `u`: Input as a function of time, that is, a function mapping a `float` to an _m_-vector (or to a scalar if _m_ = 1). Alternatively, the _m_ x _n_<sub>_t_</sub> matrix (or _n_<sub>_t_</sub>-vector if _m_ = 1) where column _j_ is the input vector corresponding to time `t[j]`. In this case, **u**(_t_) is approximated by a cubic spline interpolating the given inputs. This argument is only required if `'B'` is in `modelform`.
    - Other keyword arguments for [`scipy.integrate.solve_ivp()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
- **Returns**
    - `X_ROM`: _n_ x _n_<sub>_t_</sub> matrix of approximate solution to the full-order system over `t`, or, if `Vr=None` in `fit()`, the _r_ x _n_<sub>_t_</sub> solution in the reduced-order space. Each column is one snapshot of the solution.


### InterpolatedInferredDiscreteROM

This class constructs a reduced-order model for the continuous, parametric system
<p align="center"><img src="./img/documentation/eq29.svg"></p>

via Operator Inference.
The strategy is to take snapshot data for several parameter samples and a global basis, compute a reduced model for each parameter sample via Operator Inference, then construct a general parametric model by interpolating the entries of the inferred operators [\[1\]](#references).

**`InterpolatedInferredDiscreteROM.fit(Vr, µs, Xs, Us=None, P=0)`**: Compute the operators of the reduced-order model via Operator Inference.
- **Parameters**
    - `Vr`: The (global) _n_ x _r_ basis for the linear reduced space on which the full-order model will be projected. Each column is a basis vector. The column space of `Vr` should be a good approximation of the column space of the full-order snapshot matrices `Xs`. If given as `None`, `Xs` is assumed to be the list of the projected snapshot matrices **V**<sub>_r_</sub><sup>T</sup>**X**<sub>_i_</sub>.
    - `µs`: _s_ parameter values corresponding to the snapshot sets.
    - `Xs`: List of _s_ snapshot matrices, each _n_ x _k_ (full-order solutions) or _r_ x _k_ (projected solutions). The _i_th array `Xs[i]` corresponds to the _i_th parameter, `µs[i]`; each column each of array is one snapshot.
    - `Us`: List of _s_ input matrices, each _m_ x _k_ (or a _k_-vector if _m_=1). The _i_th array `Us[i]` corresponds to the _i_th parameter, `µs[i]`. The _j_th column of the _i_th array, `Us[i][:,j]`, is the input for the corresponding snapshot `Xs[i][:,j]`. Only required when `'B'` is in `modelform`.
    - `P`: Tikhonov regularization matrix for the least-squares problem; see [`lstsq`](#least-squares-solvers).
- **Returns**
    - Trained `InterpolatedInferredDiscreteROM` object.

**`InterpolatedInferredDiscreteROM.predict(µ, x0, niters, U=None)`**: Step forward the learned ROM `niters` steps.
- **Parameters**
    - `µ`: Parameter value at which to simulate the ROM.
    - `x0`: Initial state vector, either full order (_n_-vector) or projected to reduced order (_r_-vector). If `Vr=None` in `fit()`, this must be the projected initial state **V**<sub>_r_</sub><sup>T</sup>**x**<sub>0</sub>.
    - `niters`: Number of times to step the system forward.
    - `U`: Inputs for the next `niters`-1 time steps, as an _m_ x `niters`-1 matrix (or an (`niters`-1)-vector if _m_ = 1). This argument is only required if `'B'` is in `modelform`.
- **Returns**
    - `X_ROM`: _n_ x `niters` matrix of approximate solutions to the full-order system, including the initial condition; or, if `Vr=None` in `fit()`, the _r_ x _n_<sub>_t_</sub> solution in the reduced-order space. Each column is one iteration of the solution.


### AffineInferredContinuousROM

This class constructs a reduced-order model via Operator Inference for the continuous, affinely parametric system

<p align="center"><img src="./img/documentation/eq31.svg"></p>

where the operators that define **f** may only depend affinely on the parameter, e.g.,

<p align="center"><img src="./img/documentation/eq32.svg"></p>

**`AffineInferredContinuousROM.fit(Vr, µs, affines, Xs, Xdots, Us=None, P=0)`**: Compute the operators of the reduced-order model via Operator Inference.
- **Parameters**
    - `Vr`: The (global) _n_ x _r_ basis for the linear reduced space on which the full-order model will be projected. Each column is a basis vector. The column space of `Vr` should be a good approximation of the column space of the full-order snapshot matrices `Xs`. If given as `None`, `Xs` is assumed to be the list of the projected snapshot matrices **V**<sub>_r_</sub><sup>T</sup>**X**<sub>_i_</sub> and `Xdots` is assumed to be the list of projected time derivative matrices.
    - `µs`: _s_ parameter values corresponding to the snapshot sets.
    - `affines` A dictionary mapping labels of the operators that depend affinely on the parameter to the list of functions that define that affine dependence. The keys are entries of `modelform`. For example, if the constant term has the affine structure **c**(_**µ**_) = _θ_<sub>1</sub>(_**µ**_)**c**<sub>1</sub> + _θ_<sub>2</sub>(_**µ**_)**c**<sub>2</sub> + _θ_<sub>3</sub>(_**µ**_)**c**<sub>3</sub>, then `'c' -> [θ1, θ2, θ3]`.
    - `Xs`: List of _s_ snapshot matrices, each _n_ x _k_ (full-order solutions) or _r_ x _k_ (projected solutions). The _i_th array `Xs[i]` corresponds to the _i_th parameter, `µs[i]`; each column each of array is one snapshot.
    - `Xdots`: List of _s_ snapshot time derivative matrices, each _n_ x _k_ (full-order velocities) or _r_ x _k_ (projected velocities).  The _i_th array `Xdots[i]` corresponds to the _i_th parameter, `µs[i]`. The _j_th column of the _i_th array, `Xdots[i][:,j]`, is the time derivative d**x**/dt for the corresponding snapshot column `Xs[i][:,j]`.
    - `Us`: List of _s_ input matrices, each _m_ x _k_ (or a _k_-vector if _m_=1). The _i_th array `Us[i]` corresponds to the _i_th parameter, `µs[i]`. The _j_th column of the _i_th array, `Us[i][:,j]`, is the input for the corresponding snapshot `Xs[i][:,j]`. Only required when `'B'` is in `modelform`.
    - `P`: Tikhonov regularization matrix for the least-squares problem; see [`lstsq`](#least-squares-solvers).
- **Returns**:
    - Trained `AffineInferredContinuousROM` object.

**`AffineInferredContinuousROM.predict(µ, x0, t, u=None, **options)`**: Simulate the learned reduced-order model at the given parameter value with [`scipy.integrate.solve_ivp()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
- **Parameters**
    - `µ`: Parameter value at which to simulate the model.
    - `x0`: Initial state vector, either full order (_n_-vector) or projected to reduced order (_r_-vector).
    - `t`: Time domain, an _n_<sub>_t_</sub>-vector, over which to integrate the reduced-order model.
    - `u`: Input as a function of time, that is, a function mapping a `float` to an _m_-vector (or to a scalar if _m_ = 1). Alternatively, the _m_ x _n_<sub>_t_</sub> matrix (or _n_<sub>_t_</sub>-vector if _m_ = 1) where column _j_ is the input vector corresponding to time `t[j]`. In this case, **u**(_t_) is approximated by a cubic spline interpolating the given inputs. This argument is only required if `'B'` is in `modelform`.
    - Other keyword arguments for [`scipy.integrate.solve_ivp()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
- **Returns**
    - `X_ROM`: _n_ x _n_<sub>_t_</sub> matrix of approximate solution to the full-order system over `t`. Each column is one snapshot of the solution.


### AffineInferredDiscreteROM

This class constructs a reduced-order model via Operator Inference for the discrete, affinely parametric system

<p align="center"><img src="./img/documentation/eq34.svg"></p>

where the operators that define **f** may only depend affinely on the parameter, e.g.,

<p align="center"><img src="./img/documentation/eq32.svg"></p>

**`AffineInferredDiscreteROM.fit(Vr, µs, affines, Xs, Us, P)`**: Compute the operators of the reduced-order model via Operator Inference.
- **Parameters**:
    - `Vr`: The (global) _n_ x _r_ basis for the linear reduced space on which the full-order model will be projected. Each column is a basis vector. The column space of `Vr` should be a good approximation of the column space of the full-order snapshot matrices `Xs`. If given as `None`, `Xs` is assumed to be the list of the projected snapshot matrices **V**<sub>_r_</sub><sup>T</sup>**X**<sub>_i_</sub>.
    - `µs`: _s_ parameter values corresponding to the snapshot sets.
    - `affines` A dictionary mapping labels of the operators that depend affinely on the parameter to the list of functions that define that affine dependence. The keys are entries of `modelform`. For example, if the constant term has the affine structure **c**(_**µ**_) = _θ_<sub>1</sub>(_**µ**_)**c**<sub>1</sub> + _θ_<sub>2</sub>(_**µ**_)**c**<sub>2</sub> + _θ_<sub>3</sub>(_**µ**_)**c**<sub>3</sub>, then `'c' -> [θ1, θ2, θ3]`.
    - `Xs`: List of _s_ snapshot matrices, each _n_ x _k_ (full-order solutions) or _r_ x _k_ (projected solutions). The _i_th array `Xs[i]` corresponds to the _i_th parameter, `µs[i]`; each column each of array is one snapshot.
    - `Us`: List of _s_ input matrices, each _m_ x _k_ (or a _k_-vector if _m_=1). The _i_th array `Us[i]` corresponds to the _i_th parameter, `µs[i]`. The _j_th column of the _i_th array, `Us[i][:,j]`, is the input for the corresponding snapshot `Xs[i][:,j]`. Only required when `'B'` is in `modelform`.
    - `P`: Tikhonov regularization matrix for the least-squares problem; see [`lstsq`](#least-squares-solvers).
- **Returns**:
    - Trained `AffineInferredDiscreteROM` object.

**`AffineInferredDiscreteROM.predict(µ, x0, niters, U=None)`**: Step forward the learned ROM `niters` steps at the given parameter value.
- **Parameters**
    - `µ`: Parameter value at which to simulate the model.
    - `x0`: Initial state vector, either full order (_n_-vector) or projected to reduced order (_r_-vector).
    - `niters`: Number of times to step the system forward.
    - `U`: Inputs for the next `niters`-1 time steps, as an _m_ x `niters`-1 matrix (or an (`niters`-1)-vector if _m_ = 1). This argument is only required if `'B'` is in `modelform`.
- **Returns**
    - `X_ROM`: _n_ x `niters` matrix of approximate solutions to the full-order system, including the initial condition. Each column is one iteration of the solution.


### IntrusiveContinuousROM

This class constructs a reduced-order model for the continuous, nonparametric system

<p align="center"><img src="./img/documentation/eq27.svg"></p>

via intrusive projection, i.e.,

<p align="center"><img src="./img/documentation/eq30.svg"></p>

where ⊗ denotes the full matrix Kronecker product.
The class requires the actual full-order operators (**c**, **A**, **H**, **G**, and/or **B**) that define **f**.

**`IntrusiveContinuousROM.fit(Vr, operators)`**: Compute the operators of the reduced-order model by projecting the operators of the full-order model.
- **Parameters**
    - `Vr`: _n_ x _r_ basis for the linear reduced space on which the full-order operators will be projected.
    - `operators`: A dictionary mapping labels to the full-order operators that define **f**. The operators are indexed by the entries of `modelform`; for example, if `modelform="cHB"`, then `operators={'c':c, 'H':H, 'B':B}`.
- **Returns**
    - Trained `IntrusiveContinuousROM` object.

**`IntrusiveContinuousROM.predict(x0, t, u=None, **options)`**: Simulate the learned reduced-order model with [`scipy.integrate.solve_ivp()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
- **Parameters**
    - `x0`: Initial state vector, either full order (_n_-vector) or projected to reduced order (_r_-vector).
    - `t`: Time domain, an _n_<sub>_t_</sub>-vector, over which to integrate the reduced-order model.
    - `u`: Input as a function of time, that is, a function mapping a `float` to an _m_-vector (or to a scalar if _m_ = 1). Alternatively, the _m_ x _n_<sub>_t_</sub> matrix (or _n_<sub>_t_</sub>-vector if _m_ = 1) where column _j_ is the input vector corresponding to time `t[j]`. In this case, **u**(_t_) is approximated by a cubic spline interpolating the given inputs. This argument is only required if `'B'` is in `modelform`.
    - Other keyword arguments for [`scipy.integrate.solve_ivp()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
- **Returns**
    - `X_ROM`: _n_ x _n_<sub>_t_</sub> matrix of approximate solution to the full-order system over `t`. Each column is one snapshot of the solution.


### IntrusiveDiscreteROM

This class constructs a reduced-order model for the discrete, nonparametric system

<p align="center"><img src="./img/documentation/eq02.svg"></p>

via intrusive projection, i.e.,

<p align="center"><img src="./img/documentation/eq30.svg"></p>

The class requires the actual full-order operators (**c**, **A**, **H**, **G**, and/or **B**) that define **f**.

**`IntrusiveDiscreteROM.fit(Vr, operators)`**: Compute the operators of the reduced-order model by projecting the operators of the full-order model.
- **Parameters**
    - `Vr`: _n_ x _r_ basis for the linear reduced space on which the full-order operators will be projected.
    - `operators`: A dictionary mapping labels to the full-order operators that define **f**. The operators are indexed by the entries of `modelform`; for example, if `modelform="cHB"`, then `operators={'c':c, 'H':H, 'B':B}`.
- **Returns**
    - Trained `IntrusiveDiscreteROM` object.

**`IntrusiveDiscreteROM.predict(x0, niters, U=None)`**: Step forward the learned ROM `niters` steps.
- **Parameters**
    - `x0`: Initial state vector, either full order (_n_-vector) or projected to reduced order (_r_-vector).
    - `niters`: Number of times to step the system forward.
    - `U`: Inputs for the next `niters`-1 time steps, as an _m_ x `niters`-1 matrix (or an (`niters`-1)-vector if _m_ = 1). This argument is only required if `'B'` is in `modelform`.
- **Returns**
    - `X_ROM`: _n_ x `niters` matrix of approximate solutions to the full-order system, including the initial condition. Each column is one iteration of the solution.


### AffineIntrusiveContinuousROM

This class constructs a reduced-order model for the continuous, affinely parametric system

<p align="center"><img src="./img/documentation/eq31.svg"></p>

where the operators that define **f** may only depend affinely on the parameter, e.g.,

<p align="center"><img src="./img/documentation/eq32.svg"></p>

The reduction is done via intrusive projection, i.e.,

<p align="center"><img src="./img/documentation/eq33.svg"></p>

The class requires the actual full-order operators (**c**, **A**, **H**, and/or **B**) that define **f** _and_ the functions that define any affine parameter dependencies (i.e., the _θ_<sub>_ℓ_</sub> functions).

**`AffineIntrusiveContinuousROM.fit(Vr, affines, operators)`**: Compute the operators of the reduced-order model by projecting the operators of the full-order model.
- **Parameters**
    - `Vr`: _n_ x _r_ basis for the linear reduced space on which the full-order operators will be projected.
    - `affines` A dictionary mapping labels of the operators that depend affinely on the parameter to the list of functions that define that affine dependence. The keys are entries of `modelform`. For example, if the constant term has the affine structure **c**(_**µ**_) = _θ_<sub>1</sub>(_**µ**_)**c**<sub>1</sub> + _θ_<sub>2</sub>(_**µ**_)**c**<sub>2</sub> + _θ_<sub>3</sub>(_**µ**_)**c**<sub>3</sub>, then `'c' -> [θ1, θ2, θ3]`.
    - `operators`: A dictionary mapping labels to the full-order operators that define **f**. The keys are entries of `modelform`. Terms with affine structure should be given as a list of the component matrices. For example, suppose `modelform="cA"`. If **A** has the affine structure **A**(_**µ**_) = _θ_<sub>1</sub>(_**µ**_)**A**<sub>1</sub> + _θ_<sub>2</sub>(_**µ**_)**A**<sub>2</sub>, then `'A' -> [A1, A2]`. If **c** does not vary with the parameter, then `'c' -> c`, the complete full-order order.
- **Returns**:
    - Trained `AffineIntrusiveContinuousROM` object.

**`AffineIntrusiveContinuousROM.predict(µ, x0, t, u=None, **options)`**: Simulate the learned reduced-order model at the given parameter value with [`scipy.integrate.solve_ivp()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
- **Parameters**
    - `µ`: Parameter value at which to simulate the model.
    - `x0`: Initial state vector, either full order (_n_-vector) or projected to reduced order (_r_-vector).
    - `t`: Time domain, an _n_<sub>_t_</sub>-vector, over which to integrate the reduced-order model.
    - `u`: Input as a function of time, that is, a function mapping a `float` to an _m_-vector (or to a scalar if _m_ = 1). Alternatively, the _m_ x _n_<sub>_t_</sub> matrix (or _n_<sub>_t_</sub>-vector if _m_ = 1) where column _j_ is the input vector corresponding to time `t[j]`. In this case, **u**(_t_) is approximated by a cubic spline interpolating the given inputs. This argument is only required if `'B'` is in `modelform`.
    - Other keyword arguments for [`scipy.integrate.solve_ivp()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
- **Returns**
    - `X_ROM`: _n_ x _n_<sub>_t_</sub> matrix of approximate solution to the full-order system over `t`. Each column is one snapshot of the solution.


### AffineIntrusiveDiscreteROM

This class constructs a reduced-order model for the discrete, affinely parametric system

<p align="center"><img src="./img/documentation/eq34.svg"></p>

where the operators that define **f** may only depend affinely on the parameter, e.g.,

<p align="center"><img src="./img/documentation/eq32.svg"></p>

The reduction is done via intrusive projection, i.e.,

<p align="center"><img src="./img/documentation/eq33.svg"></p>

The class requires the actual full-order operators (**c**, **A**, **H**, and/or **B**) that define **f** _and_ the functions that define any affine parameter dependencies (i.e., the _θ_<sub>_ℓ_</sub> functions).

**`AffineIntrusiveDiscreteROM.fit(Vr, affines, operators)`**: Compute the operators of the reduced-order model by projecting the operators of the full-order model.
- **Parameters**
    - `Vr`: _n_ x _r_ basis for the linear reduced space on which the full-order operators will be projected.
    - `affines` A dictionary mapping labels of the operators that depend affinely on the parameter to the list of functions that define that affine dependence. The keys are entries of `modelform`. For example, if the constant term has the affine structure **c**(_**µ**_) = _θ_<sub>1</sub>(_**µ**_)**c**<sub>1</sub> + _θ_<sub>2</sub>(_**µ**_)**c**<sub>2</sub> + _θ_<sub>3</sub>(_**µ**_)**c**<sub>3</sub>, then `'c' -> [θ1, θ2, θ3]`.
    - `operators`: A dictionary mapping labels to the full-order operators that define **f**. The keys are entries of `modelform`. Terms with affine structure should be given as a list of the component matrices. For example, suppose `modelform="cA"`. If **A** has the affine structure **A**(_**µ**_) = _θ_<sub>1</sub>(_**µ**_)**A**<sub>1</sub> + _θ_<sub>2</sub>(_**µ**_)**A**<sub>2</sub>, then `'A' -> [A1, A2]`. If **c** does not vary with the parameter, then `'c' -> c`, the complete full-order order.
- **Returns**:
    - Trained `AffineIntrusiveDiscreteROM` object.

**`AffineIntrusiveDiscreteROM.predict(µ, x0, niters, U=None)`**: Step forward the learned ROM `niters` steps at the given parameter value.
- **Parameters**
    - `µ`: Parameter value at which to simulate the model.
    - `x0`: Initial state vector, either full order (_n_-vector) or projected to reduced order (_r_-vector).
    - `niters`: Number of times to step the system forward.
    - `U`: Inputs for the next `niters`-1 time steps, as an _m_ x `niters`-1 matrix (or an (`niters`-1)-vector if _m_ = 1). This argument is only required if `'B'` is in `modelform`.
- **Returns**
    - `X_ROM`: _n_ x `niters` matrix of approximate solutions to the full-order system, including the initial condition. Each column is one iteration of the solution.


## Least-squares Solvers

The `lstsq` module defines classes to handle the solution of the actual Operator Inference least-squares problems.
Each class has `fit(A, B)`and `predict(𝚪)` methods, where the arguments **A**, **B** = [**b**<sub>_1_</sub>, ..., **b**<sub>_r_</sub>], and **𝚪** correspond to the problem

<p align="center"><img src="./img/documentation/eq44.svg"></p>

which is the Frobenius-norm least-squares problem decoupled by columns.
In the context of Operator Inference, **A** is the data matrix **D**, **B** is the projected time derivative matrix **R**<sup>T</sup>, and **𝚪** determines the penalties on the entries of the operator matrix **O**<sup>T</sup>.
See [**Operator-Inference**](sec-opinf-math) for additional mathematical explanation.

<!-- TODO: cond(), regcond(), misfit(), residual() for each class. -->

### lstsq.SolverL2

Solve (LS) with _L_<sub>2</sub> regularization, meaning **𝚪**<sub>_j_</sub> = λ**I**, _j = 1,...,r_, for some λ ≥ 0.
Setting λ = 0 is equivalent to ordinary, non-regularized least squares.
The solution is obtained via the SVD of **A**:

<p align="center"><img src="./img/documentation/eq45.svg"></p>

**lstsq.SolverL2.fit(A, B)**: Take the SVD of **A** in preparation to solve (LS).
- **Parameters**
    - `A`: _k_ x _d_ data matrix.
    - `B`: _k_ x _r_ right-hand-side matrix.
- **Returns**
    - Initialized `lstsq.SolverL2` object.

**lstsq.SolverL2.predict(λ)**: Solve (LS) with regularization hyperparameter λ.
- **Parameters**
    - `λ`: Scalar, non-negative regularization hyperparameter.
- **Returns**
    - `X`: _d_ x _r_ matrix of least-squares solutions.

### lstsq.SolverL2Decoupled

Solve (LS) with a different _L_<sub>2</sub> regularization for each column of **B**, i.e., **𝚪**<sub>_j_</sub> = λ<sub>_j_</sub>**I** where λ<sub>_j_</sub> ≥ 0, _j = 1,...,r_.
The solution is obtained via the SVD of **A**:

<p align="center"><img src="./img/documentation/eq46.svg"></p>

**lstsq.SolverL2Decoupled.fit()**: Take the SVD of **A** in preparation to solve (LS).
- **Parameters**
    - `A`: _k_ x _d_ data matrix.
    - `B`: _k_ x _r_ right-hand-side matrix.
- **Returns**
    - Initialized `lstsq.SolverL2Decoupled` object.

**lstsq.SolverL2Decoupled.predict(λs)**: Solve (LS) with parameters `λs` = [λ<sub>_1_</sub>, ..., λ<sub>_r_</sub>].
- **Parameters**
    - `λs`: _r_ non-negative regularization hyperparameters.
- **Returns**
    - `X`: _d_ x _r_ matrix of least-squares solutions.

### lstsq.SolverTikhonov

Solve (LS) with a given matrix **𝚪**, using the same **𝚪** for each _j_: **𝚪**<sub>_j_</sub> = **𝚪** for _j = 1, ..., r_.
The solution is obtained by solving the modified normal equations (**A**<sup>T</sup>**A** + **𝚪**<sup>T</sup>**𝚪**)**x**<sub>_j_</sub> = **A**<sup>T</sup>**b**<sub>_j_</sub> via [`scipy.linalg.solve()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve.html) with `assume_a="pos"` ([POSV](https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/lapack-linear-equation-routines/lapack-linear-equation-driver-routines/posv.html) from LAPACK).
If these equations are highly ill-conditioned, the solver uses [`scipy.linalg.lstsq()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html) ([GELSD](https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/lapack-routines/lapack-least-squares-and-eigenvalue-problem-routines/lapack-least-squares-and-eigenvalue-problem-driver-routines/linear-least-squares-lls-problems-lapack-driver-routines/gelsd.html) from LAPACK) as a backup, which is slightly slower but generally more stable.

**lstsq.SolverTikhonov.fit(A, B)**: Compute **A**<sup>T</sup>**A** and **A**<sup>T</sup>**B** in preparation to solve (LS).
- **Parameters**
    - `A`: _k_ x _d_ data matrix.
    - `B`: _k_ x _r_ right-hand-side matrix.
- **Returns**
    - Initialized `lstsq.SolverTikhonov` object.

**lstsq.SolverTikhonov.predict(P)**:
- **Parameters**
    - `P`: _d_ x _d_ regularization matrix **𝚪** OR a _d_-vector, in which case **𝚪** is interpreted as `diag(P)`.
- **Returns**
    - `X`: _d_ x _r_ matrix of least-squares solutions.

### lstsq.SolverTikhonovDecoupled

Solve (LS) with given matrices **𝚪**<sub>_j_</sub>, _j = 1, ..., r_, by solving the modified normal equations (**A**<sup>T</sup>**A** + **𝚪**<sub>_j_</sub><sup>T</sup>**𝚪**<sub>_j_</sub>)**x**<sub>_j_</sub> = **A**<sup>T</sup>**b**<sub>_j_</sub> ([unless ill conditioned](#lstsqsolvertikhonov)).

**lstsq.SolverTikhonovDecoupled.fit(A, B)**:
- **Parameters**
    - `A`: _k_ x _d_ data matrix.
    - `B`: _k_ x _r_ right-hand-side matrix.
- **Returns**
    - Initialized `lstsq.SolverTikhonovDecoupled` object.

**lstsq.SolverTikhonovDecoupled.predict(Ps)**:
- **Parameters**
    - `Ps`: sequence of _r_ (_d_ x _d_) regularization matrices **𝚪**<sub>_1_</sub>, ..., **𝚪**<sub>_r_</sub>, OR sequence of _r_ _d_-vectors, in which case **𝚪**<sub>_j_</sub> is interpreted as `diag(P[j])`.
- **Returns**
    - `X`: _d_ x _r_ matrix of least-squares solutions.

The following helper functions interface with the least-squares solver classes.

**`lstsq.lstsq_size(modelform, r, m=0, affines=None)`**: Calculate the number of columns of the operator matrix **O** in the Operator Inference least squares problem (called _d(r,m)_ in [**Operator-Inference**](sec-opinf-math)).
Useful for determining the dimensions of the regularization matrix **𝚪**.
- **Parameters**
    - `modelform`: the structure of the [desired model](#constructor).
    - `r`: Dimension of the reduced order model.
    - `m`: Dimension of the inputs of the model. Must be zero unless `'B'` is in `modelform`.
    - `affines`: A dictionary mapping labels of the operators that depend affinely on the parameter to the list of functions that define that affine dependence. The keys are entries of `modelform`. For example, if the constant term has the affine structure **c**(_**µ**_) = _θ_<sub>1</sub>(_**µ**_)**c**<sub>1</sub> + _θ_<sub>2</sub>(_**µ**_)**c**<sub>2</sub> + _θ_<sub>3</sub>(_**µ**_)**c**<sub>3</sub>, then `'c' -> [θ1, θ2, θ3]`.
- **Returns**
    - Number of columns of the unknown matrix in the Operator Inference least squares problem.

**`lstsq.solver(A, b, P=0)`**: Select and initialize an appropriate solver for the (LS), i.e., pick a solver class and call its `fit()` method.

**`lstsq.solve(A, b, P=0)`**: Select, initialize, and execute an appropriate solver for (LS), i.e., pick a solver class and call its `fit()` _and_ `predict()` methods.

## Preprocessing Tools

The `pre` submodule is a collection of common routines for preparing data to be used by the `ROM` classes.
None of these routines are novel, but they may be instructive for new Python users.

**`pre.shift(X, shift_by=None)`**: Shift the columns of `X` by the vector `shift_by`. If `shift_by=None`, shift `X` by the mean of its columns.

**`pre.scale(X, scale_to, scale_from=None)`**: Scale the entries of `X` from the interval `[scale_from[0], scale_from[1]]` to the interval `[scale_to[0], scale_to[1]]`. If `scale_from=None`, learn the scaling by setting `scale_from[0] = min(X)`; `scale_from[1] = max(X)`.

<!-- TODO: kwarg for absolute vs minmax scaling -->

**`pre.pod_basis(X, r=None, mode="dense", **options)`**: Compute the POD basis of rank `r` and the associated singular values for a snapshot matrix `X`. If `r = None`, compute all singular vectors / values. This function simply wraps a few SVD methods, selected by `mode`:
- `mode="dense"`: [`scipy.linalg.svd()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html)
- `mode="sparse"`: [`scipy.sparse.linalg.svds()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html)
- `mode="randomized"`: [`sklearn.utils.extmath.randomized_svd()`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html)

Use `**options` to specify additional parameters for these wrapped functions.

**`pre.svdval_decay(singular_values, eps, plot=False)`**: Count the number of singular values that are greater than `eps`. The singular values can be computed with, for example, `singular_values = scipy.linalg.svdvals(X)` where `X` is a snapshot matrix. If `plot=True`, plot the singular values on a log scale.

**`pre.cumulative_energy(singular_values, thresh, plot=False)`**: Compute the number of singular values needed to surpass the energy threshold `thresh`; the energy of the first _r_ singular values is defined by <p align="center"><img src="./img/documentation/eq36.svg"></p>The singular values can be computed with, for example, `singular_values = scipy.linalg.svdvals(X)` where `X` is a snapshot matrix. If `plot=True`, plot the cumulative energy on a log scale.

**`pre.projection_error(X, Vr)`**: Compute the relative Frobenius-norm projection error on **X** induced by the basis matrix **V**<sub>_r_</sub>, <p align="center"><img src="./img/documentation/eq37.svg"></p>

<!-- TODO: allow norms other than Frobenius? -->

**`pre.minimal_projection_error(X, V, eps, plot=False)`**: Compute the number of POD basis vectors required to obtain a projection error less than `eps`, up to the number of columns of `V`. If `plot=True`, plot the projection error on a log scale as a function of the basis size.

<!-- TODO: allow norms other than Frobenius? -->

**`pre.reproject_continuous(f, Vr, X, U=None)`**: Sample re-projected trajectories [\[5\]](#references) of the continuous system of ODEs defined by `f`.

**`pre.reproject_discrete(f, Vr, x0, niters, U=None)`**: Sample re-projected trajectories [\[5\]](#references) of the discrete dynamical system defined by `f`.

**`pre.xdot_uniform(X, dt, order=2)`**: Approximate the first time derivative of a snapshot matrix `X` in which the snapshots are evenly spaced in time.

**`pre.xdot_nonuniform(X, t)`**: Approximate the first time derivative of a snapshot matrix `X` in which the snapshots are **not** evenly spaced in time.

**`pre.xdot(X, *args, **kwargs)`**: Call `pre.xdot_uniform()` or `pre.xdot_nonuniform()`, depending on the arguments.


## Postprocessing Tools

The `post` submodule is a collection of common routines for computing the absolute and relative errors produced by a ROM approximation.
Given a norm, "true" data **X**, and an approximation **Y** to **X**, these errors are defined by <p align="center"><img src="./img/documentation/eq38.svg"></p>

**`post.frobenius_error(X, Y)`**: Compute the absolute and relative Frobenius-norm errors between snapshot sets `X` and `Y`, assuming `Y` is an approximation to `X`.
The [Frobenius matrix norm](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm) is defined by <p align="center"><img src="./img/documentation/eq39.svg"></p>

**`post.lp_error(X, Y, p=2, normalize=False)`**: Compute the absolute and relative _l_<sup>_p_</sup>-norm errors between snapshot sets `X` and `Y`, assuming `Y` is an approximation to `X`.
The [_l_<sup>_p_</sup> norm](https://en.wikipedia.org/wiki/Lp_space#The_p-norm_in_finite_dimensions) is defined by <p align="center"><img src="./img/documentation/eq40.svg"></p>
With _p = 2_ this is the usual Euclidean norm.
The errors are calculated for each pair of columns of `X` and `Y`.
If `normalize=True`, then the _normalized absolute error_ is computed instead of the relative error: <p align="center"><img src="./img/documentation/eq41.svg"></p>

**`post.Lp_error(X, Y, t=None, p=2)`**: Approximate the absolute and relative _L_<sup>_p_</sup>-norm errors between snapshot sets `X` and `Y` corresponding to times `t`, assuming `Y` is an approximation to `X`.
The [_L_<sup>_p_</sup> norm](https://en.wikipedia.org/wiki/Lp_space#Lp_spaces) for vector-valued functions is defined by <p align="center"><img src="./img/documentation/eq42.svg"></p>
For finite _p_, the integrals are approximated by the trapezoidal rule: <p align="center"><img src="./img/documentation/eq43.svg"></p>

The `t` argument can be omitted if _p_ is infinity (`p = np.inf`).


## Utility Functions

These functions are helper routines for matricized higher-order tensor operations.

**`utils.kron2c(x)`**: Compute the compact, quadratic, column-wise (Khatri-Rao) Kronecker product of `x` with itself.

**`utils.kron3c(x)`**: Compute the compact, cubic, column-wise (Khatri-Rao) Kronecker product of `x` with itself three times.

**`utils.compress_H(H)`**: Convert the full _r_ x _r_<sup>2</sup> matricized quadratic operator `H` to its compact _r_ x (_r_(_r_+1)/2) form.

**`utils.expand_H(H)`**: Convert the compact _r_ x (_r_(_r_+1)/2) matricized quadratic operator `H` to the full, symmetric, _r_ x _r_<sup>2</sup> form.

**`utils.compress_G(G)`**: Convert the full _r_ x _r_<sup>3</sup> matricized cubic operator `G` to its compact _r_ x (_r_(_r_+1)(_r_+2)/6) form.

**`utils.expand_G(G)`**: Convert the compact _r_ x (_r_(_r_+1)(_r_+2)/6) matricized cubic operator `G` to the full, symmetric, _r_ x _r_<sup>3</sup> form.
