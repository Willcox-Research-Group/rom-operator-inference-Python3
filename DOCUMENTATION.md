# Documentation

This document contains code documentation for `rom_operator_inference` classes and functions.
[The code itself](rom_operator_inference/) is also internally documented and can be accessed on the fly with dynamic object introspection.

**Contents**
- [**ROM Classes**](#rom-classes)
    - [**InferredContinuousROM**](#inferredcontinuousrom)
    - [**InferredDiscreteROM**](#inferreddiscreterom)
    - [**InterpolatedInferredContinuousROM**](#interpolatedinferredcontinuousrom)
    - [**InterpolatedInferredDiscreteROM**](#interpolatedinferreddiscreterom)
    - [**IntrusiveContinuousROM**](#intrusivecontinuousrom)
    - [**IntrusiveDiscreteROM**](#intrusivediscreterom)
    - [**AffineIntrusiveContinuousROM**](#affineintrusivecontinuousrom)
    - [**AffineIntrusiveDiscreteROM**](#affineintrusivediscreterom)
- [**Preprocessing**](#preprocessing-tools)
- [**Postprocessing**](#postprocessing-tools)
- [**Utility Functions**](#utility-functions)
- [**Index of Notation**](#index-of-notation)
- [**References**](#references)

<!-- **TODO**: The complete documentation at _\<insert link to sphinx-generated readthedocs page\>_. -->
<!-- Here we include a short catalog of functions and their inputs. -->

## ROM Classes

The core of `rom_operator_inference` is highly object oriented and defines several `ROM` classes that serve as the workhorse of the package.
The API for these classes adopts some principles from the [scikit-learn](https://scikit-learn.org/stable/index.html) [API](https://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects): there are `fit()` and `predict()` methods, hyperparameters are set in the constructor, estimated attributes end with underscore, and so on.

Each class corresponds to a type of full-order model (continuous vs. discrete, non-parametric vs. parametric) and a strategy for constructing the ROM.
In the following table, only those with "Operator Inference" as the strategy are novel; the others are included in the package for comparison purposes.

| Class Name | Problem Statement | ROM Strategy |
| :--------- | :---------------: | :----------- |
| `InferredContinuousROM` | <img src="img/doc/cnp.svg"/> | Operator Inference |
| `InferredDiscreteROM` | <img src="img/doc/dnp.svg"/> | Operator Inference |
| `InterpolatedInferredContinuousROM` | <img src="img/doc/csp.svg"/> | Operator Inference |
| `InterpolatedInferredDiscreteROM` | <img src="img/doc/dsp.svg"/> | Operator Inference |
| `IntrusiveContinuousROM` | <img src="img/doc/cnp.svg"/> | Intrusive Projection |
| `IntrusiveDiscreteROM` | <img src="img/doc/dnp.svg"/> | Intrusive Projection |
| `AffineIntrusiveContinuousROM` | <img src="img/doc/cvp.svg"/> | Intrusive Projection |
| `AffineIntrusiveDiscreteROM` | <img src="img/doc/dvp.svg"/> | Intrusive Projection |

<!-- | `AffineInferredContinuousROM` | <img src="https://latex.codecogs.com/svg.latex?\frac{d}{dt}\mathbf{x}(t;\boldsymbol{\mu})=\mathbf{f}(t,\mathbf{x}(t),\mathbf{u}(t);\boldsymbol{\mu})"/> | Operator Inference | -->
<!-- | `AffineInferredDiscreteROM` | <img src="https://latex.codecogs.com/svg.latex?\mathbf{x}_{k+1}=\mathbf{f}(\mathbf{x}_{k},\mathbf{u}_{k};\boldsymbol{\mu})"/> | Operator Inference | -->

<!-- More classes will be added in the future. -->
The following function may be helpful for selecting an appropriate class.

`**select_model(time, rom_strategy, parametric=False)**`: select the appropriate ROM model class for the situation.
Parameters:
- `time`: The type of full-order model to be reduced, either `"continuous"` or `"discrete"`.
- `rom_strategy`: Whether to use Operator Inference (`"inferred"`) or intrusive projection (`"intrusive"`) to compute the operators of the reduced model.
- `parametric`: Whether or not the model depends on an external parameter, and how to handle the parametric dependence. Options:
    - `False` (default): the problem is nonparametric.
    - `"affine"`: one or more operators in the problem depends affinely on the parameter (see SECTION). Only valid for `rom_strategy="intrusive"`.
    - `"interpolated"`: construct individual models for each sample parameter and interpolate them for general parameter inputs. Only valid for rom_strategy="inferred", and only when the parameter is a scalar.

The return value is the class type for the situation, e.g., `InferredContinuousROM`.

### Constructor

All `ROM` classes are instantiated with a single argument, `modelform`, which is a string denoting the structure of the full-order operator **f**.
Each character in the string corresponds to a single term of the operator, given in the following table.

| Character | Name | Continuous Term | Discrete Term |
| :-------- | :--- | :-------------- | :------------ |
| `c` | Constant | <img src="img/doc/chat.svg"/> | <img src="img/doc/chat.svg"/> |
| `A` | Linear | <img src="img/doc/Ahatxhat(t).svg"> | <img src="img/doc/Ahatxhatk.svg"> |
| `H` | Quadratic | <img src="img/doc/Hhatkronx(t).svg"> | <img src="img/doc/Hhatkronxk.svg"> |
| `B` | Input | <img src="img/doc/Bhatu(t).svg"/> | <img src="img/doc/Bhatuk.svg"/> |

<!-- | `O` | **O**utput | <img src="https://latex.codecogs.com/svg.latex?\mathbf{y}(t)=\hat{C}\hat{\mathbf{x}}(t)"/> | <img src="https://latex.codecogs.com/svg.latex?\mathbf{y}_{k}=\hat{C}\hat{\mathbf{x}}_{k}"/> | -->

These are all input as a single string.
Examples:

| `modelform` | Continuous ROM Structure | Discrete ROM Structure |
| :---------- | :----------------------- | ---------------------- |
|  `"A"`   | <img src="img/doc/xhatdot=Ahatxhat(t).svg"/> | <img src="img/doc/xhatk+1=Ahatxhatk.svg"/>
|  `"cA"`  | <img src="img/doc/xhatdot=chat+Ahatxhat(t).svg"/> | <img src="img/doc/xhatk+1=chat+Ahatxhatk.svg"/>
|  `"HB"`  | <img src="img/doc/xhatdot=Hhatkronx(t)+Bhatu(t).svg"/> | <img src="img/doc/xhatk+1=Hhatkronxk+Bhatuk.svg"/>
|  `"cAHB"` | <img src="img/doc/rom_c.svg"/> | <img src="img/doc/rom_d.svg"/>

### Attributes

All `ROM` classes have the following attributes.

- Structure of model:
    - `modelform`: set in the [constructor](#constructor).
    - `has_constant`: boolean, whether or not there is a constant term _**c**_.
    - `has_linear`: boolean, whether or not there is a linear term _A**x**_.
    - `has_quadratic`: boolean, whether or not there is a quadratic term _H(**x**⊗**x**)_.
    - `has_inputs`: boolean, whether or not there is an input term _B**u**_.
    <!-- - `has_outputs`: boolean, whether or not there is an output _C**x**_. -->

- Dimensions:
    - `n`: The dimension of the original model
    - `r`: The dimension of the learned reduced-order model
    - `m`: The dimension of the input **u**, or `None` if `'B'` is not in `modelform`.
    <!-- - `l`: The dimension of the output **y**, or `None` if `has_outputs` is `False`. -->

- Reduced operators `c_`, `A_`, `H_`, `Hc_`, and `B_`: the [NumPy](https://numpy.org/) arrays corresponding to the learned parts of the reduced-order model.
Set to `None` if the operator is not included in the prescribed `modelform` (e.g., if `modelform="AH"`, then `c_` and `B_` are `None`).

- Reduced model function `f_`: the ROM function, defined by the reduced operators listed above.
This attribute is constructed in `fit()`.
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
The input argument `u` is **only** present if `B` is in `modelform`.
Therefore, the signature of `f_` is one of the following.
|                | Continuous   | Discrete   |
| -------------: | :----------- | ---------- |
| **Has inputs** | `f_(t,x_,u)` | `f_(x_,u)` |
| **No inputs**  | `f_(t,x_)`   | `f_(x_)`   |

### InferredContinuousROM

This class constructs a reduced-order model for the continuous, nonparametric system

<p align="center"><img src="img/doc/icrom.svg"/></p>

via Operator Inference [\[1\]](https://www.sciencedirect.com/science/article/pii/S0045782516301104).
That is, given snapshot data, a basis, and a form for a reduced model, it computes the reduced model operators by solving an ordinary least-squares problem (see [DETAILS.md](DETAILS.md)).

**`InferredContinuousROM.fit(Vr, X, Xdot, U=None, P=0)`**: Compute the operators of the reduced-order model that best fit the data.
- **Parameters**
    - `Vr`: The _n_ x _r_ basis for the linear reduced space on which the full-order model will be projected (for example, a POD basis matrix; see [`pre.pod_basis()`](#preprocessing-tools)). Each column is a basis vector. The column space of `Vr` should be a good approximation of the column space of the full-order snapshot matrix `X`.
    - `X`: An _n_ x _k_ snapshot matrix of solutions to the full-order model, or the _r_ x _k_ projected snapshot matrix _V_<sub>_r_</sub><sup>T</sup>_X_. Each column is one snapshot.
    - `Xdot`: The _n_ x _k_ snapshot velocity matrix for the full-order model, or the _r_ x _k_ projected snapshot velocity matrix. Each column is the velocity _d**x**/dt_ for the corresponding column of `X`. See the [`pre`](#preprocessing-tools) submodule for some simple derivative approximation tools.
    - `U`: The _m_ x _k_ input matrix (or a _k_-vector if _m_ = 1). Each column is the input vector for the corresponding column of `X`. Only required when `'B'` is in `modelform`.
    - `P`: Tikhonov regularization matrix for the least-squares problem; see [`utils.lstsq_reg()`](#utility-functions).
- **Returns**
    - The trained `InferredContinuousROM` object.

**`InferredContinuousROM.predict(x0, t, u=None, **options)`**: Simulate the learned reduced-order model with [`scipy.integrate.solve_ivp()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
- **Parameters**
    - `x0`: The initial state vector, either full order (_n_-vector) or projected to reduced order (_r_-vector).
    - `t`: The time domain, an _n_<sub>_t_</sub>-vector, over which to integrate the reduced-order model.
    - `u`: The input as a function of time, that is, a function mapping a `float` to an _m_-vector (or to a scalar if _m_ = 1). Alternatively, the _m_ x _n_<sub>_t_</sub> matrix (or _n_<sub>_t_</sub>-vector if _m_ = 1) where column _j_ is the input vector corresponding to time `t[j]`. In this case, _**u**_(_t_) is appriximated by a cubic spline interpolating the given inputs. This argument is only required if `'B'` is in `modelform`.
    - Other keyword arguments for [`scipy.integrate.solve_ivp()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
- **Returns**
    - `X_ROM`: The _n_ x _n_<sub>_t_</sub> matrix of approximate solution to the full-order system over `t`. Each column is one snapshot of the solution.


### InferredDiscreteROM

This class constructs a reduced-order model for the discrete, nonparametric system

<p align="center"><img src="img/doc/dnp.svg"/></p>

via Operator Inference.

**`InferredDiscreteROM.fit(Vr, X, U=None, P=0)`**: Compute the operators of the reduced-order model that best fit the data.
- **Parameters**
    - `Vr`: The _n_ x _r_ basis for the linear reduced space on which the full-order model will be projected. Each column is a basis vector. The column space of `Vr` should be a good approximation of the column space of the full-order snapshot matrix `X`.
    - `X`: An _n_ x _k_ snapshot matrix of solutions to the full-order model, or the _r_ x _k_ projected snapshot matrix _V_<sub>_r_</sub><sup>T</sup>_X_. Each column is one snapshot.
    - `U`: The _m_ x _k-1_ input matrix (or a (_k_-1)-vector if _m_ = 1). Each column is the input for the corresponding column of `X`. Only required when `'B'` is in `modelform`.
    - `P`: Tikhonov regularization matrix for the least-squares problem.
- **Returns**
    - The trained `InferredDiscreteROM` object.

**`InferredDiscreteROM.predict(x0, niters, U=None)`**: Step forward the learned ROM `niters` steps.
- **Parameters**
    - `x0`: The initial state vector, either full order (_n_-vector) or projected to reduced order (_r_-vector).
    - `niters`: The number of times to step the system forward.
    - `U`: The inputs for the next `niters`-1 time steps, as an _m_ x `niters`-1 matrix (or an (`niters`-1)-vector if _m_ = 1). This argument is only required if `'B'` is in `modelform`.
- **Returns**
    - `X_ROM`: The _n_ x `niters` matrix of approximate solutions to the full-order system, including the initial condition. Each column is one iteration of the solution.


### InterpolatedInferredContinuousROM

This class constructs a reduced-order model for the continuous, parametric system
<p align="center"><img src="img/doc/icsprom.svg"/></p>

via Operator Inference.
The strategy is to take snapshot data for several parameter samples and a global basis, compute a reduced model for each parameter sample via Operator Inference, then construct a general parametric model by interpolating the entries of the inferred operators [\[1\]](https://www.sciencedirect.com/science/article/pii/S0045782516301104).

**`InterpolatedInferredContinuousROM.fit(Vr, µs, Xs, Xdots, Us=None, P=0)`**: Compute the operators of the reduced-order model that best fit the data.
- **Parameters**
    - `Vr`: The (global) _n_ x _r_ basis for the linear reduced space on which the full-order model will be projected. Each column is a basis vector. The column space of `Vr` should be a good approximation of the column space of the full-order snapshot matrices `Xs`.
    - `µs`: The _s_ parameter values corresponding to the snapshot sets.
    - `Xs`: List of _s_ snapshot matrices, each _n_ x _k_ (full-order solutions) or _r_ x _k_ (projected solutions). The _i_th array `Xs[i]` corresponds to the _i_th parameter, `µs[i]`; each column each of array is one snapshot.
    - `Xdots`: List of _s_ snapshot velocity matrices, each _n_ x _k_ (full-order velocities) or _r_ x _k_ (projected velocities).  The _i_th array `Xdots[i]` corresponds to the _i_th parameter, `µs[i]`. The _j_th column of the _i_th array, `Xdots[i][:,j]`, is the velocity _d**x**/dt_ for the corresponding snapshot column `Xs[i][:,j]`.
    - `Us`: List of _s_ input matrices, each _m_ x _k_ (or a _k_-vector if _m_=1). The _i_th array `Us[i]` corresponds to the _i_th parameter, `µs[i]`. The _j_th column of the _i_th array, `Us[i][:,j]`, is the input for the corresponding snapshot `Xs[i][:,j]`. Only required when `'B'` is in `modelform`.
    - `P`: Tikhonov regularization matrix for the least-squares problem.
- **Returns**
    - The trained `InterpolatedInferredContinuousROM` object.

**`InterpolatedInferredContinuousROM.predict(µ, x0, t, u=None, **options)`**: Simulate the learned reduced-order model with [`scipy.integrate.solve_ivp()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
- **Parameters**
    - `µ`: The parameter value at which to simulate the ROM.
    - `x0`: The initial state vector, either full order (_n_-vector) or projected to reduced order (_r_-vector).
    - `t`: The time domain, an _n_<sub>_t_</sub>-vector, over which to integrate the reduced-order model.
    - `u`: The input as a function of time, that is, a function mapping a `float` to an _m_-vector (or to a scalar if _m_ = 1). Alternatively, the _m_ x _n_<sub>_t_</sub> matrix (or _n_<sub>_t_</sub>-vector if _m_ = 1) where column _j_ is the input vector corresponding to time `t[j]`. In this case, _**u**_(_t_) is appriximated by a cubic spline interpolating the given inputs. This argument is only required if `'B'` is in `modelform`.
    - Other keyword arguments for [`scipy.integrate.solve_ivp()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
- **Returns**
    - `X_ROM`: The _n_ x _n_<sub>_t_</sub> matrix of approximate solution to the full-order system over `t`. Each column is one snapshot of the solution.


### InterpolatedInferredDiscreteROM

This class constructs a reduced-order model for the continuous, parametric system
<p align="center"><img src="img/doc/idsprom.svg"/></p>

via Operator Inference.
The strategy is to take snapshot data for several parameter samples and a global basis, compute a reduced model for each parameter sample via Operator Inference, then construct a general parametric model by interpolating the entries of the inferred operators [\[1\]](https://www.sciencedirect.com/science/article/pii/S0045782516301104).

**`InterpolatedInferredDiscreteROM.fit(Vr, µs, Xs, Us=None, P=0)`**: Compute the operators of the reduced-order model that best fit the data.
- **Parameters**
    - `Vr`: The (global) _n_ x _r_ basis for the linear reduced space on which the full-order model will be projected. Each column is a basis vector. The column space of `Vr` should be a good approximation of the column space of the full-order snapshot matrices `Xs`.
    - `µs`: The _s_ parameter values corresponding to the snapshot sets.
    - `Xs`: List of _s_ snapshot matrices, each _n_ x _k_ (full-order solutions) or _r_ x _k_ (projected solutions). The _i_th array `Xs[i]` corresponds to the _i_th parameter, `µs[i]`; each column each of array is one snapshot.
    - `Us`: List of _s_ input matrices, each _m_ x _k_ (or a _k_-vector if _m_=1). The _i_th array `Us[i]` corresponds to the _i_th parameter, `µs[i]`. The _j_th column of the _i_th array, `Us[i][:,j]`, is the input for the corresponding snapshot `Xs[i][:,j]`. Only required when `'B'` is in `modelform`.
    - `P`: Tikhonov regularization matrix for the least-squares problem.
- **Returns**
    - The trained `InterpolatedInferredDiscreteROM` object.

**`InterpolatedInferredDiscreteROM.predict(µ, x0, niters, U=None)`**: Step forward the learned ROM `niters` steps.
- **Parameters**
    - `µ`: The parameter value at which to simulate the ROM.
    - `x0`: The initial state vector, either full order (_n_-vector) or projected to reduced order (_r_-vector).
    - `niters`: The number of times to step the system forward.
    - `U`: The inputs for the next `niters`-1 time steps, as an _m_ x `niters`-1 matrix (or an (`niters`-1)-vector if _m_ = 1). This argument is only required if `'B'` is in `modelform`.
- **Returns**
    - `X_ROM`: The _n_ x `niters` matrix of approximate solutions to the full-order system, including the initial condition. Each column is one iteration of the solution.


### IntrusiveContinuousROM

This class constructs a reduced-order model for the continuous, nonparametric system

<p align="center"><img src="img/doc/icrom.svg"/></p>

via intrusive projection, i.e.,

<p align="center"><img src="img/doc/intrusive_ops.svg"/></p>

The class requires the actual full-order operators (_**c**_, _A_, _H_, and/or _B_) that define **f**.

**`IntrusiveContinuousROM.fit(Vr, operators)`**: Compute the operators of the reduced-order model by projecting the operators of the full-order model.
- **Parameters**
    - `Vr`: The _n_ x _r_ basis for the linear reduced space on which the full-order operators will be projected.
    - `operators`: A dictionary mapping labels to the full-order operators that define **f**. The operators are indexed by the entries of `modelform`; for example, if `modelform="cHB"`, then `operators={'c':c, 'H':H, 'B':B}`.
- **Returns**
    - The trained `IntrusiveContinuousROM` object.

**`IntrusiveContinuousROM.predict(x0, t, u=None, **options)`**: Simulate the learned reduced-order model with [`scipy.integrate.solve_ivp()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
- **Parameters**
    - `x0`: The initial state vector, either full order (_n_-vector) or projected to reduced order (_r_-vector).
    - `t`: The time domain, an _n_<sub>_t_</sub>-vector, over which to integrate the reduced-order model.
    - `u`: The input as a function of time, that is, a function mapping a `float` to an _m_-vector (or to a scalar if _m_ = 1). Alternatively, the _m_ x _n_<sub>_t_</sub> matrix (or _n_<sub>_t_</sub>-vector if _m_ = 1) where column _j_ is the input vector corresponding to time `t[j]`. In this case, _**u**_(_t_) is appriximated by a cubic spline interpolating the given inputs. This argument is only required if `'B'` is in `modelform`.
    - Other keyword arguments for [`scipy.integrate.solve_ivp()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
- **Returns**
    - `X_ROM`: The _n_ x _n_<sub>_t_</sub> matrix of approximate solution to the full-order system over `t`. Each column is one snapshot of the solution.


### IntrusiveDiscreteROM

This class constructs a reduced-order model for the discrete, nonparametric system

<p align="center"><img src="img/doc/dnp.svg"/></p>

via intrusive projection, i.e.,

<p align="center"><img src="img/doc/intrusive_ops.svg"/></p>

The class requires the actual full-order operators (_**c**_, _A_, _H_, and/or _B_) that define **f**.

**`IntrusiveDiscreteROM.fit(Vr, operators)`**: Compute the operators of the reduced-order model by projecting the operators of the full-order model.
- **Parameters**
    - `Vr`: The _n_ x _r_ basis for the linear reduced space on which the full-order operators will be projected.
    - `operators`: A dictionary mapping labels to the full-order operators that define **f**. The operators are indexed by the entries of `modelform`; for example, if `modelform="cHB"`, then `operators={'c':c, 'H':H, 'B':B}`.
- **Returns**
    - The trained `IntrusiveDiscreteROM` object.

**`IntrusiveDiscreteROM.predict(x0, niters, U=None)`**: Step forward the learned ROM `niters` steps.
- **Parameters**
    - `x0`: The initial state vector, either full order (_n_-vector) or projected to reduced order (_r_-vector).
    - `niters`: The number of times to step the system forward.
    - `U`: The inputs for the next `niters`-1 time steps, as an _m_ x `niters`-1 matrix (or an (`niters`-1)-vector if _m_ = 1). This argument is only required if `'B'` is in `modelform`.
- **Returns**
    - `X_ROM`: The _n_ x `niters` matrix of approximate solutions to the full-order system, including the initial condition. Each column is one iteration of the solution.


### AffineIntrusiveContinuousROM

This class constructs a reduced-order model for the continuous, affinely parametric system

<p align="center"><img src="img/doc/icvprom.svg"/></p>

where the operators that define **f** may have an affine dependence on the parameter, for example,

<p align="center"><img src="img/doc/affine.svg"/></p>

The reduction is done via intrusive projection, i.e.,

<p align="center"><img src="img/doc/affine_intrusive.svg"/></p>

The class requires the actual full-order operators (_**c**_, _A_, _H_, and/or _B_) that define **f** _and_ the functions that define any affine parameter dependencies (i.e., the _θ_<sub>_j_</sub> functions).

**`AffineIntrusiveContinuousROM.fit(Vr, affines, operators)`**: Compute the operators of the reduced-order model by projecting the operators of the full-order model.
- **Parameters**
    - `Vr`: The _n_ x _r_ basis for the linear reduced space on which the full-order operators will be projected.
    - `affines` A dictionary mapping labels of the operators that depend affinely on the parameter to the list of functions that define that affine dependence. The keys are entries of `modelform`. For example, if the constant term has the affine structure _c_(_**µ**_) = _θ_<sub>1</sub>(_**µ**_)_c_<sub>1</sub> + _θ_<sub>2</sub>(_**µ**_)_c_<sub>2</sub> + _θ_<sub>3</sub>(_**µ**_)_c_<sub>3</sub>, then `'c' -> [θ1, θ2, θ3]`.
    - `operators`: A dictionary mapping labels to the full-order operators that define **f**. The keys are entries of `modelform`. Terms with affine structure should be given as a list of the component matrices. For example, suppose `modelform="cA"`. If _A_ has the affine structure _A_(_**µ**_) = _θ_<sub>1</sub>(_**µ**_)_A_<sub>1</sub> + _θ_<sub>2</sub>(_**µ**_)_A_<sub>2</sub>, then `'A' -> [A1, A2]`. If _**c**_ does not vary with the parameter, then `'c' -> c`, the complete full-order order.
- **Returns**:
    - The trained `AffineIntrusiveContinuousROM` object.

**`AffineIntrusiveContinuousROM.predict(µ, x0, t, u=None, **options)`**: Simulate the learned reduced-order model at the given parameter value with [`scipy.integrate.solve_ivp()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
- **Parameters**
    - `µ`: The parameter value at which to simulate the model.
    - `x0`: The initial state vector, either full order (_n_-vector) or projected to reduced order (_r_-vector).
    - `t`: The time domain, an _n_<sub>_t_</sub>-vector, over which to integrate the reduced-order model.
    - `u`: The input as a function of time, that is, a function mapping a `float` to an _m_-vector (or to a scalar if _m_ = 1). Alternatively, the _m_ x _n_<sub>_t_</sub> matrix (or _n_<sub>_t_</sub>-vector if _m_ = 1) where column _j_ is the input vector corresponding to time `t[j]`. In this case, _**u**_(_t_) is appriximated by a cubic spline interpolating the given inputs. This argument is only required if `'B'` is in `modelform`.
    - Other keyword arguments for [`scipy.integrate.solve_ivp()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
- **Returns**
    - `X_ROM`: The _n_ x _n_<sub>_t_</sub> matrix of approximate solution to the full-order system over `t`. Each column is one snapshot of the solution.


### AffineIntrusiveDiscreteROM

This class constructs a reduced-order model for the continuous, affinely parametric system

<p align="center"><img src="img/doc/idvprom.svg"/></p>

where the operators that define **f** may have an affine dependence on the parameter, for example,

<p align="center"><img src="img/doc/affine.svg"/></p>

The reduction is done via intrusive projection, i.e.,

<p align="center"><img src="img/doc/affine_intrusive.svg"/></p>

The class requires the actual full-order operators (_**c**_, _A_, _H_, and/or _B_) that define **f** _and_ the functions that define any affine parameter dependencies (i.e., the _θ_<sub>_j_</sub> functions).

**`AffineIntrusiveDiscreteROM.fit(Vr, affines, operators)`**: Compute the operators of the reduced-order model by projecting the operators of the full-order model.
- **Parameters**
    - `Vr`: The _n_ x _r_ basis for the linear reduced space on which the full-order operators will be projected.
    - `affines` A dictionary mapping labels of the operators that depend affinely on the parameter to the list of functions that define that affine dependence. The keys are entries of `modelform`. For example, if the constant term has the affine structure _c_(_**µ**_) = _θ_<sub>1</sub>(_**µ**_)_c_<sub>1</sub> + _θ_<sub>2</sub>(_**µ**_)_c_<sub>2</sub> + _θ_<sub>3</sub>(_**µ**_)_c_<sub>3</sub>, then `'c' -> [θ1, θ2, θ3]`.
    - `operators`: A dictionary mapping labels to the full-order operators that define **f**. The keys are entries of `modelform`. Terms with affine structure should be given as a list of the component matrices. For example, suppose `modelform="cA"`. If _A_ has the affine structure _A_(_**µ**_) = _θ_<sub>1</sub>(_**µ**_)_A_<sub>1</sub> + _θ_<sub>2</sub>(_**µ**_)_A_<sub>2</sub>, then `'A' -> [A1, A2]`. If _**c**_ does not vary with the parameter, then `'c' -> c`, the complete full-order order.
- **Returns**:
    - The trained `AffineIntrusiveDiscreteROM` object.

**`AffineIntrusiveDiscreteROM.predict(µ, x0, niters, U=None)`**: Step forward the learned ROM `niters` steps at the given parameter value.
- **Parameters**
    - `µ`: The parameter value at which to simulate the model.
    - `x0`: The initial state vector, either full order (_n_-vector) or projected to reduced order (_r_-vector).
    - `niters`: The number of times to step the system forward.
    - `U`: The inputs for the next `niters`-1 time steps, as an _m_ x `niters`-1 matrix (or an (`niters`-1)-vector if _m_ = 1). This argument is only required if `'B'` is in `modelform`.
- **Returns**
    - `X_ROM`: The _n_ x `niters` matrix of approximate solutions to the full-order system, including the initial condition. Each column is one iteration of the solution.


## Preprocessing Tools

The `pre` submodule is a collection of common routines for preparing data to be used by the `ROM` classes.
None of these routines are novel, but they may be instructive for new Python users.

**`pre.mean_shift(X)`**: Compute the mean of the columns of `X` and shift `X` by that mean so that the result has mean column of zero.

**`pre.pod_basis(X, r, mode="arpack", **options)`**: Compute the POD basis of rank `r` for a snapshot matrix `X`.

**`pre.significant_svdvals(X, eps, plot=False)`**: Count the number of singular values of `X` that are greater than `eps`.

**`pre.energy_capture(X, thresh, plot=False)`**: Compute the number of singular values of `X` needed to surpass the energy threshold `thresh`; the energy of the first _j_ singular values is defined by <p align="center"><img src="img/doc/energy.svg"/></p>

**`pre.projection_error(X, Vr)`**: Compute the relative projection error on _X_ induced by the basis matrix _V<sub>r</sub>_, <p align="center"><img src="img/doc/proj_err.svg"/></p>

**`pre.minimal_projection_error(X, eps, rmax=_np.inf, plot=False, **options)`**: Compute the number of POD basis vectors required to obtain a projection
error less than `eps`, capped at `rmax`.

**`pre.xdot_uniform(X, dt, order=2)`**: Approximate the first derivative of a snapshot matrix `X` in which the snapshots are evenly spaced in time.

**`pre.xdot_nonuniform(X, t)`**: Approximate the first derivative of a snapshot matrix `X` in which the snapshots are **not** evenly spaced in time.

**`pre.xdot(X, *args, **kwargs)`**: Call `pre.xdot_uniform()` or `pre.xdot_nonuniform()`, depending on the arguments.


## Postprocessing Tools

The `post` submodule is a collection of common routines for computing the absolute and relative errors produced by a ROM approximation.
Given a norm, "true" data _X_, and an approximation _Y_ to _X_, these errors are defined by <p align="center"><img src="img/doc/errors.svg"/></p>

**`post.frobenius_error(X, Y)`**: Compute the absolute and relative Frobenius-norm errors between snapshot sets `X` and `Y`, assuming `Y` is an approximation to `X`.
The [Frobenius matrix norm](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm) is defined by <p align="center"><img src="img/doc/frobenius.svg"></p>

**`post.lp_error(X, Y, p=2, normalize=False)`**: Compute the absolute and relative _l_<sup>_p_</sup>-norm errors between snapshot sets `X` and `Y`, assuming `Y` is an approximation to `X`.
The [_l_<sup>_p_</sup> norm](https://en.wikipedia.org/wiki/Lp_space#The_p-norm_in_finite_dimensions) is defined by <p align="center"><img src="img/doc/vlpnorm.svg"/></p>
With _p=2_ this is the usual Euclidean norm.
The errors are calculated for each pair of columns of `X` and `Y`.
If `normalize=True`, then the _normalized absolute error_ is computed instead of the relative error: <p align="center"><img src="img/doc/normabserr.svg"></p>

**`post.Lp_error(X, Y, t=None, p=2)`**: Approximate the absolute and relative _L_<sup>_p_</sup>-norm errors between snapshot sets `X` and `Y` corresponding to times `t`, assuming `Y` is an approximation to `X`.
The [_L_<sup>_p_</sup> norm](https://en.wikipedia.org/wiki/Lp_space#Lp_spaces) for vector-valued functions is defined by <p align="center"><img src="img/doc/fLpnorm.svg"/></p>
For finite _p_, the integrals are approximated by the trapezoidal rule: <p align="center"><img src="img/doc/trap.svg"/></p>
The `t` argument can be omitted if _p_ is infinity (`p = np.inf`).


## Utility Functions

These functions are helper routines that are used internally for `fit()` or `predict()` methods.
See [DETAILS.md](DETAILS.md) for more mathematical explanation.

**`utils.lstsq_reg(A, b, P=0)`**: Solve the Tikhonov-regularized ordinary least-squares problem
<p align="center"><img src="img/doc/reg.svg"/></p>

  where _P_ is the regularization matrix. If `b` is a matrix, solve the above problem for each column of `b`. If `P` is a scalar, use the identity matrix times that scalar for the regularization matrix _P_.

**`utils.kron_compact(x)`**: Compute the compact column-wise (Khatri-Rao) Kronecker product of `x` with itself.

**`utils.kron_col(x, y)`**: Compute the full column-wise (Khatri-Rao) Kronecker product of `x` and `y`.

**`utils.compress_H(H)`**: Convert the full matricized quadratic operator `H` to the compact matricized quadratic operator `Hc`.

**`utils.expand_Hc(Hc)`**: Convert the compact matricized quadratic operator `Hc` to the full, symmetric, matricized quadratic operator `H`.


## Index of Notation

We generally denote scalars in lower case, vectors in bold lower case, matrices in upper case, and indicate low-dimensional quantities with a hat.
In the code, a low-dimensional quantity ends with an underscore, so that the model classes follow some principles from the [scikit-learn](https://scikit-learn.org/stable/index.html) [API](https://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects).

### Dimensions

| Symbol | Code | Description |
| :----: | :--- | :---------- |
| <img src="img/ntn/n.svg"/> | `n`  | Dimension of the full-order system (large) |
| <img src="img/ntn/r.svg"/> | `r`  | Dimension of the reduced-order system (small) |
| <img src="img/ntn/m.svg"/> | `m`  | Dimension of the input **u** |
| <img src="img/ntn/k.svg"/> | `k`  | Number of state snapshots, i.e., the number of training points |
| <img src="img/ntn/s.svg"/> | `s`  | Number of parameter samples for parametric training |
| <img src="img/ntn/p.svg"/> | `p` | Dimension of the parameter space |
| <img src="img/ntn/d.svg"/> | `d` | Number of columns of the data matrix _D_ |
| <img src="img/ntn/nt.svg"/> | `nt`  | Number of time steps in a simulation |

<!-- | <img src="https://latex.codecogs.com/svg.latex?\ell"/> | `l` | Dimension of the output **y** | -->


### Vectors

<!-- \sigma_j\in\text{diag}(\Sigma) &= \textrm{singular value of }X\\
\boldsymbol{\mu}\in\mathcal{P} &= \text{system parameter}\\
\mathcal{P}\subset\mathbb{R}^{p} &= \text{parameter space}\\
\Omega\subset\mathbb{R}^{d} &= \text{spatial domain}\\
% \omega\in\Omega &= \text{spatial point (one dimension)}\\
\boldsymbol{\omega}\in\Omega &= \text{spatial point}\\
t\ge 0 &= \text{time}\\
\hat{} &= \textrm{reduced variable, e.g., }\hat{\mathbf{x}}\textrm{ or }\hat{A}\\
\dot{} = \frac{d}{dt} &= \text{time derivative} -->


| Symbol | Code | Size | Description |
| :----: | :--- | :--: | :---------- |
| <img src="img/ntn/x.svg"/> | `x` | <img src="img/ntn/n.svg"/> | Full-order state vector |
| <img src="img/ntn/xhat.svg"/> | `x_` | <img src="img/ntn/r.svg"/> | Reduced-order state vector |
| <img src="img/ntn/xhatdot.svg"/> | `xdot_` | <img src="img/ntn/r.svg"/> | Reduced-order state velocity vector |
| <img src="img/ntn/xrom.svg"/> | `x_ROM` | <img src="img/ntn/n.svg"/> | Approximation to **x** produced by ROM |
| <img src="img/ntn/chat.svg"/> | `c_` | <img src="img/ntn/m.svg"/> | Learned constant term  |
| <img src="img/ntn/u.svg"/> | `u` | <img src="img/ntn/m.svg"/> | Input vector  |
| <img src="img/ntn/f.svg"/> | `f(t,x,u)` | <img src="img/ntn/n.svg"/>  | Full-order system operator |
| <img src="img/ntn/fhat.svg"/> | `f_(t,x_,u)` | <img src="img/ntn/n.svg"/>  | Reduced-order system operator |
| <img src="img/ntn/kronx.svg"/> | `np.kron(x,x)` | <img src="img/ntn/n2.svg"/> | Kronecker product of full state (quadratic terms) |
| <img src="img/ntn/kronxhat.svg"/> | `np.kron(x_,x_)` | <img src="img/ntn/r2.svg"/>  | Kronecker product of reduced state (quadratic terms) |
| <img src="img/ntn/kronxhatc.svg"/> | `kron_compact(x_)` | <img src="img/ntn/r2c.svg"/>  | Compact Kronecker product of reduced state (quadratic terms) |
| <img src="img/ntn/vj.svg"/> | `vj` | <img src="img/ntn/n.svg"/> | _j_<sup>th</sup> subspace basis vector, i.e., column _j_ of _V_<sub>_r_</sub> |

<!-- | **y**  | `y`             | Output vector | -->
<!-- | **y_ROM**, **y~** | `y_ROM`      | Approximation to **y** produced by ROM | -->

### Matrices

| Symbol | Code | Shape | Description |
| :----: | :--- | :---: | :---------- |
| <img src="img/ntn/Vr.svg"/> | `Vr` | <img src="img/ntn/nxr.svg"/> | low-rank basis of rank _r_ (usually the POD basis) |
| <img src="img/ntn/XX.svg"/> | `X` | <img src="img/ntn/nxk.svg"/> | Snapshot matrix |
| <img src="img/ntn/XXdot.svg"/> | `Xdot` | <img src="img/ntn/nxk.svg"/> | Snapshot velocity matrix |
| <img src="img/ntn/UU.svg"/> | `U` | <img src="img/ntn/mxk.svg"/> | Input matrix (inputs corresonding to the snapshots) |
| <img src="img/ntn/XXhat.svg"/> | `X_` | <img src="img/ntn/rxk.svg"/> | Projected snapshot matrix |
| <img src="img/ntn/XXhatdot.svg"/> | `Xdot_` | <img src="img/ntn/rxk.svg"/> | Projected snapshot velocity matrix |
| <img src="img/ntn/DD.svg"/> | `D` | <img src="img/ntn/kxd.svg"/> | Data matrix |
| <img src="img/ntn/OO.svg"/> | `O` | <img src="img/ntn/dxr.svg"/> | Operator matrix |
| <img src="img/ntn/RR.svg"/> | `R` | <img src="img/ntn/kxr.svg"/> | Right-hand side matrix |
| <img src="img/ntn/PP.svg"/> | `P` | <img src="img/ntn/dxd.svg"/> | Tikhonov regularization matrix |
| <img src="img/ntn/AAhat.svg"/> | `A_` | <img src="img/ntn/rxr.svg"/> | Learned state matrix |
| <img src="img/ntn/HHhat.svg"/> | `H_` | <img src="img/ntn/rxr2.svg"/> | Learned matricized quadratic tensor |
| <img src="img/ntn/HHhatc.svg"/> | `Hc_` | <img src="img/ntn/rxr2c.svg"/> | Learned matricized quadratic tensor without redundancy (compact) |
| <img src="img/ntn/BBhat.svg"/> | `B_` | <img src="img/ntn/rxm.svg"/> | Learned input matrix |

<!-- | <img src="https://latex.codecogs.com/svg.latex?\hat{N}_i"/> | `Ni_` | <img src="https://latex.codecogs.com/svg.latex?r\times%20r"/> | Bilinear state-input matrix for _i_th input | -->

<!-- | <img src="https://latex.codecogs.com/svg.latex?\hat{C}"/> | `C_` | <img src="https://latex.codecogs.com/svg.latex?q\times%20r"/> | Learned output matrix | -->

<!-- I_{a\times%20a}\in\mathbb{R}^{a\times a} | | identity matrix\\ -->
<!-- \Sigma \in \mathbb{R}^{\ell\times\ell} &= \text{diagonal singular value matrix}\\ -->


## References

- \[1\] Peherstorfer, B. and Willcox, K.,
[Data-driven operator inference for non-intrusive projection-based model reduction.](https://www.sciencedirect.com/science/article/pii/S0045782516301104)
_Computer Methods in Applied Mechanics and Engineering_, Vol. 306, pp. 196-215, 2016.
([Download](https://kiwi.oden.utexas.edu/papers/Non-intrusive-model-reduction-Peherstorfer-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{Peherstorfer16DataDriven,
    title     = {Data-driven operator inference for nonintrusive projection-based model reduction},
    author    = {Peherstorfer, B. and Willcox, K.},
    journal   = {Computer Methods in Applied Mechanics and Engineering},
    volume    = {306},
    pages     = {196--215},
    year      = {2016},
    publisher = {Elsevier}
}</pre></details>

- \[2\] Qian, E., Kramer, B., Marques, A., and Willcox, K.,
[Transform & Learn: A data-driven approach to nonlinear model reduction](https://arc.aiaa.org/doi/10.2514/6.2019-3707).
In the AIAA Aviation 2019 Forum & Exhibition, Dallas, TX, June 2019. ([Download](https://kiwi.oden.utexas.edu/papers/learn-data-driven-nonlinear-reduced-model-Qian-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@inbook{QKMW2019aviation,
    author    = {Qian, E. and Kramer, B. and Marques, A. N. and Willcox, K. E.},
    title     = {Transform \&; Learn: A data-driven approach to nonlinear model reduction},
    booktitle = {AIAA Aviation 2019 Forum},
    doi       = {10.2514/6.2019-3707},
    URL       = {https://arc.aiaa.org/doi/abs/10.2514/6.2019-3707},
    eprint    = {https://arc.aiaa.org/doi/pdf/10.2514/6.2019-3707}
}</pre></details>

- \[3\] Swischuk, R., Mainini, L., Peherstorfer, B., and Willcox, K.,
[Projection-based model reduction: Formulations for physics-based machine learning.](https://www.sciencedirect.com/science/article/pii/S0045793018304250)
_Computers & Fluids_, Vol. 179, pp. 704-717, 2019.
([Download](https://kiwi.oden.utexas.edu/papers/Physics-based-machine-learning-swischuk-willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{swischuk2019projection,
  title     = {Projection-based model reduction: Formulations for physics-based machine learning},
  author    = {Swischuk, Renee and Mainini, Laura and Peherstorfer, Benjamin and Willcox, Karen},
  journal   = {Computers \& Fluids},
  volume    = {179},
  pages     = {704--717},
  year      = {2019},
  publisher = {Elsevier}
}</pre></details>

- \[4\] Swischuk, R., Physics-based machine learning and data-driven reduced-order modeling. Master's thesis, Massachusetts Institute of Technology, 2019.
<!-- TODO: Link, BibTeX when published <details><summary>BibTeX</summary><pre>@article{CITATION}</pre></details> -->

- \[5\] Peherstorfer, B. [Sampling low-dimensional Markovian dynamics for pre-asymptotically recovering reduced models from data with operator inference](https://arxiv.org/abs/1908.11233). arXiv:1908.11233.
([Download](https://arxiv.org/pdf/1908.11233.pdf))<details><summary>BibTeX</summary><pre>
@article{peherstorfer2019sampling,
  title   = {Sampling low-dimensional Markovian dynamics for pre-asymptotically recovering reduced models from data with operator inference},
  author  = {Peherstorfer, Benjamin},
  journal = {arXiv preprint arXiv:1908.11233},
  year    = {2019}
}</pre></details>

- \[6\] Swischuk, R., Kramer, B., Huang, C., and Willcox, K., Learning physics-based reduced-order models for a single-injector combustion process. _AIAA Journal_, to appear, 2020. Also in Proceedings of 2020 AIAA SciTech Forum & Exhibition, Orlando FL, January, 2020. Also Oden Institute Report 19-13. ([Download](https://kiwi.oden.utexas.edu/papers/learning-reduced-model-combustion-Swischuk-Kramer-Huang-Willcox.pdf))
<!-- TODO: new BibTeX when published <details><summary>BibTeX</summary><pre>@article{swischuk2019learning, title={Learning physics-based reduced-order models for a single-injector combustion process}, author={Swischuk, Renee and Kramer, Boris and Huang, Cheng and Willcox, Karen}, journal={arXiv preprint arXiv:1908.03620}, year={2019}}</pre></details> -->

- \[7\] Qian, E., Kramer, B., Peherstorfer, B., and Willcox, K. Lift & Learn: Physics-informed machine learning for large-scale nonlinear dynamical systems. _Physica D: Nonlinear Phenomena_, to appear, 2020. ([Download](https://kiwi.oden.utexas.edu/papers/lift-learn-scientific-machine-learning-Qian-Willcox.pdf))
<!-- TODO: Link, BibTeX when published <details><summary>BibTeX</summary><pre>@article{CITATION}</pre></details> -->
