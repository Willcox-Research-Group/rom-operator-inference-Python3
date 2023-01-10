(sec-continuous)=
# Continuous-time ROMs

A continuous-time ROM is a surrogate for a system of ordinary differential equations, written generally as

$$
\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t;\mu)
= \widehat{\mathbf{F}}(t, \widehat{\mathbf{q}}(t;\mu), \mathbf{u}(t); \mu).
$$

The following ROM classes target the continuous-time setting.
- `ContinuousOpInfROM` (nonparametric)
- `InterpolatedContinuousOpInfROM` (parametric via interpolation)

## Time Derivative Data

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

## ROM Evaluation

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


## Time Integration

The `predict()` method of `ContinuousOpInfROM` wraps [`scpiy.integrate.solve_ivp()`](https://docs.scipy.org/doc/scipy/reference/integrate.html) to solve the reduced-order model over a given time domain.

```python
predict(self, state0, t, input_func=None, reconstruct=True, **options)
```

| Argument | Type | Description |
| :------- | :--- | :---------- |
| `state0` | `(n,) or (r,) ndarray` | Initial state vector $\mathbf{q}(0)\in\mathbb{R}^{n}$ or $\widehat{\mathbf{q}}(0)\in\mathbb{R}^{r}$ |
| `t` | `(nt,) ndarray` | Time domain over which to integrate the ROM |
| `input_func` | `callable` | Mapping $t \mapsto \mathbf{u}(t)$ |
| `reconstruct` | `bool` | If True and the `basis` is not `None`, decode the results to the $n$-dimensional state space |
| `**options` | | Additional arguments for `scipy.integrate.solve_ivp()` |

<!-- TODO: implement common solvers and document here. -->