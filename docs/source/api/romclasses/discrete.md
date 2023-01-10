(sec-discrete)=
# Discrete-time ROMs

The OpInf framework can be used to construct reduced-order models for approximating _discrete_ dynamical systems, as may arise from discretizing PDEs in both space and time.
A discrete-time ROM is a surrogate for a system of difference equations, written generally as

$$
\widehat{\mathbf{q}}_{j+1}(\mu)
= \widehat{\mathbf{F}}(\widehat{\mathbf{q}}_{j}(\mu), \mathbf{u}_{j}; \mu).
$$

The following ROM classes target the discrete setting.
- `DiscreteOpInfROM` (nonparametric)
- `InterpolatedDiscreteOpInfROM` (parametric via interpolation)

## Iterated Training Data

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

## ROM Evaluation

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

## Solution Iteration

The `predict()` method of `DiscreteOpInfROM` iterates the system to solve the reduced-order model for a given number of steps.
Unlike the continuous-time case, there are no choices to make about what scheme to use to solve the problem: the solution iteration is explicitly described by the reduced-order model.

```python
predict(self, state0, niters, inputs=None, reconstruct=True)
```

| Argument | Type | Description |
| :------- | :--- | :---------- |
| `state0` | `(n,) or (r,) ndarray` | Initial state vector $\mathbf{q}_{0}\in\mathbb{R}^{n}$ or $\widehat{\mathbf{q}}_{0}\in\mathbb{R}^{r}$ |
| `niters` | `int` | Number of times to step the system forward |
| `inputs` | `(m, niters-1) ndarray` | Inputs $\mathbf{u}_{j}$ for the next `niters-1` time steps |
| `reconstruct` | `bool` | If True and the `basis` is not `None`, decode the results to the $n$-dimensional state space |

<!-- TODO: implement common solvers and document here. -->