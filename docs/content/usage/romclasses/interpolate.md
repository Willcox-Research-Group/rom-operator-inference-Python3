# Interpolatory ROMs

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

## Training Strategy

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

## Choose an Interpolator

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

## Training Data Organization

Interpolated ROM `fit()` methods accept the training data in the  following formats.
- The basis
- A list of training parameters $[\mu_{1},\ldots,\mu_{s}]$ for which we have data
- A list of states $[\mathbf{Q}(\mu_{1}),\ldots,\mathbf{Q}(\mu_{s})]$ corresponding to the training parameters
- A single regularization parameter or a list of $s$ regularization parameters

## ROM Evaluation

As with all parametric ROM classes, evaluation the ROM by calling the object on the specifies parameter, e.g., `rom(parameter).predict(...)`.