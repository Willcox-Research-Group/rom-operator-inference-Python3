(sec-romclasses)=
# ROM Classes

There are several reduced-order model (ROM) classes defined in the main namespace of `opinf`.
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
