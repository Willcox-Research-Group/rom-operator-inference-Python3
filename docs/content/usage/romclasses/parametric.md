(subsec-parametric-roms)=
# Parametric ROMs

The `ContinuousOpInfROM` and `DiscreteOpInfROM` classes are _non-parametric_ ROMs.
A _parametric_ ROM is one that depends on one or more external parameters $\mu\in\mathbb{R}^{p}$, meaning the operators themselves may depend on the external parameters.
This is different from the ROM depending on external inputs $\mathbf{u}$ that are provided at prediction time; by "parametric ROM" we mean the _operators_ of the ROM depend on $\mu$.
For example, a linear time-continuous parametric ROM has the form

$$
\frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t;\mu)
= \widehat{\mathbf{A}}(\mu)\widehat{\mathbf{q}}(t;\mu).
$$

## Additional Attributes

Parametric ROM classes have the following additional attributes.

| Attribute | Description |
| :-------- | :---------- |
| `p` | Dimension of the parameter $\mu$. |
| `s` | Number of training parameter samples. |

## Parametric Operators

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
>>> import opinf

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
opinf.core.operators._nonparametric.ConstantOperator

>>> c_static_.evaluate()
array([0.89308692, 0.81232528, 0.52454941])
```

Parametric operator evaluation is taken care of under the hood during parametric ROM evaluation.

## Parametric ROM Evaluation

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
