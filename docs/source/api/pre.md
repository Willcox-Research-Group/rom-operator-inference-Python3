(sec-preprocessing)=
# Preprocessing (`opinf.pre`)

## Introduction

Our goal is to learn an efficient computational surrogate for a dynamical system

$$
    \frac{\text{d}}{\text{d}t}\mathbf{q}(t)
    = \mathbf{F}(t, \mathbf{q}(t), \mathbf{u}(t)),
$$

which has high-dimensional state $\mathbf{q}(t) \in \mathbb{R}^{n}$.
To achieve a computational speedup, we introduce a low-dimensional approximation

$$
    \mathbf{q}(t) \approx \boldsymbol{\Gamma}(\widehat{\mathbf{q}}(t)),
$$ (eq-preproc-approx)

where $\widehat{\mathbf{q}}(t)\in\mathbb{R}^{r}$ and $r \ll n$.
Operator Inference learns a reduced-order model that determines the evolution of the latent coordinates $\widehat{\mathbf{q}}(t)$.
This chapter discusses modeling choices for $\boldsymbol{\Gamma}$, the mapping that bridges the latent coordinates and the original state space.
We approach this in two stages.

**Data scaling.**
Raw dynamical systems data often need to be lightly preprocessed before use in Operator Inference.
Common preprocessing steps include
1. Variable transformations / lifting to induce a polynomial structure.
2. Centering or shifting to account for boundary conditions.
3. Scaling / nondimensionalizing the variables represented in the state.

**Data compression.**
Once the data is properly normalized, a dimensionality reduction technique encodes the data in a latent low-dimensional coordinate system.

Proper preprocessing can improve the dimensionality reduction, promote stability in the inference of the reduced-order operators, and increase the stability and accuracy of the resulting reduced-order model.

:::::{admonition} Imporant Terminology
:class: important
The mapping $\boldsymbol{\Gamma} : \mathbb{R}^{r} \to \mathbb{R}^{n}$ is called the _decoder_.
The corresponding _encoder_ $\boldsymbol{\Gamma}^{*} : \mathbb{R}^{n} \to \mathbb{R}^{r}$ is defined by the minimization

$$
    \boldsymbol{\Gamma}^{*}(\mathbf{q})
    = \underset{\widehat{\mathbf{q}}\in\mathbb{R}^{r}}{\textrm{arg min}}\left\|
        \mathbf{q} - \boldsymbol{\boldsymbol{\Gamma}}(\widehat{\mathbf{q}})
    \right\|.
$$

In other words, the encoding of state vector $\mathbf{q} \in \mathbb{R}^{n}$ is the low-dimensional vector $\widehat{\mathbf{q}}\in\mathbb{R}^{r}$ such that the approximation {eq}`eq-preproc-approx` is as exact as possible.
The mapping from $\mathbf{q}$ to its best approximation in terms of $\boldsymbol{\Gamma}$ is called _projection_, and the quantity

$$
    \left\|\mathbf{q} - \boldsymbol{\Gamma}(\boldsymbol{\Gamma}^{*}(\mathbf{q}))\right\|
$$

is called the _projection error_ of $\mathbf{q}$ induced by $\boldsymbol{\Gamma}$.

::::{grid}
:gutter: 3
:margin: 2 2 0 0

:::{grid-item-card}
`encode(state)`
^^^
$\mathbf{q} \mapsto \boldsymbol{\Gamma}^{*}(\mathbf{q})$
:::

:::{grid-item-card}
`decode(state_)`
^^^
$\widehat{\mathbf{q}} \to \boldsymbol{\Gamma}(\widehat{\mathbf{q}})$
:::

:::{grid-item-card}
`project(state)`
^^^
$\mathbf{q}\mapsto \boldsymbol{\Gamma}(\boldsymbol{\Gamma}^{*}(\mathbf{q}))$
:::
::::
:::::

:::{eval-rst}
.. currentmodule:: opinf.pre

Classes
=======
.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    LinearBasis
    LinearBasisMulti
    PODBasis
    PODBasisMulti
    SnapshotTransformer
    SnapshotTransformerMulti

Functions
=========
.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    cumulative_energy
    ddt
    ddt_nonuniform
    ddt_uniform
    pod_basis
    projection_error
    reproject_continuous
    reproject_discrete
    residual_energy
    scale
    shift
    svdval_decay
:::

```{tableofcontents}
```
