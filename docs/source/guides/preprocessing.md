(sec-preprocessing-guide)=
# Preprocessing Guide

## Introduction

Our goal is to learn an efficient computational surrogate for a dynamical system

$$
    \frac{\text{d}}{\text{d}t}\mathbf{q}(t)
    = \mathbf{F}(t, \mathbf{q}(t), \mathbf{u}(t)),
$$

which has high-dimensional state $\mathbf{q}(t) \in \mathbb{R}^{n}$.
To achieve a computational speedup, we introduce a low-dimensional approximation

$$
    \mathbf{q}(t)
    \approx \boldsymbol{\Gamma}(\widehat{\mathbf{q}}(t)),
$$ (eq-preproc-approx)

where $\widehat{\mathbf{q}}(t)\in\mathbb{R}^{r}$ and $r \ll n$.
Operator Inference learns a reduced-order model that determines the evolution of the latent coordinates $\widehat{\mathbf{q}}(t)$.
This chapter discusses choices for $\boldsymbol{\Gamma}$, the mapping that bridges the latent coordinates and the original state space.
We approach this in two stages.

**Data scaling.**
Raw dynamical systems data often need to be lightly preprocessed before use in Operator Inference.
Common preprocessing steps include
1. Variable transformations / lifting to induce a polynomial structure.
2. Centering or shifting to account for boundary conditions.
3. Scaling / nondimensionalizing the variables represented in the state.

**Data compression.**
Once the data is properly normalized, a dimensionality reduction technique compresses the data to a latent low-dimensional coordinate system.

Proper preprocessing can improve the dimensionality reduction, promote stability in the inference of the reduced-order operators, and increase the stability and accuracy of the resulting reduced-order model.

```{tableofcontents}
```
