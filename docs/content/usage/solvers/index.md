(sec-lstsq)=
# Least-Squares Solvers

:::{warning}
This page is under construction.
:::

Options for solving the least-squares regression problem at the heart of Operator Inference, i.e.,

$$
\min_{\widehat{\mathbf{O}}}\left\|
    \mathbf{D}\widehat{\mathbf{O}}^{\mathsf{T}} - \mathbf{R}^{\mathsf{T}}
\right\|_{F}^{2} + \mathcal{P}(\widehat{\mathbf{O}}),
$$

where $\mathbf{D}$ is the _data matrix_ containing projected state snapshot data, $\widehat{\mathbf{O}}$ is the _operator matrix_ of unknown operators to be inferred, and $\mathcal{P}$ represents a regularization on the unknowns.
The structure of $\mathbf{D}$ and $\mathbf{O}$ depend on the problem structure, and the definition of $\mathbf{R}$ depends on the temporal context ([continuous time](sec-continuous) or [discrete time](sec-discrete)).

Accessible from `fit()` method...TODO

```{tableofcontents}
```
