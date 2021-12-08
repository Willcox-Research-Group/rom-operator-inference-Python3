(sec-lstsq)=
# Least-Squares Solvers

Options for solving the least-squares regression problem at the heart of Operator Inference, i.e.,

$$
\min_{\widehat{\mathbf{O}}}\left\|
    \mathbf{D}\widehat{\mathbf{O}}^{\mathsf{T}} - \mathbf{R}^{\mathsf{T}}
\right\|_{F}^{2},
$$

where $\mathbf{D}$ is the _data matrix_ containing projected state snapshot data and $\widehat{\mathbf{O}}$ is the _operator matrix_ of unknown operators to be inferred.
The structure of $\mathbf{D}$ and $\mathbf{O}$ depend on the problem structure, and the definition of $\mathbf{R}$ depends on the temporal context ([continuous time](sec-continuous), [discrete time](sec-discrete), or [steady state](sec-steady)).

Accessible from `fit()` method...TODO

```{tableofcontents}
```
