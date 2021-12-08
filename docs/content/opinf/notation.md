# Index of Notation

We generally denote scalars in lower case, vectors in bold lower case, matrices in upper case, and indicate low-dimensional quantities with a hat.
In the code, a low-dimensional quantity ends with an underscore, so that the model classes follow some principles from the [scikit-learn](https://scikit-learn.org/stable/index.html) [API](https://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects).

## Dimensions

| Symbol | Code | Description |
| :----: | :--- | :---------- |
| $n$ | `n` | Dimension of the full-order system (large) |
| $r$ | `r` | Dimension of the reduced-order system (small) |
| $m$ | `m` | Dimension of the input $\mathbf{u}$ |
| $k$ | `k` | Number of state snapshots, i.e., the number of training points |
| $s$ | `s` | Number of parameter samples for parametric training |
| $p$ | `p` | Dimension of the parameter space |
| $d$ | `d` | Number of columns of the data matrix $\mathbf{D}$ |

<!-- | <img src="./img/notation/eq04.svg"> | `l` | Dimension of the output **y** | -->

<!-- ### Scalars

| Symbol | Code | Description |
| :----: | :--- | :---------- |
| $n_{t}$ | `nt`  | Number of time steps in a simulation |
| $\mu$ | `µ` | Scalar parameter (_p_ = 1). | -->


## Vectors

| Symbol | Code | Size | Description |
| :----: | :--- | :--: | :---------- |
| $\mathbf{q}$ | `state` | $n$ | Full-order state vector |
| $\widehat{\mathbf{q}}$ | `state_` | $r$ | Reduced-order state vector |
| $\dot{\widehat{\mathbf{q}}}$ | `ddt_` | $r$ | Reduced-order state time derivative vector |
| $\mathbf{q}_{\text{ROM}}$ | `q_ROM` | $n$ | Approximation to $\mathbf{q}$ produced by ROM |
| $\widehat{\mathbf{c}}$ | `c_` | $r$ | Learned constant term  |
| $\mathbf{u}$ | `inputs` | $m$ | Input vector  |
| $\mathbf{f}$ | `f(t,x,u(t))` or `f(x,u)` | $n$ | Full-order system operator |
| $\widehat{\mathbf{f}}$ | `f_(t,x_,u(t))` or `f(x_,u)` | $r$  | Reduced-order system operator |
| $\mathbf{q}\otimes\mathbf{q}$ | `np.kron(q,q)` | $n^2$ | Quadratic Kronecker product of full state |
| $\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}$ | `np.kron(x_,x_)` | $r^2$  | Quadratic Kronecker product of reduced state |
| $\widehat{\mathbf{q}}\,\widehat{\otimes}\,\widehat{\mathbf{q}}$ | `utils.kron2c(x_)` | $\frac{r(r+1)}{2}$ | Compact quadratic Kronecker product of reduced state |
| $\mathbf{q}\otimes\mathbf{q}\otimes\mathbf{q}$ | `np.kron(q,np.kron(q,q))` | $n^3$ | Cubic Kronecker product of full state |
| $\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}$ | `np.kron(x_,np.kron(x_,x_))` | $r^3$  | Cubic Kronecker product of reduced state |
| $\widehat{\mathbf{q}}\,\widehat{\otimes}\,\widehat{\mathbf{q}}\widehat{\otimes}\,\widehat{\mathbf{q}}$ | `utils.kron3c(x_)` | $\frac{r(r+1)(r+2)}{6}$ | Compact cubic Kronecker product of reduced state |
| $\mathbf{v}_{j}$ | `vj` | $n$ | $j$th basis vector, i.e., column $j$ of $\mathbf{V}_{r}$ |

<!-- | **y**  | `y`             | Output vector | -->
<!-- | **y_ROM**, **y~** | `y_ROM`      | Approximation to **y** produced by ROM | -->

## Matrices

| Symbol | Code | Shape | Description |
| :----: | :--- | :---: | :---------- |
| $\mathbf{V}_{r}$ | `Vr` | $n \times r$ | low-rank basis of rank _r_ (usually the POD basis) |
| $\mathbf{Q}$ | `states` | $n \times k$ | Snapshot matrix |
| $\dot{\mathbf{Q}}$ | `ddts` | $n \times k$ | Snapshot time derivative matrix |
| $\mathbf{U}$ | `inputs` | $m \times k$ | Input matrix (inputs corresonding to the snapshots) |
| $\widehat{\mathbf{Q}}$ | `states_` | $r \times k$ | Projected snapshot matrix |
| $\dot{\widehat{\mathbf{Q}}}$ | `ddts_` | $r \times k$ | Projected snapshot time derivative matrix |
| $\mathbf{D}$ | `D` | $k \times d(r,m)$ | Data matrix |
| $\widehat{\mathbf{O}}$ | `Ohat` | $r \times d(r,m)$ | Operator matrix |
| $\mathbf{R}$ | `R` | $r \times k$ | Right-hand side matrix |
| $\boldsymbol{\Gamma}$ | `regularizer` | $d(r,m) \times d(r,m)$ | Tikhonov regularization matrix |
| $\mathbf{A}$ | `A` | $n \times n$ | Full-order linear state matrix |
| $\widehat{\mathbf{A}}$ | `A_` | $r \times r$ | Reduced-order linear state matrix |
| $\mathbf{H}$ | `H` | $n \times n^2$ | Full-order matricized quadratic state tensor |
| $\widehat{\mathbf{H}}$ | `H_` | $r \times \frac{r(r+1)}{2}$ | Compact reduced-order matricized quadratic state tensor |
| $\mathbf{G}$ | `G` | $r \times r^3$ | Full-order matricized quadratic state tensor |
| $\widehat{\mathbf{G}}$ | `G_` | $r \times \frac{r(r+1)(r+2)}{6}$ | Compact reduced-order matricized quadratic state tensor |
| $\mathbf{B}$ | `B` | $n \times m$ | Full-order input matrix |
| $\widehat{\mathbf{B}}$ | `B_` | $r \times m$ | Reduced-order input matrix |

<!-- | <img src="./img/notation/eq72.svg"> | `C` | <img src="./img/notation/eq73.svg"> | Full-order output matrix | -->
<!-- | <img src="./img/notation/eq74.svg"> | `C_` | <img src="./img/notation/eq75.svg"> | Reduced-order output matrix | -->
<!-- | <img src="\hat{N}_i"> | `Ni_` | <img src="./img/notation/eq55.svg"> | Bilinear state-input matrix for _i_th input | -->
