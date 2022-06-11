(sec-notation)=
# Index of Notation

In the documentation, we generally denote scalars in lower case, vectors in bold lower case, matrices in upper case, and indicate low-dimensional quantities with a hat.
In the code, low-dimensional quantities ends with an underscore (e.g., `state` is high-dimensional and `state_` is low-dimensional).

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
| $\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}$ | `np.kron(q_,q_)` | $r^2$  | Full quadratic Kronecker product of reduced state |
| $\widehat{\mathbf{q}}\,\widehat{\otimes}\,\widehat{\mathbf{q}}$ | `utils.kron2c(q_)` | $\frac{r(r+1)}{2}$ | Compact quadratic Kronecker product of reduced state |
| $\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}\otimes\widehat{\mathbf{q}}$ | `np.kron(q_,np.kron(q_,q_))` | $r^3$  | Full cubic Kronecker product of reduced state |
| $\widehat{\mathbf{q}}\,\widehat{\otimes}\,\widehat{\mathbf{q}}\widehat{\otimes}\,\widehat{\mathbf{q}}$ | `utils.kron3c(q_)` | $\frac{r(r+1)(r+2)}{6}$ | Compact cubic Kronecker product of reduced state |
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
| $\widehat{\mathbf{A}}$ | `A_` | $r \times r$ | Reduced-order linear state matrix |
| $\widehat{\mathbf{H}}$ | `H_` | $r \times \frac{r(r+1)}{2}$ | Compact reduced-order matricized quadratic state tensor |
| $\widehat{\mathbf{G}}$ | `G_` | $r \times \frac{r(r+1)(r+2)}{6}$ | Compact reduced-order matricized quadratic state tensor |
| $\widehat{\mathbf{B}}$ | `B_` | $r \times m$ | Reduced-order input matrix |

<!-- | $\widehat{\mathbf{C}}$ | `C_` | $\ell \times r$ | Reduced-order output matrix | -->
<!-- | $\widehat{\mathbf{N}}$ | `N_` | $r \times rm$ | Bilinear state-input matrix | -->
