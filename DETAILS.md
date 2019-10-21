# Summary of Mathematical Details

This document gives a short explanation of the mathematical details behind the package.
For a full treatment, see [\[1\]](https://www.sciencedirect.com/science/article/pii/S0045782516301104).
However, note that some notation has been altered for coding convenience and clarity.

**Contents**
- [**Problem Statement**](#problem-statement)
- [**Projection-based Model Reduction**](#projection-based-model-reduction)
- [**Operator Inference via Least Squares**](#operator-inference-via-least-squares)
- [**Index of Notation**](#index-of-notation)
- [**References**](#references)


## Problem Statement

Consider the (possibly nonlinear) system of _n_ ordinary differential equations with state variable **x**, input (control) variable **u**, and independent variable _t_:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\dot{\mathbf{x}}(t)=\mathbf{f}(t,\mathbf{x}(t),\mathbf{u}(t)),"/>
</p>

where

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\mathbf{x}:\mathbb{R}\to\mathbb{R}^n,\qquad\mathbf{u}:\mathbb{R}\to\mathbb{R}^m,\qquad\mathbf{f}:\mathbb{R}\times\mathbb{R}^n\times\mathbb{R}^m\to\mathbb{R}^n."/>
</p>

This system is called the _full-order model_ (FOM).
If _n_ is large (as it often is in applications), it is computationally expensive to numerically solve the FOM.
This package provides tools for constructing a _reduced-order model_ (ROM) that is linear or quadratic in the state **x**, possibly with a constant term **c**, and with optional control inputs **u**.
The procedure is data-driven, non-intrusive, and relatively inexpensive.
In the most general case, the code can construct and solve a system of the form

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}(t)=\hat{\mathbf{c}}+\hat{A}\hat{\mathbf{x}}(t)+\hat{H}(\hat{\mathbf{x}}\otimes\hat{\mathbf{x}})(t)+\hat{B}\mathbf{u}(t),"/>
</p>

<!-- <p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}(t)=\hat{A}\hat{\mathbf{x}}(t)+\hat{H}(\hat{\mathbf{x}}\otimes\hat{\mathbf{x}})(t)+\hat{B}\mathbf{u}(t)+\sum_{i=1}^m\hat{N}_{i}\hat{\mathbf{x}}(t)u_{i}(t)+\hat{\mathbf{c}},"/>
</p> -->

where now

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\hat{\mathbf{x}}:\mathbb{R}\to\mathbb{R}^r,\qquad\mathbf{u}:\mathbb{R}\to\mathbb{R}^m,\qquad\hat{\mathbf{c}}\in\mathbb{R}^r,\qquad%20r\ll%20n,"/>
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\hat{A}\in\mathbb{R}^{r\times%20r},\qquad\hat{H}\in\mathbb{R}^{r\times%20r^2},\qquad\hat{B}\in\mathbb{R}^{r\times%20m}."/>
</p>

<!-- <p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\hat{A}\in\mathbb{R}^{r\times%20r},\qquad\hat{H}\in\mathbb{R}^{r\times%20r^2},\qquad\hat{B}\in\mathbb{R}^{r\times%20m},\qquad\hat{N}_{i}\in\mathbb{R}^{r\times%20r}."/>
</p> -->

This reduced low-dimensional system approximates the original high-dimensional system, but it is much easier (faster) to solve because of its low dimension _r_ << _n_.

<!-- TODO: discrete setting -->

## Projection-based Model Reduction

Model reduction via projection occurs in three steps:
1. (**Data Collection**) Gather snapshot data, i.e., solutions to the full-order model (FOM) at various times / parameters.
2. (**Compression**) Compute a low-rank basis (which defines a low-dimensional linear subspace) that captures most of the behavior of the snapshots.
3. (**Projection**) Use the low-rank basis to construct a low-dimensional ODE (the ROM) that approximates the FOM.

<!-- These steps comprise what is called the _offline phase_ in the literature, since they can all be done before the resulting ROM is simulated. -->

This package focuses on step 3, constructing the ROM given the snapshot data and the low-rank basis from steps 1 and 2, respectively.

Let _X_ be the _n_ x _k_ matrix whose _k_ columns are each solutions to the FOM of length _n_ (step 1), and let _V_<sub>_r_</sub> be an orthonormal _n_ x _r_ matrix representation for an _r_-dimensional subspace (step 2).
For example, a common choice for _V_<sub>_r_</sub> is the POD Basis of rank _r_, the matrix comprised of the first _r_ singular vectors of _X_.
We call _X_ the _snapshot matrix_ and _V_<sub>_r_</sub> the _reduced basis matrix_.

The classical approach to the projection step is to make the Ansatz

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\mathbf{x}(t)\approx%20V_{r}\hat{\mathbf{x}}(t)."/>
</p>

Inserting this into the FOM and multiplying both sides by the transpose of _V_<sub>_r_</sub> yields

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}(t)=V_{r}^\mathsf{T}\mathbf{f}(t,V_{r}\hat{\mathbf{x}}(t),\mathbf{u}(t))=:\hat{\mathbf{f}}(t,\hat{\mathbf{x}}(t),\mathbf{u}(t))."/>
</p>

This new system is _r_-dimensional in the sense that

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\hat{\mathbf{f}}:\mathbb{R}\times\mathbb{R}^r\times\mathbb{R}^m\to\mathbb{R}^r."/>
</p>

If the FOM operator **f** is known and has a nice structure, this reduced system can be solved cheaply by precomputing any involved matrices and then applying a time-stepping scheme.
For example, if **f** is linear in **x** and there is no input **u**, then

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\mathbf{f}(t,\mathbf{x}(t))=A\mathbf{x}(t)\qquad\Longrightarrow\qquad\hat{\mathbf{f}}(t,\hat{\mathbf{x}}(t))=V_r^\mathsf{T}AV_r\hat{\mathbf{x}}(t)=\hat{A}\hat{\mathbf{x}},"/>
</p>

where

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\hat{A}:=V_r^\mathsf{T}AV_r\in\mathbb{R}^{r\times%20r}."/>
</p>

However, _this approach breaks down if the FOM operator **f** is unknown, uncertain, or highly nonlinear_.

## Operator Inference via Least Squares

Instead of directly computing the reduced operators, the operator inference framework takes a data-driven approach: assuming a specific structure of the ROM (linear, quadratic, etc.), solve for the involved operators that best fit the data.
For example, suppose that we seek a ROM of the form

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}(t)=\hat{\mathbf{c}}+\hat{A}\hat{\mathbf{x}}(t)+\hat{H}(\hat{\mathbf{x}}\otimes\hat{\mathbf{x}})(t)+\hat{B}\mathbf{u}(t)."/>
</p>

We have only the snapshot matrix _X_, the low-rank basis matrix _V_<sub>_r_</sub> (which was derived from _X_), the inputs _U_, and perhaps the snapshot velocities _X'_ (if not, these must be approximated).
Here the (_ij_)<sup>th</sup> entry of _U_ is the _i_<sup>th</sup> component of **u** at the time corresponding to the _j_<sup>th</sup> snapshot.
To solve for the linear operators on the right-hand side of the preceding equation, we project the snapshot data via the basis matrix,

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\hat{X}=V_{r}^\mathsf{T}X,\qquad\qquad\dot{\hat{X}}=V_{r}^\mathsf{T}\dot{X},"/>
</p>

then solve the least squares problem

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\begin{align*}\underset{\substack{\hat{\mathbf{c}}\in\mathbb{R}^{r},\,\hat{A}\in\mathbb{R}^{r\times%20r},\\\hat{H}\in\mathbb{R}^{r\times%20r^2},\,\hat{B}\in\mathbb{R}^{r\times%20m}}}{\text{min}}\,\Big\|\hat{\mathbf{c}}\mathbf{1}^\mathsf{T}+\hat{A}\hat{X}+\hat{H}\big(\hat{X}\otimes\hat{X}\big)+\hat{B}U-\dot{\hat{X}}&\Big\|_{F}^2\\=\underset{\substack{\hat{\mathbf{c}}\in\mathbb{R}^{r},\,\hat{A}\in\mathbb{R}^{r\times%20r},\\\hat{H}\in\mathbb{R}^{r\times%20r^2},\,\hat{B}\in\mathbb{R}^{r\times%20m}}}{\text{min}}\,\Big\|\mathbf{1}\hat{\mathbf{c}}^\mathsf{T}+\hat{X}^\mathsf{T}\hat{A}^\mathsf{T}+\big(\hat{X}\otimes\hat{X}\big)^\mathsf{T}\hat{H}^\mathsf{T}+U^\mathsf{T}\hat{B}^\mathsf{T}-\dot{\hat{X}}^\mathsf{T}&\Big\|_{F}^2\\=\min_{O^\mathsf{T}\in\mathbb{R}^{(1+r+r^2+m)\times%20r}}\Big\|DO^\mathsf{T}-R&\Big\|_F^2,\end{align*}"/>
</p>

where **1** is a _k_-vector of 1's and

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\begin{align*}D&=\left[\begin{array}{cccc}\mathbf{1}&\hat{X}^\mathsf{T}&(\hat{X}\otimes\hat{X})^\mathsf{T}&U^\mathsf{T}\end{array}\right]&&\text{(Data)},\\O&=\left[\begin{array}{cccc}\hat{\mathbf{c}}&\hat{A}&\hat{H}&\hat{B}\end{array}\right]&&\text{(Operators)},\\R&=\dot{\hat{X}}^\mathsf{T}&&\text{(Right-hand%20side)}.\end{align*}"/>
</p>

For our purposes, any ⊗ between matrices denotes a column-wise Kronecker product (also called the [Khatri-Rao product](https://en.wikipedia.org/wiki/Kronecker_product#Khatri%E2%80%93Rao_product)).
The minimization problem given above decouples into _r_ independent ordinary least-squares problems, one for each of the columns of _O<sup>T</sup>_:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\begin{align*}&\min_{O^\mathsf{T}}\sum_{j=1}^r||D\mathbf{o}_j-\mathbf{r}_j||_2^2,\\O^\mathsf{T}&=\begin{bmatrix}\mathbf{o}_1&\mathbf{o}_2&\cdots&\mathbf{o}_r\end{bmatrix},\\R&=\begin{bmatrix}\mathbf{r}_1&\mathbf{r}_2&\cdots&\mathbf{r}_r\end{bmatrix}.\end{align*}"/>
</p>

The entire routine is relatively inexpensive to solve.
The code also allows for a Tikhonov regularization matrix or list of matrices (the `P` keyword argument for `predict()` methods), in which case the problem being solved is

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\min_{O^\mathsf{T}}\sum_{j=1}^r||D\mathbf{o}_j-\mathbf{r}_j||_2^2+||P_j\mathbf{o}_j||_2^2."/>
</p>

It can be shown [\[1\]](https://www.sciencedirect.com/science/article/pii/S0045782516301104) that, under some idealized assumptions, these inferred operators converge to the operators computed by explicit projection.
The key idea, however, is that _the inferred operators can be cheaply computed without knowing the full-order model_.
This is very convenient in situations where the FOM is given by a "black box," such as a legacy code for complex fluid simulations.

#### The Discrete Case

The framework described above can also be used to construct reduced-order models for approximating _discrete_ dynamical systems.
For instance, consider the full-order model

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\mathbf{x}_{k+1}=\mathbf{c}+A\mathbf{x}_{k}+H(\mathbf{x}_{k}\otimes\mathbf{x}_{k})+B\mathbf{u}_{k}."/>
</p>

Instead of collecting snapshot velocities, we collect _k+1_ snapshots and let _X_ be the _n x k_ matrix whose columns are the first _k_ snapshots and _X'_ be the _n x k_ matrix whose columns are the last _k_ snapshots.
That is, the columns **x**<sub>_k_</sub> of _X_ and **x**<sub>_k_</sub>' satisfy

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\mathbf{x}'_{k}=\mathbf{x}_{k+1}."/>
</p>

Then we set up the same least squares problem as before, but now the right-hand side matrix is

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?R=(\hat{X}')^\mathsf{T}=(V_{r}^\mathsf{T}X')^\mathsf{T}=(X')^\mathsf{T}V_{r}."/>
</p>


#### Implementation Note: The Kronecker Product

The vector [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product) ⊗ introduces some redundancies.
For example, the product **x** ⊗ **x** contains both _x_<sub>1</sub>_x_<sub>2</sub> and _x_<sub>2</sub>_x_<sub>1</sub>.
To avoid these redundancies, we introduce a "compact" Kronecker product <img src="https://latex.codecogs.com/svg.latex?\widetilde{\otimes}" height=10/> which only computes the unique terms of the usual vector Kronecker product:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\mathbf{x}\,\widetilde{\otimes}\,\mathbf{x}=\left[\begin{array}{c}\mathbf{x}^{(1)}\\\vdots\\\mathbf{x}^{(n)}\end{array}\right]\in\mathbb{R}^{n(n+1)/2},\qquad\text{where}\qquad\mathbf{x}^{(i)}=x_{i}\left[\begin{array}{c}x_{1}\\\vdots\\x_{i}\end{array}\right]\in\mathbb{R}^{i}."/>
</p>

When the compact Kronecker product is used, we call the resulting operator _H<sub>c</sub>_ instead of _H_.
Thus, the reduced order model becomes

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}(t)=\hat{A}\hat{\mathbf{x}}(t)+\hat{H}_{c}(\hat{\mathbf{x}}\,\widetilde{\otimes}\,\hat{\mathbf{x}})(t)+\hat{B}\mathbf{u}(t)+\hat{\mathbf{c}},"/>
</p>

and the corresponding operator inference least squares problem is

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\underset{\substack{\hat{\mathbf{c}}\in\mathbb{R}^{r},\,\hat{A}\in\mathbb{R}^{r\times%20r},\\\hat{H}_{c}\in\mathbb{R}^{r\times(r(r+1)/1)},\,\hat{B}\in\mathbb{R}^{r\times%20m},}}{\text{min}}\,\Big\|\hat{X}^\mathsf{T}\hat{A}^\mathsf{T}+\big(\hat{X}\,\widetilde{\otimes}\,\hat{X}\big)^\mathsf{T}\hat{H}_{c}^\mathsf{T}+U^\mathsf{T}\hat{B}^\mathsf{T}+\mathbf{1}\hat{\mathbf{c}}^\mathsf{T}-\dot{\hat{X}}^\mathsf{T}\Big\|_{F}^2."/>
</p>

## Index of Notation

We generally denote scalars in lower case, vectors in bold lower case, matrices in upper case, and indicate low-dimensional quantities with a hat.
In the code, a low-dimensional quantity ends with an underscore, so that the model classes follow some principles from the [scikit-learn](https://scikit-learn.org/stable/index.html) [API](https://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects).

#### Dimensions

| Symbol | Code | Description |
| :----: | :--- | :---------- |
| <img src="https://latex.codecogs.com/svg.latex?n"/> | `n`  | Dimension of the full-order system (large) |
| <img src="https://latex.codecogs.com/svg.latex?r"/> | `r`  | Dimension of the reduced-order system (small) |
| <img src="https://latex.codecogs.com/svg.latex?m"/> | `m`  | Dimension of the input **u** |
| <img src="https://latex.codecogs.com/svg.latex?k"/> | `k`  | Number of state snapshots, i.e., the number of training points |
| <img src="https://latex.codecogs.com/svg.latex?s"/> | `s`  | Number of parameter samples for parametric training |
| <img src="https://latex.codecogs.com/svg.latex?n_t"/> | `nt`  | Number of time steps in a simulation |
| <img src="https://latex.codecogs.com/svg.latex?p"/> | `p` | Dimension of the parameter space |


<!-- | <img src="https://latex.codecogs.com/svg.latex?\ell"/> | `l` | Dimension of the output **y** | -->

<!-- | <img src="https://latex.codecogs.com/svg.latex?d"/> | `d` | Dimension of the spatial domain | -->

<!-- | <img src="https://latex.codecogs.com/svg.latex?\frac{n(n+1)}{2}"/> | `_n2`  | Number of unique quadratic reduced-state interactions, _n_(_n_+1)/2 | -->
<!-- | <img src="https://latex.codecogs.com/svg.latex?\frac{r(r+1)}{2}"/> | `_r2`  | Number of unique quadratic reduced-state interactions, _r_(_r_+1)/2 | -->


#### Vectors

<!-- \sigma_j\in\text{diag}(\Sigma) &= \textrm{singular value of }X\\
\boldsymbol{\mu}\in\mathcal{P} &= \text{system parameter}\\
\mathcal{P}\subset\mathbb{R}^{p} &= \text{parameter space}\\
\Omega\subset\mathbb{R}^{d} &= \text{spatial domain}\\
% \omega\in\Omega &= \text{spatial point (one dimension)}\\
\boldsymbol{\omega}\in\Omega &= \text{spatial point}\\
t\ge 0 &= \text{time}\\
\hat{} &= \textrm{reduced variable, e.g., }\hat{\mathbf{x}}\textrm{ or }\hat{A}\\
\dot{} = \frac{d}{dt} &= \text{time derivative} -->


| Symbol | Code | Size | Description |
| :----: | :--- | :--: | :---------- |
| <img src="https://latex.codecogs.com/svg.latex?\mathbf{x}"/> | `x` | <img src="https://latex.codecogs.com/svg.latex?n"/> | Full-order state vector |
| <img src="https://latex.codecogs.com/svg.latex?\hat{\mathbf{x}}"/> | `x_` | <img src="https://latex.codecogs.com/svg.latex?r"/> | Reduced-order state vector |
| <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}"/> | `xdot_` | <img src="https://latex.codecogs.com/svg.latex?r"/> | Reduced-order state velocity vector |
| <img src="https://latex.codecogs.com/svg.latex?\mathbf{x}_\text{ROM}"/> | `x_ROM` | <img src="https://latex.codecogs.com/svg.latex?n"/> | Approximation to **x** produced by ROM |
| <img src="https://latex.codecogs.com/svg.latex?\hat{\mathbf{c}}"/> | `c_` | <img src="https://latex.codecogs.com/svg.latex?m"/> | Learned constant term  |
| <img src="https://latex.codecogs.com/svg.latex?\mathbf{u}"/> | `u` | <img src="https://latex.codecogs.com/svg.latex?m"/> | Input vector  |
| <img src="https://latex.codecogs.com/svg.latex?\mathbf{f}"/> | `f(t,x)` | <img src="https://latex.codecogs.com/svg.latex?n"/>  | Full-order system operator |
| <img src="https://latex.codecogs.com/svg.latex?\hat{\mathbf{f}}"/> | `f_(t,x_)` | <img src="https://latex.codecogs.com/svg.latex?n"/>  | Reduced-order system operator |
| <img src="https://latex.codecogs.com/svg.latex?\mathbf{x}\otimes\mathbf{x}"/> | `np.kron(x,x)` | <img src="https://latex.codecogs.com/svg.latex?n^2"/> | Kronecker product of full state (quadratic terms) |
| <img src="https://latex.codecogs.com/svg.latex?\hat{\mathbf{x}}\otimes\hat{\mathbf{x}}"/> | `np.kron(x_,x_)` | <img src="https://latex.codecogs.com/svg.latex?r^2"/>  | Kronecker product of reduced state (quadratic terms) |
| <img src="https://latex.codecogs.com/svg.latex?\hat{\mathbf{x}}\,\widetilde{\otimes}\,\hat{\mathbf{x}}"/> | `kron_compact(x_)` | <img src="https://latex.codecogs.com/svg.latex?s"/>  | Compact Kronecker product of reduced state (quadratic terms) |
| <img src="https://latex.codecogs.com/svg.latex?\mathbf{v}_j"/> | `vj` | <img src="https://latex.codecogs.com/svg.latex?n"/> | _j_<sup>th</sup> subspace basis vector, i.e., column _j_ of _V_<sub>_r_</sub> |

<!-- | **y**  | `y`             | Output vector | -->
<!-- | **y_ROM**, **y~** | `y_ROM`      | Approximation to **y** produced by ROM | -->

#### Matrices

| Symbol | Code | Shape | Description |
| :----: | :--- | :---: | :---------- |
| <img src="https://latex.codecogs.com/svg.latex?X"/> | `X` | <img src="https://latex.codecogs.com/svg.latex?n\times%20k"/> | Snapshot matrix |
| <img src="https://latex.codecogs.com/svg.latex?\dot{X}"/> | `Xdot` | <img src="https://latex.codecogs.com/svg.latex?n\times%20k"/> | Snapshot velocity matrix |
| <img src="https://latex.codecogs.com/svg.latex?V_r"/> | `Vr` | <img src="https://latex.codecogs.com/svg.latex?n\times%20r"/> | low-rank basis of rank _r_ (usually the POD basis) |
| <img src="https://latex.codecogs.com/svg.latex?U"/> | `U` | <img src="https://latex.codecogs.com/svg.latex?m\times%20k"/> | Input matrix (inputs corresonding to the snapshots) |
| <img src="https://latex.codecogs.com/svg.latex?\hat{X}"/> | `X_` | <img src="https://latex.codecogs.com/svg.latex?r\times%20k"/> | Projected snapshot matrix |
| <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{X}}"/> | `Xdot_` | <img src="https://latex.codecogs.com/svg.latex?r\times%20k"/> | Projected snapshot velocity matrix |
| <img src="https://latex.codecogs.com/svg.latex?\hat{A}"/> | `A_` | <img src="https://latex.codecogs.com/svg.latex?r\times%20r"/> | Learned state matrix |
| <img src="https://latex.codecogs.com/svg.latex?\hat{H}"/> | `H_` | <img src="https://latex.codecogs.com/svg.latex?r\times%20r^2"/> | Learned matricized quadratic tensor |
| <img src="https://latex.codecogs.com/svg.latex?\hat{F}"/> | `Hc_` | <img src="https://latex.codecogs.com/svg.latex?r\times%20s"/> | Learned matricized quadratic tensor without redundancy (compact) |
| <img src="https://latex.codecogs.com/svg.latex?\hat{B}"/> | `B_` | <img src="https://latex.codecogs.com/svg.latex?r\times%20m"/> | Learned input matrix |

<!-- | <img src="https://latex.codecogs.com/svg.latex?\hat{N}_i"/> | `Ni_` | <img src="https://latex.codecogs.com/svg.latex?r\times%20r"/> | Bilinear state-input matrix for _i_th input | -->

<!-- | <img src="https://latex.codecogs.com/svg.latex?\hat{C}"/> | `C_` | <img src="https://latex.codecogs.com/svg.latex?q\times%20r"/> | Learned output matrix | -->

<!-- I_{a\times%20a}\in\mathbb{R}^{a\times a} | | identity matrix\\ -->
<!-- \Sigma \in \mathbb{R}^{\ell\times\ell} &= \text{diagonal singular value matrix}\\ -->

## References

- \[1\] Peherstorfer, B. and Willcox, K.
[Data-driven operator inference for non-intrusive projection-based model reduction.](https://www.sciencedirect.com/science/article/pii/S0045782516301104)
Computer Methods in Applied Mechanics and Engineering, 306:196-215, 2016.
([Download](https://cims.nyu.edu/~pehersto/preprints/Non-intrusive-model-reduction-Peherstorfer-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{Peherstorfer16DataDriven,
    title     = {Data-driven operator inference for nonintrusive projection-based model reduction},
    author    = {Peherstorfer, B. and Willcox, K.},
    journal   = {Computer Methods in Applied Mechanics and Engineering},
    volume    = {306},
    pages     = {196--215},
    year      = {2016},
    publisher = {Elsevier}
}</pre></details>

- \[2\] Qian, E., Kramer, B., Marques, A., and Willcox, K.
[Transform & Learn: A data-driven approach to nonlinear model reduction](https://arc.aiaa.org/doi/10.2514/6.2019-3707).
In the AIAA Aviation 2019 Forum, June 17-21, Dallas, TX. ([Download](https://www.dropbox.com/s/5znea6z1vntby3d/QKMW_aviation19.pdf?dl=0))<details><summary>BibTeX</summary><pre>
@inbook{QKMW2019aviation,
    author    = {Qian, E. and Kramer, B. and Marques, A. N. and Willcox, K. E.},
    title     = {Transform \&amp; Learn: A data-driven approach to nonlinear model reduction},
    booktitle = {AIAA Aviation 2019 Forum},
    doi       = {10.2514/6.2019-3707},
    URL       = {https://arc.aiaa.org/doi/abs/10.2514/6.2019-3707},
    eprint    = {https://arc.aiaa.org/doi/pdf/10.2514/6.2019-3707}
}</pre></details>

- \[3\] Swischuk, R. and Mainini, L. and Peherstorfer, B. and Willcox, K.
[Projection-based model reduction: Formulations for physics-based machine learning.](https://www.sciencedirect.com/science/article/pii/S0045793018304250)
Computers & Fluids 179:704-717, 2019.
([Download](https://kiwi.oden.utexas.edu/papers/Physics-based-machine-learning-swischuk-willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{swischuk2019projection,
  title    = {Projection-based model reduction: Formulations for physics-based machine learning},
  author   = {Swischuk, Renee and Mainini, Laura and Peherstorfer, Benjamin and Willcox, Karen},
  journal  = {Computers \& Fluids},
  volume   = {179},
  pages    = {704--717},
  year     = {2019},
  publisher={Elsevier}
}</pre></details>

- \[4\] Swischuk, R. Physics-based machine learning and data-driven reduced-order modeling, Master's thesis, Massachusetts Institute of Technology, 2019.
<!-- TODO: link -->

- \[5\] Swischuk, R. and Kramer, B. and Huang, C. and Willcox, K. [Learning physics-based reduced-order models for a single-injector combustion process](https://arxiv.org/abs/1908.03620). ArXiv preprint arXiv:1908.03620.
([Download](https://arxiv.org/pdf/1908.03620.pdf))<details><summary>BibTeX</summary><pre>
@article{swischuk2019learning,
  title={Learning physics-based reduced-order models for a single-injector combustion process},
  author={Swischuk, Renee and Kramer, Boris and Huang, Cheng and Willcox, Karen},
  journal={arXiv preprint arXiv:1908.03620},
  year={2019}
}</pre></details>

- \[6\] Peherstorfer, B. [Sampling low-dimensional Markovian dynamics for pre-asymptotically recovering reduced models from data with operator inference](https://arxiv.org/abs/1908.11233). ArXiv preprint arXiv:1908.11233.
([Download](https://arxiv.org/pdf/1908.11233.pdf))<details><summary>BibTeX</summary><pre>
@article{peherstorfer2019sampling,
  title={Sampling low-dimensional Markovian dynamics for pre-asymptotically recovering reduced models from data with operator inference},
  author={Peherstorfer, Benjamin},
  journal={arXiv preprint arXiv:1908.11233},
  year={2019}
}</pre></details>
