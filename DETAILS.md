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

<p align="center"><img src="img/prb/eq1.svg"/></p>

where

<p align="center"><img src="img/prb/eq2.svg"/></p>

This system is called the _full-order model_ (FOM).
If _n_ is large, as it often is in high-consequence engineering applications, it is computationally expensive to numerically solve the FOM.
This package provides tools for constructing a _reduced-order model_ (ROM) that is up to quadratic in the state **x** with optional linear control inputs **u**.
The procedure is data-driven, non-intrusive, and relatively inexpensive.
In the most general case, the code can construct and solve a reduced-order system with the polynomial form

<p align="center"><img src="img/prb/eq3.svg"/></p>

<!-- <p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\dot{\hat{\mathbf{x}}}(t)=\hat{A}\hat{\mathbf{x}}(t)+\hat{H}(\hat{\mathbf{x}}\otimes\hat{\mathbf{x}})(t)+\hat{B}\mathbf{u}(t)+\sum_{i=1}^m\hat{N}_{i}\hat{\mathbf{x}}(t)u_{i}(t)+\hat{\mathbf{c}},"/>
</p> -->

where now

<p align="center"><img src="img/prb/eq4.svg"/></p>
<p align="center"><img src="img/prb/eq5.svg"/></p>

<!-- <p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\hat{A}\in\mathbb{R}^{r\times%20r},\qquad\hat{H}\in\mathbb{R}^{r\times%20r^2},\qquad\hat{B}\in\mathbb{R}^{r\times%20m},\qquad\hat{N}_{i}\in\mathbb{R}^{r\times%20r}."/>
</p> -->

This reduced low-dimensional system approximates the original high-dimensional system, but it is much easier (faster) to solve because of its low dimension _r_ << _n_.


## Projection-based Model Reduction

Model reduction via projection occurs in three steps:
1. **Data Collection**: Gather snapshot data, i.e., solutions to the full-order model (the FOM) at various times / parameters.
2. **Compression**: Compute a low-rank basis (which defines a low-dimensional linear subspace) that captures most of the behavior of the snapshots.
3. **Projection**: Use the low-rank basis to construct a low-dimensional ODE (the ROM) that approximates the FOM.

<!-- These steps comprise what is called the _offline phase_ in the literature, since they can all be done before the resulting ROM is simulated. -->

This package focuses on step 3, constructing the ROM given the snapshot data and the low-rank basis from steps 1 and 2, respectively.

Let _X_ be the _n_ x _k_ matrix whose _k_ columns are each solutions to the FOM of length _n_ (step 1), and let _V_<sub>_r_</sub> be an orthonormal _n_ x _r_ matrix representation for an _r_-dimensional subspace (step 2).
For example, a common choice for _V_<sub>_r_</sub> is the POD Basis of rank _r_, the matrix comprised of the first _r_ singular vectors of _X_.
We call _X_ the _snapshot matrix_ and _V_<sub>_r_</sub> the _reduced basis matrix_.

The classical approach to the projection step is to make the Ansatz

<p align="center"><img src="img/dtl/eq01.svg"/></p>

Inserting this into the FOM and multiplying both sides by the transpose of _V_<sub>_r_</sub> yields

<p align="center"><img src="img/dtl/eq02.svg"/></p>

This new system is _r_-dimensional in the sense that

<p align="center"><img src="img/dtl/eq03.svg"/></p>

If the FOM operator **f** is known and has a nice structure, this reduced system can be solved cheaply by precomputing any involved matrices and then applying a time-stepping scheme.
For example, if **f** is linear in **x** and there is no input **u**, then

<p align="center"><img src="img/dtl/eq04.svg"/></p>

where

<p align="center"><img src="img/dtl/eq05.svg"/></p>

However, _this approach breaks down if the FOM operator **f** is unknown, uncertain, or highly nonlinear_.

## Operator Inference via Least Squares

Instead of directly computing the reduced operators, the Operator Inference framework takes a data-driven approach: assuming a specific structure of the ROM (linear, quadratic, etc.), solve for the involved operators that best fit the data.
For example, suppose that we seek a ROM of the form

<p align="center"><img src="img/dtl/eq06.svg"/></p>

We have only the snapshot matrix _X_, the low-rank basis matrix _V_<sub>_r_</sub> (which was derived from _X_), the inputs _U_, and perhaps the snapshot velocities _X'_ (if not, these must be approximated).
Here the (_ij_)<sup>th</sup> entry of _U_ is the _i_<sup>th</sup> component of **u** at the time corresponding to the _j_<sup>th</sup> snapshot.
To solve for the linear operators on the right-hand side of the preceding equation, we project the snapshot data via the basis matrix,

<p align="center"><img src="img/dtl/eq07.svg"/></p>

then solve the least squares problem

<p align="center"><img src="img/dtl/eq08.svg"/></p>

where **1** is a _k_-vector of 1's and

<p align="center"><img src="img/dtl/eq09.svg"/></p>

For our purposes, the ⊗ operator between matrices denotes a column-wise Kronecker product (also called the [Khatri-Rao product](https://en.wikipedia.org/wiki/Kronecker_product#Khatri%E2%80%93Rao_product)).
The minimization problem given above decouples into _r_ independent ordinary least-squares problems, one for each of the columns of _O<sup>T</sup>_:

<p align="center"><img src="img/dtl/eq10.svg"/></p>

The entire routine is relatively inexpensive to solve.
The code also allows for a Tikhonov regularization matrix or list of matrices (the `P` keyword argument for `predict()` methods), in which case the problem being solved is

<p align="center"><img src="img/dtl/eq11.svg"/></p>

It can be shown [\[1\]](https://www.sciencedirect.com/science/article/pii/S0045782516301104) that, under some idealized assumptions, these inferred operators converge to the operators computed by explicit projection.
The key idea, however, is that _the inferred operators can be cheaply computed without knowing the full-order model_.
This is very convenient in situations where the FOM is given by a "black box," such as a legacy code for complex fluid simulations.

#### The Discrete Case

The framework described above can also be used to construct reduced-order models for approximating _discrete_ dynamical systems.
For instance, consider the full-order model

<p align="center"><img src="img/dtl/eq12.svg"/></p>

Instead of collecting snapshot velocities, we collect _k+1_ snapshots and let _X_ be the _n x k_ matrix whose columns are the first _k_ snapshots and _X'_ be the _n x k_ matrix whose columns are the last _k_ snapshots.
That is, the columns **x**<sub>_k_</sub> of _X_ and **x**<sub>_k_</sub>' satisfy

<p align="center"><img src="img/dtl/eq13.svg"/></p>

Then we set up the same least squares problem as before, but now the right-hand side matrix is

<p align="center"><img src="img/dtl/eq14.svg"/></p>

The resulting reduced-order model has the form

<p align="center"><img src="img/dtl/eq15.svg"><p>


<!-- TODO: #### Re-projection and Recovering Intrusive Models -->


#### Implementation Note: The Kronecker Product

The vector [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product) ⊗ introduces some redundancies.
For example, the product **x** ⊗ **x** contains both _x_<sub>1</sub>_x_<sub>2</sub> and _x_<sub>2</sub>_x_<sub>1</sub>.
To avoid these redundancies, we introduce a "compact" Kronecker product <img src="img/dtl/eq16.svg" height=10/> which only computes the unique terms of the usual vector Kronecker product:

<p align="center"><img src="img/dtl/eq17.svg"/></p>

When the compact Kronecker product is used, we call the resulting operator _H<sub>c</sub>_ instead of _H_.
Thus, the reduced order model becomes

<p align="center"><img src="img/dtl/eq18.svg"/></p>

and the corresponding Operator Inference least squares problem is

<p align="center"><img src="img/dtl/eq19.svg"/></p>


## Index of Notation

We generally denote scalars in lower case, vectors in bold lower case, matrices in upper case, and indicate low-dimensional quantities with a hat.
In the code, a low-dimensional quantity ends with an underscore, so that the model classes follow some principles from the [scikit-learn](https://scikit-learn.org/stable/index.html) [API](https://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects).

### Dimensions

| Symbol | Code | Description |
| :----: | :--- | :---------- |
| <img src="img/ntn/n.svg"/> | `n`  | Dimension of the full-order system (large) |
| <img src="img/ntn/r.svg"/> | `r`  | Dimension of the reduced-order system (small) |
| <img src="img/ntn/m.svg"/> | `m`  | Dimension of the input **u** |
| <img src="img/ntn/k.svg"/> | `k`  | Number of state snapshots, i.e., the number of training points |
| <img src="img/ntn/s.svg"/> | `s`  | Number of parameter samples for parametric training |
| <img src="img/ntn/p.svg"/> | `p` | Dimension of the parameter space |
| <img src="img/ntn/d.svg"/> | `d` | Number of columns of the data matrix _D_ |
| <img src="img/ntn/nt.svg"/> | `nt`  | Number of time steps in a simulation |

<!-- | <img src="https://latex.codecogs.com/svg.latex?\ell"/> | `l` | Dimension of the output **y** | -->


### Vectors

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
| <img src="img/ntn/x.svg"/> | `x` | <img src="img/ntn/n.svg"/> | Full-order state vector |
| <img src="img/ntn/xhat.svg"/> | `x_` | <img src="img/ntn/r.svg"/> | Reduced-order state vector |
| <img src="img/ntn/xhatdot.svg"/> | `xdot_` | <img src="img/ntn/r.svg"/> | Reduced-order state velocity vector |
| <img src="img/ntn/xrom.svg"/> | `x_ROM` | <img src="img/ntn/n.svg"/> | Approximation to **x** produced by ROM |
| <img src="img/ntn/chat.svg"/> | `c_` | <img src="img/ntn/m.svg"/> | Learned constant term  |
| <img src="img/ntn/u.svg"/> | `u` | <img src="img/ntn/m.svg"/> | Input vector  |
| <img src="img/ntn/f.svg"/> | `f(t,x,u)` | <img src="img/ntn/n.svg"/>  | Full-order system operator |
| <img src="img/ntn/fhat.svg"/> | `f_(t,x_,u)` | <img src="img/ntn/n.svg"/>  | Reduced-order system operator |
| <img src="img/ntn/kronx.svg"/> | `np.kron(x,x)` | <img src="img/ntn/n2.svg"/> | Kronecker product of full state (quadratic terms) |
| <img src="img/ntn/kronxhat.svg"/> | `np.kron(x_,x_)` | <img src="img/ntn/r2.svg"/>  | Kronecker product of reduced state (quadratic terms) |
| <img src="img/ntn/kronxhatc.svg"/> | `kron2c(x_)` | <img src="img/ntn/r2c.svg"/>  | Compact Kronecker product of reduced state (quadratic terms) |
| <img src="img/ntn/vj.svg"/> | `vj` | <img src="img/ntn/n.svg"/> | _j_<sup>th</sup> subspace basis vector, i.e., column _j_ of _V_<sub>_r_</sub> |

<!-- | **y**  | `y`             | Output vector | -->
<!-- | **y_ROM**, **y~** | `y_ROM`      | Approximation to **y** produced by ROM | -->

### Matrices

| Symbol | Code | Shape | Description |
| :----: | :--- | :---: | :---------- |
| <img src="img/ntn/Vr.svg"/> | `Vr` | <img src="img/ntn/nxr.svg"/> | low-rank basis of rank _r_ (usually the POD basis) |
| <img src="img/ntn/XX.svg"/> | `X` | <img src="img/ntn/nxk.svg"/> | Snapshot matrix |
| <img src="img/ntn/XXdot.svg"/> | `Xdot` | <img src="img/ntn/nxk.svg"/> | Snapshot velocity matrix |
| <img src="img/ntn/UU.svg"/> | `U` | <img src="img/ntn/mxk.svg"/> | Input matrix (inputs corresonding to the snapshots) |
| <img src="img/ntn/XXhat.svg"/> | `X_` | <img src="img/ntn/rxk.svg"/> | Projected snapshot matrix |
| <img src="img/ntn/XXhatdot.svg"/> | `Xdot_` | <img src="img/ntn/rxk.svg"/> | Projected snapshot velocity matrix |
| <img src="img/ntn/DD.svg"/> | `D` | <img src="img/ntn/kxd.svg"/> | Data matrix |
| <img src="img/ntn/OO.svg"/> | `O` | <img src="img/ntn/dxr.svg"/> | Operator matrix |
| <img src="img/ntn/RR.svg"/> | `R` | <img src="img/ntn/kxr.svg"/> | Right-hand side matrix |
| <img src="img/ntn/PP.svg"/> | `P` | <img src="img/ntn/dxd.svg"/> | Tikhonov regularization matrix |
| <img src="img/ntn/AAhat.svg"/> | `A_` | <img src="img/ntn/rxr.svg"/> | Learned state matrix |
| <img src="img/ntn/HHhat.svg"/> | `H_` | <img src="img/ntn/rxr2.svg"/> | Learned matricized quadratic tensor |
| <img src="img/ntn/HHhatc.svg"/> | `Hc_` | <img src="img/ntn/rxr2c.svg"/> | Learned matricized quadratic tensor without redundancy (compact) |
| <img src="img/ntn/GGhat.svg"/> | `G_` | <img src="img/ntn/rxr3.svg"/> | Learned matricized cubic tensor |
| <img src="img/ntn/GGhatc.svg"/> | `Gc_` | <img src="img/ntn/rxr3c.svg"/> | Learned matricized cubic tensor without redundancy (compact) |
| <img src="img/ntn/BBhat.svg"/> | `B_` | <img src="img/ntn/rxm.svg"/> | Learned input matrix |

<!-- | <img src="https://latex.codecogs.com/svg.latex?\hat{N}_i"/> | `Ni_` | <img src="https://latex.codecogs.com/svg.latex?r\times%20r"/> | Bilinear state-input matrix for _i_th input | -->

<!-- | <img src="https://latex.codecogs.com/svg.latex?\hat{C}"/> | `C_` | <img src="https://latex.codecogs.com/svg.latex?q\times%20r"/> | Learned output matrix | -->

<!-- I_{a\times%20a}\in\mathbb{R}^{a\times a} | | identity matrix\\ -->
<!-- \Sigma \in \mathbb{R}^{\ell\times\ell} &= \text{diagonal singular value matrix}\\ -->


## References

- \[1\] Peherstorfer, B. and Willcox, K.,
[Data-driven operator inference for non-intrusive projection-based model reduction.](https://www.sciencedirect.com/science/article/pii/S0045782516301104)
_Computer Methods in Applied Mechanics and Engineering_, Vol. 306, pp. 196-215, 2016.
([Download](https://kiwi.oden.utexas.edu/papers/Non-intrusive-model-reduction-Peherstorfer-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{Peherstorfer16DataDriven,
    title     = {Data-driven operator inference for nonintrusive projection-based model reduction},
    author    = {Peherstorfer, B. and Willcox, K.},
    journal   = {Computer Methods in Applied Mechanics and Engineering},
    volume    = {306},
    pages     = {196--215},
    year      = {2016},
    publisher = {Elsevier}
}</pre></details>

- \[2\] Qian, E., Kramer, B., Marques, A., and Willcox, K.,
[Transform & Learn: A data-driven approach to nonlinear model reduction](https://arc.aiaa.org/doi/10.2514/6.2019-3707).
In the AIAA Aviation 2019 Forum & Exhibition, Dallas, TX, June 2019. ([Download](https://kiwi.oden.utexas.edu/papers/learn-data-driven-nonlinear-reduced-model-Qian-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@inbook{QKMW2019aviation,
    title     = {Transform \\& Learn: A data-driven approach to nonlinear model reduction},
    author    = {Qian, E. and Kramer, B. and Marques, A. N. and Willcox, K. E.},
    booktitle = {AIAA Aviation 2019 Forum},
    doi       = {10.2514/6.2019-3707},
    URL       = {https://arc.aiaa.org/doi/abs/10.2514/6.2019-3707},
    eprint    = {https://arc.aiaa.org/doi/pdf/10.2514/6.2019-3707}
}</pre></details>

- \[3\] Swischuk, R., Mainini, L., Peherstorfer, B., and Willcox, K.,
[Projection-based model reduction: Formulations for physics-based machine learning.](https://www.sciencedirect.com/science/article/pii/S0045793018304250)
_Computers & Fluids_, Vol. 179, pp. 704-717, 2019.
([Download](https://kiwi.oden.utexas.edu/papers/Physics-based-machine-learning-swischuk-willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{swischuk2019projection,
    title     = {Projection-based model reduction: Formulations for physics-based machine learning},
    author    = {Swischuk, R. and Mainini, L. and Peherstorfer, B. and Willcox, K.},
    journal   = {Computers \\& Fluids},
    volume    = {179},
    pages     = {704--717},
    year      = {2019},
    publisher = {Elsevier}
}</pre></details>

- \[4\] Swischuk, R., [Physics-based machine learning and data-driven reduced-order modeling](https://dspace.mit.edu/handle/1721.1/122682). Master's thesis, Massachusetts Institute of Technology, 2019. ([Download](https://dspace.mit.edu/bitstream/handle/1721.1/122682/1123218324-MIT.pdf))<details><summary>BibTeX</summary><pre>
@phdthesis{swischuk2019physics,
    title  = {Physics-based machine learning and data-driven reduced-order modeling},
    author = {Swischuk, Renee},
    year   = {2019},
    school = {Massachusetts Institute of Technology}
}</pre></details>

- \[5\] Peherstorfer, B. [Sampling low-dimensional Markovian dynamics for pre-asymptotically recovering reduced models from data with operator inference](https://arxiv.org/abs/1908.11233). arXiv:1908.11233.
([Download](https://arxiv.org/pdf/1908.11233.pdf))<details><summary>BibTeX</summary><pre>
@article{peherstorfer2019sampling,
    title   = {Sampling low-dimensional Markovian dynamics for pre-asymptotically recovering reduced models from data with operator inference},
    author  = {Peherstorfer, Benjamin},
    journal = {arXiv preprint arXiv:1908.11233},
    year    = {2019}
}</pre></details>

- \[6\] Swischuk, R., Kramer, B., Huang, C., and Willcox, K., [Learning physics-based reduced-order models for a single-injector combustion process](https://arc.aiaa.org/doi/10.2514/1.J058943). _AIAA Journal_, published online March 2020. Also in Proceedings of 2020 AIAA SciTech Forum & Exhibition, Orlando FL, January, 2020. Also Oden Institute Report 19-13. ([Download](https://kiwi.oden.utexas.edu/papers/learning-reduced-model-combustion-Swischuk-Kramer-Huang-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{SKHW2019_learning_ROMs_combustor,
    title   = {Learning physics-based reduced-order models for a single-injector combustion process},
    author  = {Swischuk, R. and Kramer, B. and Huang, C. and Willcox, K.},
    journal = {AIAA Journal},
    volume  = {},
    pages   = {Published Online: 19 Mar 2020},
    url     = {},
    year    = {2020}
}</pre></details>

- \[7\] Qian, E., Kramer, B., Peherstorfer, B., and Willcox, K. [Lift & Learn: Physics-informed machine learning for large-scale nonlinear dynamical systems](https://www.sciencedirect.com/science/article/abs/pii/S0167278919307651). _Physica D: Nonlinear Phenomena_, Volume 406, May 2020, 132401. ([Download](https://kiwi.oden.utexas.edu/papers/lift-learn-scientific-machine-learning-Qian-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{QKPW2020_lift_and_learn,
    title   = {Lift \\& Learn: Physics-informed machine learning for large-scale nonlinear dynamical systems.},
    author  = {Qian, E. and Kramer, B. and Peherstorfer, B. and Willcox, K.},
    journal = {Physica {D}: {N}onlinear {P}henomena},
    volume  = {406},
    pages   = {132401},
    url     = {https://doi.org/10.1016/j.physd.2020.132401},
    year    = {2020}
}</pre></details>

- \[8\] Benner, P., Goyal, P., Kramer, B., Peherstorfer, B., and Willcox, K. [Operator inference for non-intrusive model reduction of systems with non-polynomial nonlinear terms](https://arxiv.org/abs/2002.09726). arXiv:2002.09726. Also Oden Institute Report 20-04. ([Download](https://kiwi.oden.utexas.edu/papers/Non-intrusive-nonlinear-model-reduction-Benner-Goyal-Kramer-Peherstorfer-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{benner2020operator,
    title   = {Operator inference for non-intrusive model reduction of systems with non-polynomial nonlinear terms},
    author  = {Benner, P. and Goyal, P. and Kramer, B. and Peherstorfer, B. and Willcox, K.},
    journal = {arXiv preprint arXiv:2002.09726},
    year    = {2020}
}</pre></details>
