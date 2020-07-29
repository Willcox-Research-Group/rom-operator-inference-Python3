# Summary of Mathematical Details

This document gives a short explanation of the mathematical details behind the package.
For a full treatment, see [\[1\]](#references).
Note that some notation has been altered for coding convenience and clarity.

**Contents**
- [**Problem Statement**](#problem-statement)
- [**Projection-based Model Reduction**](#projection-based-model-reduction)
- [**Operator Inference via Least Squares**](#operator-inference-via-least-squares)
- [**Extensions and Variations**](#extensions-and-variations)
- [**Index of Notation**](#index-of-notation)
- [**References**](#references)


## Problem Statement

Consider the (possibly nonlinear) system of _n_ ordinary differential equations with state variable **x**, input (control) variable **u**, and independent variable _t_:

<p align="center"><img src="./img/details/eq01.svg"></p>

where

<p align="center"><img src="./img/details/eq02.svg"></p>

This system is called the _full-order model_ (FOM).
If _n_ is large, as it often is in high-consequence engineering applications, it is computationally expensive to numerically solve the FOM.
This package provides tools for constructing a _reduced-order model_ (ROM) that is up to quadratic in the state **x** with optional linear control inputs **u**.
The procedure is data-driven, non-intrusive, and relatively inexpensive.
In the most general case, the code can construct and solve a reduced-order system with the polynomial form

<p align="center"><img src="./img/details/eq03.svg"></p>

where now

<p align="center"><img src="./img/details/eq04.svg"></p>

This reduced low-dimensional system approximates the original high-dimensional system, but it is much easier (faster) to solve because of its low dimension _r_ << _n_.


## Projection-based Model Reduction

Model reduction via projection occurs in three steps:
1. **Data Collection**: Gather snapshot data, i.e., solutions to the full-order model (the FOM) at various times / parameters.
2. **Compression**: Compute a low-rank basis (which defines a low-dimensional linear subspace) that captures most of the behavior of the snapshots.
3. **Projection**: Use the low-rank basis to construct a low-dimensional ODE (the ROM) that approximates the FOM.

This package focuses mostly on step 3 and provides a few light tools for step 2.

Let **X** be the _n_ x _k_ matrix whose _k_ columns are each solutions to the FOM of length _n_ (step 1), and let **V**<sub>_r_</sub> be an orthonormal _n_ x _r_ matrix representation for an _r_-dimensional subspace (step 2).
A common choice for **V**<sub>_r_</sub> is the POD basis of rank _r_, the matrix whose columns are the first _r_ singular vectors of **X**.
We call **X** the _snapshot matrix_ and **V**<sub>_r_</sub> the _basis matrix_.

The classical _intrusive_ approach to the projection step is to make the Ansatz

<p align="center"><img src="./img/details/eq05.svg"></p>

Inserting this into the FOM and multiplying both sides by the transpose of **V**<sub>_r_</sub> (Galerkin projection) yields

<p align="center"><img src="./img/details/eq06.svg"></p>

This new system is _r_-dimensional in the sense that

<p align="center"><img src="./img/details/eq07.svg"></p>

If the FOM operator **f** is known and has a nice structure, this reduced system can be solved cheaply by precomputing any involved matrices and then applying a time-stepping scheme.
For example, if **f** is linear in **x** and there is no input **u**, then

<p align="center"><img src="./img/details/eq08.svg"></p>

However, this approach breaks down if the FOM operator **f** is unknown, uncertain, or highly nonlinear.

## Operator Inference via Least Squares

Instead of directly computing the reduced operators, the Operator Inference framework takes a data-driven approach: assuming a specific structure of the ROM (linear, quadratic, etc.), solve for the involved operators that best fit the data.
For example, suppose that we seek a ROM of the form

<p align="center"><img src="./img/details/eq09.svg"></p>

We start with _k_ snapshots **x**<sub>_j_</sub> and inputs **u**<sub>_j_</sub>.
That is, **x**<sub>_j_</sub> is an approximate solution to the FOM at time _t_<sub>j</sub> with input **u**<sub>_j_</sub> = **u**(_t_<sub>_j_</sub>).
We compute the basis matrix **V**<sub>_r_</sub>  from the snapshots (e.g., by taking the SVD of the matrix whose columns are the **x**<sub>_j_</sub>) and project the snapshots onto the _r_-dimensional subspace defined by the basis via

<p align="center"><img src="./img/details/eq10.svg"></p>

We also require time derivative information for the snapshots.
These may be provided by the FOM solver or estimated, for example with finite differences of the projected snapshots.
With projected snapshots, inputs, and time derivative information in hand, we then solve the least-squares problem

<p align="center"><img src="./img/details/eq11.svg"></p>

Note that this minimum-residual problem is not (yet) in a typical linear least-squares form, as the unknown quantities are the _matrices_, not the vectors.
Recalling that the vector _2_-norm is related to the matrix Frobenius norm, i.e.,

<p align="center"><img src="./img/details/eq12.svg"></p>

we can rewrite the residual objective function in the more typical matrix form:

<p align="center"><img src="./img/details/eq13.svg"></p>

where

<p align="center"><img src="./img/details/eq14.svg"></p>

and where **1**<sub>_k_</sub> is a _k_-vector of 1's and _d(r,m) = 1 + r + r<sup>2</sup> + m_.
For our purposes, the ⊗ operator between matrices denotes a column-wise Kronecker product, sometimes called the [Khatri-Rao product](https://en.wikipedia.org/wiki/Kronecker_product#Khatri%E2%80%93Rao_product).

The minimization problem given above decouples into _r_ independent ordinary least-squares problems, one for each of the columns of **O**<sup>T</sup>.
Though each independent sub-problem is typically well-posed, the problem is susceptible to noise (from model misspecification, the truncation of the basis, numerical estimation of time derivatives, etc.) and can therefore suffer from overfitting.
To combat this, the problems can be regularized with a Tikhonov penalization.
In this case, the complete minimization problem is given by

<p align="center"><img src="./img/details/eq15.svg"></p>

It can be shown [\[1\]](#references) that, under some idealized assumptions, the operators inferred by solving this data-driven minimization problem converge to the operators computed by explicit projection.
The key idea, however, is that _the inferred operators can be cheaply computed without knowing the full-order model_.
This is very convenient in "glass box" situations where the FOM is given by a legacy code for complex simulations but the target dynamics are known.

### Implementation Note: The Kronecker Product

The [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product) ⊗ introduces some redundancies.
For example, **x** ⊗ **x** contains both _x_<sub>1</sub>_x_<sub>2</sub> and _x_<sub>2</sub>_x_<sub>1</sub>.
To avoid these redundancies, we introduce a "compact" Kronecker product which only computes the unique terms of the usual Kronecker product:

<p align="center"><img src="./img/details/eq16.svg"></p>

The dimension _r(r+1)/2_ arises because we choose _2_ of _r_ entries _without replacement_, i.e., this is a _multiset_ coefficient:

<p align="center"><img src="./img/details/eq17.svg"></p>

When the compact Kronecker product is used, we call the resulting inferred quadratic operator **H**<sub>c</sub> instead of **H**.
We similarly define a cubic compact product recursively with the quadratic compact product and call the resulting cubic inferred operator **G**<sub>c</sub> instead of **G**.

## Extensions and Variations

### The Discrete Setting

The framework described above can also be used to construct reduced-order models for approximating _discrete_ dynamical systems, as may arise from discretizing PDEs in both space and time.
For instance, we can learn a discrete ROM of the form

<p align="center"><img src="./img/details/eq18.svg"></p>

The procedure is the same as described in the previous section with the exception that the snapshot matrix **X** has columns **x**<sub>_j_</sub> for _j = 0,1,...,k-1_, while the right-hand side matrix **R** has columns **x**<sub>_j_</sub> for _j = 1,...,k_.
That is, the (not yet regularized) least-squares problem to be solved has the form

<p align="center"><img src="./img/details/eq19.svg"></p>

<!-- TODO: ### The Steady Setting -->

<!-- TODO: ### Lifting and Variable Transformations -->

<!-- TODO: ### Re-projection and Recovering Intrusive Models -->

<!-- TODO: ### Incorporating Nonlinear Terms with DEIM -->

<!-- TODO: ### Learning Parametric Models  -->

## Index of Notation

We generally denote scalars in lower case, vectors in bold lower case, matrices in upper case, and indicate low-dimensional quantities with a hat.
In the code, a low-dimensional quantity ends with an underscore, so that the model classes follow some principles from the [scikit-learn](https://scikit-learn.org/stable/index.html) [API](https://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects).

### Dimensions

| Symbol | Code | Description |
| :----: | :--- | :---------- |
| <img src="./img/notation/eq01.svg"> | `n`  | Dimension of the full-order system (large) |
| <img src="./img/notation/eq02.svg"> | `r`  | Dimension of the reduced-order system (small) |
| <img src="./img/notation/eq03.svg"> | `m`  | Dimension of the input **u** |
| <img src="./img/notation/eq05.svg"> | `k`  | Number of state snapshots, i.e., the number of training points |
| <img src="./img/notation/eq06.svg"> | `s`  | Number of parameter samples for parametric training |
| <img src="./img/notation/eq07.svg"> | `p` | Dimension of the parameter space |
| <img src="./img/notation/eq08.svg"> | `d` | Number of columns of the data matrix _D_ |

<!-- | <img src="./img/notation/eq04.svg"> | `l` | Dimension of the output **y** | -->

<!-- ### Scalars

| Symbol | Code | Description |
| :----: | :--- | :---------- |
| <img src="./img/notation/eq09.svg"> | `nt`  | Number of time steps in a simulation |
| | `µ` | Scalar parameter (_p_ = 1). | -->


### Vectors

| Symbol | Code | Size | Description |
| :----: | :--- | :--: | :---------- |
| <img src="./img/notation/eq10.svg"> | `x` | <img src="./img/notation/eq01.svg"> | Full-order state vector |
| <img src="./img/notation/eq11.svg"> | `x_` | <img src="./img/notation/eq02.svg"> | Reduced-order state vector |
| <img src="./img/notation/eq12.svg"> | `xdot_` | <img src="./img/notation/eq02.svg"> | Reduced-order state time derivative vector |
| <img src="./img/notation/eq13.svg"> | `x_ROM` | <img src="./img/notation/eq01.svg"> | Approximation to **x** produced by ROM |
| <img src="./img/notation/eq14.svg"> | `c_` | <img src="./img/notation/eq02.svg"> | Learned constant term  |
| <img src="./img/notation/eq15.svg"> | `u` | <img src="./img/notation/eq03.svg"> | Input vector  |
| <img src="./img/notation/eq17.svg"> | `f(t,x,u(t))` or `f(x,u)` | <img src="./img/notation/eq01.svg">  | Full-order system operator |
| <img src="./img/notation/eq18.svg"> | `f_(t,x_,u(t))` or `f(x_,u)` | <img src="./img/notation/eq02.svg">  | Reduced-order system operator |
| <img src="./img/notation/eq19.svg"> | `np.kron(x,x)` | <img src="./img/notation/eq20.svg"> | Quadratic Kronecker product of full state |
| <img src="./img/notation/eq21.svg"> | `np.kron(x_,x_)` | <img src="./img/notation/eq22.svg">  | Quadratic Kronecker product of reduced state |
| <img src="./img/notation/eq23.svg"> | `utils.kron2c(x_)` | <img src="./img/notation/eq24.svg">  | Compact quadratic Kronecker product of reduced state |
| <img src="./img/notation/eq25.svg"> | `np.kron(x,np.kron(x,x))` | <img src="./img/notation/eq26.svg"> | Cubic Kronecker product of full state |
| <img src="./img/notation/eq27.svg"> | `np.kron(x_,np.kron(x_,x_))` | <img src="./img/notation/eq28.svg">  | Cubic Kronecker product of reduced state |
| <img src="./img/notation/eq29.svg"> | `utils.kron3c(x_)` | <img src="./img/notation/eq30.svg">  | Compact cubic Kronecker product of reduced state |
| <img src="./img/notation/eq31.svg"> | `vj` | <img src="./img/notation/eq01.svg"> | _j_<sup>th</sup> subspace basis vector, i.e., column _j_ of **V**<sub>_r_</sub> |

<!-- | **y**  | `y`             | Output vector | -->
<!-- | **y_ROM**, **y~** | `y_ROM`      | Approximation to **y** produced by ROM | -->

### Matrices

| Symbol | Code | Shape | Description |
| :----: | :--- | :---: | :---------- |
| <img src="./img/notation/eq32.svg"> | `Vr` | <img src="./img/notation/eq33.svg"> | low-rank basis of rank _r_ (usually the POD basis) |
| <img src="./img/notation/eq34.svg"> | `X` | <img src="./img/notation/eq35.svg"> | Snapshot matrix |
| <img src="./img/notation/eq36.svg"> | `Xdot` | <img src="./img/notation/eq35.svg"> | Snapshot time derivative matrix |
| <img src="./img/notation/eq37.svg"> | `U` | <img src="./img/notation/eq38.svg"> | Input matrix (inputs corresonding to the snapshots) |
| <img src="./img/notation/eq41.svg"> | `X_` | <img src="./img/notation/eq42.svg"> | Projected snapshot matrix |
| <img src="./img/notation/eq43.svg"> | `Xdot_` | <img src="./img/notation/eq42.svg"> | Projected snapshot time derivative matrix |
| <img src="./img/notation/eq44.svg"> | `D` | <img src="./img/notation/eq45.svg"> | Data matrix |
| <img src="./img/notation/eq46.svg"> | `O` | <img src="./img/notation/eq47.svg"> | Operator matrix |
| <img src="./img/notation/eq48.svg"> | `R` | <img src="./img/notation/eq49.svg"> | Right-hand side matrix |
| <img src="./img/notation/eq50.svg"> | `P` | <img src="./img/notation/eq51.svg"> | Tikhonov regularization matrix |
| <img src="./img/notation/eq52.svg"> | `A` | <img src="./img/notation/eq53.svg"> | Full-order linear state matrix |
| <img src="./img/notation/eq54.svg"> | `A_` | <img src="./img/notation/eq55.svg"> | Reduced-order linear state matrix |
| <img src="./img/notation/eq56.svg"> | `H` | <img src="./img/notation/eq57.svg"> | Full-order matricized quadratic state tensor |
| <img src="./img/notation/eq58.svg"> | `H_` | <img src="./img/notation/eq59.svg"> | Reduced-order matricized quadratic state tensor |
| <img src="./img/notation/eq60.svg"> | `Hc_` | <img src="./img/notation/eq61.svg"> | Compact reduced-order matricized quadratic state tensor |
| <img src="./img/notation/eq62.svg"> | `G` | <img src="./img/notation/eq63.svg"> | Full-order matricized quadratic state tensor |
| <img src="./img/notation/eq64.svg"> | `G_` | <img src="./img/notation/eq65.svg"> | Reduced-order matricized quadratic state tensor |
| <img src="./img/notation/eq66.svg"> | `Gc_` | <img src="./img/notation/eq67.svg"> | Compact reduced-order matricized quadratic state tensor |
| <img src="./img/notation/eq68.svg"> | `B` | <img src="./img/notation/eq69.svg"> | Full-order input matrix |
| <img src="./img/notation/eq70.svg"> | `B_` | <img src="./img/notation/eq71.svg"> | Reduced-order input matrix |

<!-- | <img src="./img/notation/eq72.svg"> | `C` | <img src="./img/notation/eq73.svg"> | Full-order output matrix | -->
<!-- | <img src="./img/notation/eq74.svg"> | `C_` | <img src="./img/notation/eq75.svg"> | Reduced-order output matrix | -->
<!-- | <img src="\hat{N}_i"> | `Ni_` | <img src="./img/notation/eq55.svg"> | Bilinear state-input matrix for _i_th input | -->


## References

- \[1\] [Peherstorfer, B.](https://scholar.google.com/citations?user=C81WhlkAAAAJ) and [Willcox, K.](https://kiwi.oden.utexas.edu/),
[Data-driven operator inference for non-intrusive projection-based model reduction.](https://www.sciencedirect.com/science/article/pii/S0045782516301104)
_Computer Methods in Applied Mechanics and Engineering_, Vol. 306, pp. 196-215, 2016.
([Download](https://kiwi.oden.utexas.edu/papers/Non-intrusive-model-reduction-Peherstorfer-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{PW2016OperatorInference,
    title     = {Data-driven operator inference for nonintrusive projection-based model reduction},
    author    = {Peherstorfer, B. and Willcox, K.},
    journal   = {Computer Methods in Applied Mechanics and Engineering},
    volume    = {306},
    pages     = {196--215},
    year      = {2016},
    publisher = {Elsevier}
}</pre></details>

- \[2\] [Qian, E.](https://scholar.google.com/citations?user=jnHI7wQAAAAJ), [Kramer, B.](http://kramer.ucsd.edu/), [Marques, A.](https://scholar.google.com/citations?user=d4tBWWwAAAAJ), and [Willcox, K.](https://kiwi.oden.utexas.edu/),
[Transform & Learn: A data-driven approach to nonlinear model reduction](https://arc.aiaa.org/doi/10.2514/6.2019-3707).
In the AIAA Aviation 2019 Forum & Exhibition, Dallas, TX, June 2019. ([Download](https://kiwi.oden.utexas.edu/papers/learn-data-driven-nonlinear-reduced-model-Qian-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@inbook{QKMW2019TransformAndLearn,
    title     = {Transform \\& Learn: A data-driven approach to nonlinear model reduction},
    author    = {Qian, E. and Kramer, B. and Marques, A. N. and Willcox, K. E.},
    booktitle = {AIAA Aviation 2019 Forum},
    doi       = {10.2514/6.2019-3707},
    URL       = {https://arc.aiaa.org/doi/abs/10.2514/6.2019-3707},
    eprint    = {https://arc.aiaa.org/doi/pdf/10.2514/6.2019-3707}
}</pre></details>

- \[3\] [Swischuk, R.](https://scholar.google.com/citations?user=L9D0LBsAAAAJ), [Mainini, L.](https://scholar.google.com/citations?user=1mo8GgkAAAAJ), [Peherstorfer, B.](https://scholar.google.com/citations?user=C81WhlkAAAAJ), and [Willcox, K.](https://kiwi.oden.utexas.edu/),
[Projection-based model reduction: Formulations for physics-based machine learning.](https://www.sciencedirect.com/science/article/pii/S0045793018304250)
_Computers & Fluids_, Vol. 179, pp. 704-717, 2019.
([Download](https://kiwi.oden.utexas.edu/papers/Physics-based-machine-learning-swischuk-willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{SMPW2019PhysicsbasedML,
    title     = {Projection-based model reduction: Formulations for physics-based machine learning},
    author    = {Swischuk, R. and Mainini, L. and Peherstorfer, B. and Willcox, K.},
    journal   = {Computers \\& Fluids},
    volume    = {179},
    pages     = {704--717},
    year      = {2019},
    publisher = {Elsevier}
}</pre></details>

- \[4\] [Swischuk, R.](https://scholar.google.com/citations?user=L9D0LBsAAAAJ), [Physics-based machine learning and data-driven reduced-order modeling](https://dspace.mit.edu/handle/1721.1/122682). Master's thesis, Massachusetts Institute of Technology, 2019. ([Download](https://dspace.mit.edu/bitstream/handle/1721.1/122682/1123218324-MIT.pdf))<details><summary>BibTeX</summary><pre>
@phdthesis{swischuk2019MLandDDROM,
    title  = {Physics-based machine learning and data-driven reduced-order modeling},
    author = {Swischuk, Renee},
    year   = {2019},
    school = {Massachusetts Institute of Technology}
}</pre></details>

- \[5\] [Peherstorfer, B.](https://scholar.google.com/citations?user=C81WhlkAAAAJ) [Sampling low-dimensional Markovian dynamics for pre-asymptotically recovering reduced models from data with operator inference](https://arxiv.org/abs/1908.11233). arXiv:1908.11233.
([Download](https://arxiv.org/pdf/1908.11233.pdf))<details><summary>BibTeX</summary><pre>
@article{peherstorfer2019samplingMarkovian,
    title   = {Sampling low-dimensional Markovian dynamics for pre-asymptotically recovering reduced models from data with operator inference},
    author  = {Peherstorfer, Benjamin},
    journal = {arXiv preprint arXiv:1908.11233},
    year    = {2019}
}</pre></details>

- \[6\] [Swischuk, R.](https://scholar.google.com/citations?user=L9D0LBsAAAAJ), [Kramer, B.](http://kramer.ucsd.edu/), [Huang, C.](https://scholar.google.com/citations?user=lUXijaQAAAAJ), and [Willcox, K.](https://kiwi.oden.utexas.edu/), [Learning physics-based reduced-order models for a single-injector combustion process](https://arc.aiaa.org/doi/10.2514/1.J058943). _AIAA Journal_, Vol. 58:6, pp. 2658-2672, 2020. Also in Proceedings of 2020 AIAA SciTech Forum & Exhibition, Orlando FL, January, 2020. Also Oden Institute Report 19-13.
([Download](https://kiwi.oden.utexas.edu/papers/learning-reduced-model-combustion-Swischuk-Kramer-Huang-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{SKHW2020ROMCombustion,
    title     = {Learning physics-based reduced-order models for a single-injector combustion process},
    author    = {Swischuk, R. and Kramer, B. and Huang, C. and Willcox, K.},
    journal   = {AIAA Journal},
    volume    = {58},
    number    = {6},
    pages     = {2658--2672},
    year      = {2020},
    publisher = {American Institute of Aeronautics and Astronautics}
}</pre></details>

- \[7\] [Qian, E.](https://scholar.google.com/citations?user=jnHI7wQAAAAJ), [Kramer, B.](http://kramer.ucsd.edu/), [Peherstorfer, B.](https://scholar.google.com/citations?user=C81WhlkAAAAJ), and [Willcox, K.](https://kiwi.oden.utexas.edu/) [Lift & Learn: Physics-informed machine learning for large-scale nonlinear dynamical systems](https://www.sciencedirect.com/science/article/abs/pii/S0167278919307651). _Physica D: Nonlinear Phenomena_, Vol. 406, May 2020, 132401. ([Download](https://kiwi.oden.utexas.edu/papers/lift-learn-scientific-machine-learning-Qian-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{QKPW2020LiftAndLearn,
    title   = {Lift \\& Learn: Physics-informed machine learning for large-scale nonlinear dynamical systems.},
    author  = {Qian, E. and Kramer, B. and Peherstorfer, B. and Willcox, K.},
    journal = {Physica {D}: {N}onlinear {P}henomena},
    volume  = {406},
    pages   = {132401},
    url     = {https://doi.org/10.1016/j.physd.2020.132401},
    year    = {2020}
}</pre></details>

- \[8\] [Benner, P.](https://scholar.google.com/citations?user=6zcRrC4AAAAJ), [Goyal, P.](https://scholar.google.com/citations?user=9rEfaRwAAAAJ), [Kramer, B.](http://kramer.ucsd.edu/), [Peherstorfer, B.](https://scholar.google.com/citations?user=C81WhlkAAAAJ), and [Willcox, K.](https://kiwi.oden.utexas.edu/) [Operator inference for non-intrusive model reduction of systems with non-polynomial nonlinear terms](https://arxiv.org/abs/2002.09726). arXiv:2002.09726. Also Oden Institute Report 20-04. ([Download](https://kiwi.oden.utexas.edu/papers/Non-intrusive-nonlinear-model-reduction-Benner-Goyal-Kramer-Peherstorfer-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{BGKPW2020OpInfNonPoly,
    title   = {Operator inference for non-intrusive model reduction of systems with non-polynomial nonlinear terms},
    author  = {Benner, P. and Goyal, P. and Kramer, B. and Peherstorfer, B. and Willcox, K.},
    journal = {arXiv preprint arXiv:2002.09726},
    year    = {2020}
}</pre></details>
