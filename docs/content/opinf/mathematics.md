(sec-opinf-math)=
# Mathematical Details

This page gives a short explanation of the mathematical details behind Operator Inference.
See {cite}`PW2016OperatorInference` for a full treatment.

## Problem Statement

Consider the (possibly nonlinear) system of $n$ ordinary differential equations with state variable $\mathbf{q}$, input (control) variable $\mathbf{u}$, and independent variable $t$:

$$
\frac{\textup{d}}{\textup{d}t}\mathbf{q}(t)
= \mathbf{f}(t,\mathbf{q}(t),\mathbf{u}(t)),
\qquad
\mathbf{q}(0)
= \mathbf{q}_0,
$$

where

$$
\mathbf{q}:\mathbb{R}\to\mathbb{R}^{n},
\qquad
\mathbf{u}:\mathbb{R}\to\mathbb{R}^{m},
\qquad
\mathbf{f}:\mathbb{R}\times\mathbb{R}^{n}\times\mathbb{R}^{m}\to\mathbb{R}^{n}.
$$

This system is called the _full-order model_ (FOM).
If $n$ is large, as it often is in high-consequence engineering applications, it is computationally expensive to numerically solve the FOM.
This package provides tools for constructing a _reduced-order model_ (ROM) that is up to quadratic in the state $\mathbf{q}$ with optional linear control inputs $\mathbf{u}$.
The procedure is data-driven, non-intrusive, and relatively inexpensive.
In the most general case, the code can construct and solve a reduced-order system with the polynomial form

$$
\begin{align*}
    \frac{\textup{d}}{\textup{d}t}\widehat{\mathbf{q}}(t)
    = \widehat{\mathbf{c}}
    + \widehat{\mathbf{A}}\widehat{\mathbf{q}}(t)
    + \widehat{\mathbf{H}}(\widehat{\mathbf{q}}(t)\otimes\widehat{\mathbf{q}}(t))
    + \widehat{\mathbf{G}}(\widehat{\mathbf{q}}(t)\otimes\widehat{\mathbf{q}}(t)\otimes\widehat{\mathbf{q}}(t))
    + \widehat{\mathbf{B}}\mathbf{u}(t),
\end{align*}
$$

where now

$$
\begin{gather*}
    \widehat{\mathbf{q}}:\mathbb{R}\to\mathbb{R}^{r},
    \qquad
    \mathbf{u}:\mathbb{R}\to\mathbb{R}^{m},
    \qquad
    \widehat{\mathbf{c}}\in\mathbb{R}^{r},
    \qquad
    r \ll n,
    \\
    \widehat{\mathbf{A}}\in\mathbb{R}^{r \times r},
    \qquad
    \widehat{\mathbf{H}}\in\mathbb{R}^{r \times r^2},
    \qquad
    \widehat{\mathbf{G}}\in\mathbb{R}^{r \times r^3},
    \qquad
    \widehat{\mathbf{B}}\in\mathbb{R}^{r \times m}.
\end{gather*}
$$

This reduced low-dimensional system approximates the original high-dimensional system, but it is much easier (faster) to solve because of its low state dimension $r \ll n$.


## Projection-based Model Reduction

Model reduction via projection occurs in three steps:
1. **Data Collection**: Gather snapshot data, i.e., solutions to the full-order model (the FOM) at various times / parameters.
2. **Compression**: Compute a low-rank basis (which defines a low-dimensional linear subspace) that captures most of the behavior of the snapshots.
3. **Projection**: Use the low-rank basis to construct a low-dimensional ODE (the ROM) that approximates the FOM.

This package focuses mostly on step 3 and provides a few light tools for step 2.

Let $\mathbf{Q}\in\mathbb{R}^{n \times k}$ be the matrix whose $k$ columns are each solutions to the FOM of length $n$ (step 1), and let $\mathbf{V}_{r}\in\mathbb{R}^{n \times r}$ be an orthonormal matrix representation for an $r$-dimensional subspace (step 2).
A common choice for $\mathbf{V}_{r}$ is the POD basis of rank $r$, the matrix whose columns are the first $r$ singular vectors of $\mathbf{Q}$.
We call $\mathbf{Q}$ the _snapshot matrix_ and $\mathbf{V}_{r}$ the _basis matrix_.

The classical _intrusive_ approach to the projection step is to make the Ansatz

$$
\begin{align*}
    \mathbf{q}(t)
    \approx \mathbf{V}_{r}\widehat{\mathbf{q}}(t),
    \qquad
    \widehat{\mathbf{q}}:\mathbb{R}\to\mathbb{R}^{r}.
\end{align*}
$$

Inserting this into the FOM and multiplying both sides by $\mathbf{V}_{r}^{\mathsf{T}}$ (Galerkin projection) yields

$$
\begin{align*}
    \frac{\textup{d}}{\textup{d}t}\mathbf{q}(t)
    = \mathbf{V}_{r}^{\mathsf{T}}\mathbf{f}(t,\mathbf{V}_{r}\widehat{\mathbf{q}}(t),\mathbf{u}(t))
    =: \widehat{\mathbf{f}}(t,\widehat{\mathbf{q}}(t),\mathbf{u}(t)).
\end{align*}
$$

This new system is $r$-dimensional in the sense that

$$
\begin{align*}
    \widehat{\mathbf{f}}:\mathbb{R}\times\mathbb{R}^{r}\times\mathbb{R}^{m}\to\mathbb{R}^{r}.
\end{align*}
$$

If the FOM operator $\mathbf{f}$ is known and has a nice structure, this reduced system can be solved cheaply by precomputing any involved matrices and then applying a time-stepping scheme.
For example, if $\mathbf{f}$ is linear in the state $\mathbf{q}$ and there is no input $\mathbf{u}$, then

$$
\begin{align*}
    \mathbf{f}(t,\mathbf{q}(t)) = \mathbf{A}\mathbf{q}(t)
    \qquad\Longrightarrow\qquad
    \widetilde{\mathbf{f}}(t,\widetilde{\mathbf{q}}(t)) = \mathbf{V}_{r}^{\mathsf{T}}\mathbf{A}\mathbf{V}_{r}\widetilde{\mathbf{q}}(t) = \widetilde{\mathbf{A}}\widetilde{\mathbf{q}}(t),
    \\
    \text{where}\quad
    \widetilde{\mathbf{A}}
    := \mathbf{V}_{r}^{\mathsf{T}}\mathbf{A}\mathbf{V}_{r}\in\mathbb{R}^{r \times r}.
\end{align*}
$$

However, this approach breaks down if the FOM operator $\mathbf{f}$ is unknown, uncertain, or highly nonlinear.

## Least Squares Regression

Instead of directly computing the reduced operators, the Operator Inference framework takes a data-driven approach: assuming a specific structure of the ROM (linear, quadratic, etc.), solve for the involved operators that best fit the data.
For example, suppose that we seek a ROM of the form

$$
\begin{align*}
    \frac{\textup{d}}{\textup{d}t}\widehat{\mathbf{q}}(t)
    = \widehat{\mathbf{c}}
    + \widehat{\mathbf{A}}\widehat{\mathbf{q}}(t)
    + \widehat{\mathbf{H}}(\widehat{\mathbf{q}}(t)\otimes\widehat{\mathbf{q}}(t))
    + \widehat{\mathbf{B}}\mathbf{u}(t).
\end{align*}
$$

We start with $k$ snapshots $\mathbf{q}_{j}$ and inputs $\mathbf{u}_{j}$.
That is, $\mathbf{q}_{j}$ is an approximate solution to the FOM at time $t_{j}$ with input $\mathbf{u}_{j} := \mathbf{u}(t_{j})$.
We compute the basis matrix $\mathbf{V}_{r}$  from the snapshots (e.g., by taking the SVD of the matrix whose columns are the $\mathbf{q}_{j}$) and project the snapshots onto the $r$-dimensional subspace defined by the basis via

$$
\begin{align*}
    \widehat{\mathbf{q}}_{j} = \mathbf{V}_{r}^{\mathsf{T}}\mathbf{q}_{j},
\end{align*}
$$

We also require time derivative information for the snapshots.
These may be provided by the FOM solver or estimated, for example with finite differences of the projected snapshots.
With projected snapshots, inputs, and time derivative information in hand, we then solve the least-squares problem

$$
\begin{align*}
    \min_{\widehat{\mathbf{c}},\widehat{\mathbf{A}},\widehat{\mathbf{H}},\widehat{\mathbf{B}}}\sum_{j=0}^{k-1}\left\|
      \widehat{\mathbf{c}}
    + \widehat{\mathbf{A}}\widehat{\mathbf{q}}_j
    + \widehat{\mathbf{H}}(\widehat{\mathbf{q}}_j\otimes\widehat{\mathbf{q}}_j)
    + \widehat{\mathbf{B}}\mathbf{u}_j
    - \dot{\widehat{\mathbf{q}}}_j
    \right\|_2^2.
\end{align*}
$$

Note that this minimum-residual problem is not (yet) in a typical linear least-squares form, as the unknown quantities are the _matrices_, not the vectors.
<!-- Recalling that the vector $2$-norm is related to the matrix Frobenius norm, i.e.,

$$
\begin{align*}
    \sum_{j=0}^{k-1}\|\mathbf{z}_j\|_2^2
    = \|\mathbf{Z}\|_F^2,
    \qquad
    \mathbf{Z}
    = \left[\begin{array}{cccc}
        &&& \\
        \mathbf{z}_0 & \mathbf{z}_1 & \cdots & \mathbf{z}_{k-1} \\
        &&& \\
    \end{array}\right],
\end{align*}
$$ -->
However, we can rewrite the residual objective function in the more typical matrix form:

$$
\begin{align*}
    \sum_{j=0}^{k-1}\left\|
      \widehat{\mathbf{c}}
    + \widehat{\mathbf{A}}\widehat{\mathbf{q}}_j
    + \widehat{\mathbf{H}}(\widehat{\mathbf{q}}_j\otimes\widehat{\mathbf{q}}_j)
    + \widehat{\mathbf{B}}\mathbf{u}_j
    - \dot{\widehat{\mathbf{q}}}_j
    \right\|_2^2
    &=
    \left\|
      \widehat{\mathbf{c}}\mathbf{1}_{k}^{\mathsf{T}}
    + \widehat{\mathbf{A}}\widehat{\mathbf{Q}}
    + \widehat{\mathbf{H}}(\widehat{\mathbf{Q}}\otimes\widehat{\mathbf{Q}})
    + \widehat{\mathbf{B}}\mathbf{U}
    - \dot{\widehat{\mathbf{Q}}}
    \right\|_F^2
    \\
    &=
    \left\|
      \mathbf{1}_{k}\widehat{\mathbf{c}}^{\mathsf{T}}
    + \widehat{\mathbf{Q}}^{\mathsf{T}}\widehat{\mathbf{A}}^{\mathsf{T}}
    + (\widehat{\mathbf{Q}}\otimes\widehat{\mathbf{Q}})^{\mathsf{T}}\widehat{\mathbf{H}}^{\mathsf{T}}
    + \mathbf{U}^{\mathsf{T}}\widehat{\mathbf{B}}^{\mathsf{T}}
    - \dot{\widehat{\mathbf{Q}}}^{\mathsf{T}}
    \right\|_F^2
    \\
    &=
    \left\|\mathbf{D} \widehat{\mathbf{O}}^{\mathsf{T}} - \mathbf{R}^{\mathsf{T}}\right\|_F^2,
\end{align*}
$$ (eq_opinf_matrix)

where

$$
\begin{align*}
    \widehat{\mathbf{O}}
    &= \left[\begin{array}{cccc}
        \widehat{\mathbf{c}} &
        \widehat{\mathbf{A}} &
        \widehat{\mathbf{H}} &
        \widehat{\mathbf{B}}
    \end{array}\right]\in\mathbb{R}^{r\times d(r,m)},
    &\text{(unknown operators)}
    \\
    \mathbf{D}
    &= \left[\begin{array}{cccc}
        \mathbf{1}_{k} &
        \widehat{\mathbf{Q}}^{\mathsf{T}} &
        (\widehat{\mathbf{Q}}\otimes\widehat{\mathbf{Q}})^{\mathsf{T}} &
        \mathbf{U}^{\mathsf{T}}
    \end{array}\right]\in\mathbb{R}^{k\times d(r,m)},
    &\text{(known data)}
    \\
    \widehat{\mathbf{Q}}
    &= \left[\begin{array}{cccc}
        \widehat{\mathbf{q}}_0 & \widehat{\mathbf{q}}_1 & \cdots & \widehat{\mathbf{q}}_{k-1} \\
    \end{array}\right]\in\mathbb{R}^{r\times k}
    &\text{(snapshots)}
    \\
    \mathbf{R}
    &= \left[\begin{array}{cccc}
        \dot{\widehat{\mathbf{q}}}_0 & \dot{\widehat{\mathbf{q}}}_1 & \cdots & \dot{\widehat{\mathbf{q}}}_{k-1} \\
    \end{array}\right]\in\mathbb{R}^{r\times k},
    &\text{(time derivatives)}
    \\
    \mathbf{U}
    &= \left[\begin{array}{cccc}
        \mathbf{u}_0 & \mathbf{u}_1 & \cdots & \mathbf{u}_{k-1} \\
    \end{array}\right]\in\mathbb{R}^{m\times k},
    &\text{(inputs)}
\end{align*}
$$

and where $\mathbf{1}_{k}\in\mathbb{R}^{k}$ is a vector of 1's and $d(r,m) = 1 + r + r^2 + m$.
For our purposes, the $\otimes$ operator between matrices denotes a column-wise Kronecker product, sometimes called the [Khatri-Rao product](https://en.wikipedia.org/wiki/Kronecker_product#Khatri%E2%80%93Rao_product).

It was shown in {cite}`PW2016OperatorInference` that, under some idealized assumptions, the operators inferred by solving this data-driven minimization problem converge to the operators computed by explicit projection.
The key idea, however, is that the inferred operators can be cheaply computed without knowing the full-order model.
This is very convenient in "glass box" situations where the FOM is given by a legacy code for complex simulations and the governing dynamics are known.

```{note}
The minimization problem {eq}`eq_opinf_matrix` is only well-posed when $\mathbf{D}$ has full column rank.
Even then, the conditioning of the problem may be poor due to noise from model misspecification, the truncation of the basis, numerical estimation of time derivatives, and so forth.
The problem therefore often requires a regularization strategy to combat ill-posedness; see [Least-squares Solvers](sec-lstsq) for options implemented in this package.
```
