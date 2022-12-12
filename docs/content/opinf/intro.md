(sec-opinf-overview)=
# What is Operator Inference?

:::{image} ../../images/summary.svg
:align: center
:width: 80 %
:::

The goal of Operator Inference is to construct a low-dimensional, computationally inexpensive system whose solutions are close to those of some high-dimensional system for which we have 1) training data and 2) some knowledge about the system structure.
The main steps are the following.

1. [**Get Training Data**](subsec-training-data). Gather and [preprocess](sec-preprocessing) high-dimensional data to learn a low-dimensional model from. This package has a few common preprocessing tools, but the user must bring the data to the table.
2. [**Compute a Low-dimensional Representation**](subsec-basis-computation). Represent the high-dimensional data with only a few degrees of freedom. The simplest approach is to take the SVD of the high-dimensional training data, extract the first few left singular vectors, and use these vectors as a new coordinate basis.
3. [**Set up and Solve a Low-dimensional Regression**](subsec-opinf-regression). Use the low-dimensional representation of the training data to determine a reduced-order model that best fits the data in a minimum-residual sense. This is the core objective of the package.
4. [**Evaluate the Reduced-order Model**](subsec-rom-evaluation). Use the learned model to make computationally efficient predictions.

This page gives an overview of Operator Inference by walking through each of these steps.

---

## Problem Statement

::::{margin}
:::{note}
We are most often interested in full-order models that are spatial discretizations of partial differential equations (PDEs), but {eq}`eq:opinf-example-fom` does not necessarily have to be related to a PDE.
:::
::::

Consider a system of ordinary differential equations (ODEs) with state $\mathbf{q}(t)\in\mathbb{R}^{n}$ and inputs $\mathbf{u}(t)\in\mathbb{R}^{m}$,

$$
    \frac{\text{d}}{\text{d}t}\mathbf{q}(t)
    = \mathbf{F}(t, \mathbf{q}(t), \mathbf{u}(t)).
$$ (eq:opinf-example-fom)

We call {eq}`eq:opinf-example-fom` the _full-order model_.
Given samples of the state $\mathbf{q}(t)$, Operator Inference learns a surrogate system for {eq}`eq:opinf-example-fom` with the much smaller state $\widehat{\mathbf{q}}(t) \in \mathbb{R}^{r}, r \ll n,$ and a polynomial structure, for example:

::::{margin}
:::{note}
The $\otimes$ operator is called the [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product).
:::
::::

$$
    \frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t)
    = \widehat{\mathbf{c}}
    + \widehat{\mathbf{A}}\widehat{\mathbf{q}}(t)
    + \widehat{\mathbf{H}}[\widehat{\mathbf{q}}(t)\otimes\widehat{\mathbf{q}}(t)]
    + \widehat{\mathbf{B}}\mathbf{u}(t).
$$ (eq:opinf-example-rom)

We call {eq}`eq:opinf-example-rom` a _reduced-order model_ for {eq}`eq:opinf-example-fom`.
Our goal is to infer the _reduced-order operators_ $\widehat{\mathbf{c}} \in \mathbb{R}^{r}$, $\widehat{\mathbf{A}}\in\mathbb{R}^{r\times r}$, $\widehat{\mathbf{H}}\in\mathbb{R}^{r\times r^{2}}$, and/or $\widehat{\mathbf{B}}\in\mathbb{R}^{r\times m}$ using data from {eq}`eq:opinf-example-fom`.
The user specifies [which terms to include in the model](subsec-romclass-constructor).

::::{important}
:name: projection-preserves-structure
The right-hand side of {eq}`eq:opinf-example-rom` is a polynomial with respect to the state $\widehat{\mathbf{q}}(t)$:
$\widehat{\mathbf{c}}$ are the constant terms, $\widehat{\mathbf{A}}\widehat{\mathbf{q}}(t)$ are the linear terms, $\widehat{\mathbf{H}}[\widehat{\mathbf{q}}(t)\otimes\widehat{\mathbf{q}}(t)]$ are the quadratic terms, with input terms $\widehat{\mathbf{B}}\mathbf{u}(t)$.
The user must choose which terms to include in the reduced-order model, and this choice should be motivated by the structure of the full-order model {eq}`eq:opinf-example-fom`.
For example, if the full-order model can be written as

$$
    \frac{\text{d}}{\text{d}t}\mathbf{q}(t)
    = \mathbf{A}\mathbf{q}(t) + \mathbf{B}\mathbf{u}(t),
$$

then the reduced-order model should mirror this structure as

$$
    \frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t)
    = \widehat{\mathbf{A}}\widehat{\mathbf{q}}(t)
    + \widehat{\mathbf{B}}\mathbf{u}(t).
$$

:::{dropdown} Motivation
**Projection preserves polynomial structure** {cite}`benner2015pmorsurvey,peherstorfer2016opinf`.
The classical (Galerkin) projection-based reduced-order model for the system

$$
    \frac{\text{d}}{\text{d}t}\mathbf{q}(t)
    = \mathbf{c}
    + \mathbf{A}\mathbf{q}(t)
    + \mathbf{H}[\mathbf{q}(t)\otimes\mathbf{q}(t)]
    + \mathbf{B}\mathbf{u}(t)
$$

is obtained by substituting $\mathbf{q}(t)$ with $\mathbf{V}\widehat{\mathbf{q}}(t)$ for some orthonormal $\mathbf{V}\in\mathbb{R}^{n \times r}$ and $\widehat{\mathbf{q}}(t)\in \mathbb{R}^{r}$, then multiplying both sides by $\mathbf{V}^{\mathsf{T}}$:

$$
    \mathbf{V}^{\mathsf{T}}\frac{\text{d}}{\text{d}t}\left[\mathbf{V}\widehat{\mathbf{q}}(t)\right]
    = \mathbf{V}^{\mathsf{T}}\left(\mathbf{c}
    + \mathbf{A}\mathbf{V}\widehat{\mathbf{q}}(t)
    + \mathbf{H}[(\mathbf{V}\widehat{\mathbf{q}}(t))\otimes(\mathbf{V}\widehat{\mathbf{q}}(t))]
    + \mathbf{B}\mathbf{u}(t)\right).
$$

Since $\mathbf{V}^{\mathsf{T}}\mathbf{V}$ is the identity and $(\mathbf{X}\mathbf{Y})\otimes(\mathbf{Z}\mathbf{W}) = (\mathbf{X}\otimes \mathbf{Z})(\mathbf{Y}\otimes\mathbf{W})$, this simplifies to

$$
    \frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t)
    =
    \widehat{\mathbf{c}}
    + \widehat{\mathbf{A}}\widehat{\mathbf{q}}(t)
    + \widehat{\mathbf{H}}[\widehat{\mathbf{q}}(t)\otimes\widehat{\mathbf{q}}(t)]
    + \widehat{\mathbf{B}}\mathbf{u}(t),
$$

where $\widehat{\mathbf{c}} = \mathbf{V}^{\mathsf{T}}\mathbf{c}$, $\widehat{\mathbf{A}} = \mathbf{V}^{\mathsf{T}}\mathbf{A}\mathbf{V}$, $\widehat{\mathbf{H}} = \mathbf{V}^{\mathsf{T}}\mathbf{H}\left(\mathbf{V}\otimes\mathbf{V}^{\mathsf{T}}\right)$, and $\widehat{\mathbf{B}} = \mathbf{V}^{\mathsf{T}}\mathbf{B}$.
Operator Inference learns $\widehat{\mathbf{c}}$, $\widehat{\mathbf{A}}$, $\widehat{\mathbf{H}}$, and/or $\widehat{\mathbf{B}}$ _from data_ and is therefore useful for situations where $\mathbf{c}$, $\mathbf{A}$, $\mathbf{H}$, and/or $\mathbf{B}$ are not explicitly available for matrix computations.
:::
::::

---

(subsec-training-data)=
## Get Training Data

Operator Inference learns reduced-order models from full-order state/input data.
Start by gathering solution and input data and organizing them columnwise into the _state snapshot matrix_ $\mathbf{Q}$ and _input matrix_ $\mathbf{U}$,

\begin{align*}
    \mathbf{Q}
    &= \left[\begin{array}{cccc}
        & & & \\
        \mathbf{q}_{1} & \mathbf{q}_{2} & \cdots & \mathbf{q}_{k}
        \\ & & &
    \end{array}\right]
    \in \mathbb{R}^{n \times k},
    &
    \mathbf{U}
    &= \left[\begin{array}{cccc}
        & & & \\
        \mathbf{u}_{1} & \mathbf{u}_{2} & \cdots & \mathbf{u}_{k}
        \\ & & &
    \end{array}\right]
    \in \mathbb{R}^{m \times k},
\end{align*}

where $n$ is the dimension of the (discretized) state, $m$ is the dimension of the input, $k$ is the number of available data points, and the columns of $\mathbf{Q}$ and $\mathbf{U}$ are the solution to the full-order model at some time $t_j$:

$$
    \frac{\text{d}}{\text{d}t}\mathbf{q}\bigg|_{t = t_j}
    = \mathbf{F}(t_{j}, \mathbf{q}_{j}, \mathbf{u}_{j}).
$$

:::{important}
Raw dynamical systems data often needs to be lightly preprocessed before it can be used in Operator Inference.
Preprocessing can promote stability in the inference of the reduced-order operators and improve the stability and accuracy of the resulting reduced-order model {eq}`eq:opinf-example-rom`.
Common preprocessing steps include
1. Variable transformations / lifting to induce a polynomial structure.
2. Centering or shifting data to account for boundary conditions.
3. Scaling / nondimensionalizing the variables represented in the state.

See [Data Scaling](sec-pre-scaling) for details and examples.
:::

Operator Inference uses a regression problem to compute the reduced-order operators, which requires state data ($\mathbf{Q}$), input data ($\mathbf{U}$), _and_ data for the corresponding time derivatives:

$$
    \dot{\mathbf{Q}}
    = \left[\begin{array}{cccc}
        & & & \\
        \dot{\mathbf{q}}_{1} &
        \dot{\mathbf{q}}_{2} & \cdots &
        \dot{\mathbf{q}}_{k}
        \\ & & &
    \end{array}\right]
    \in \mathbb{R}^{n \times k},
    \qquad
    \dot{\mathbf{q}}_{j} = \frac{\text{d}}{\text{d}t}\mathbf{q}\bigg|_{t = t_j} \in \mathbb{R}^{n}.
$$

:::{note}
If these time derivatives cannot be computed directly by evaluating $\mathbf{F}(t_{j}, \mathbf{q}_{j}, \mathbf{u}_{j})$, they may be inferred from the state snapshots.
The simplest approach is to use [finite differences](https://en.wikipedia.org/wiki/Numerical_differentiation) of the state snapshots, implemented in this package as `opinf.pre.ddt()`.
:::

:::{warning}
If you do any preprocessing on the states, be sure to use the time derivatives of the _processed states_, not of the original states.
:::

<!-- :::{note}
Operator Inference can also be used to learn discrete dynamical systems with polynomial structure, for example,

$$
    \mathbf{q}_{j+1}
    = \mathbf{A}\mathbf{q}_{j}
    + \mathbf{H}(\mathbf{q}_{j}\otimes\mathbf{q}_{j})
    + \mathbf{B}\mathbf{u}_{j}.
$$

In this case, the left-hand side data is a simply subset of the state snapshot matrix.
::: -->
<!-- See TODO for more details. -->

---

(subsec-basis-computation)=
## Compute a Low-dimensional Representation

The purpose of learning a reduced-order model is to achieve a computational speedup.
This is accomplished by introducing an approximate representation of the $n$-dimensional state using only $r \ll n$ degrees of freedom.
The most common approach is to represent the state as a linear combination of $r$ vectors:

$$
    \mathbf{q}(t)
    \approx \mathbf{V}_{r} \widehat{\mathbf{q}}(t)
    = \sum_{i=1}^{r}\mathbf{v}_{i}\hat{q}_{i}(t),
$$ (eq-opinf-basis-def)

where

$$
    \mathbf{V}_{r}
    = \left[\begin{array}{ccc}
        & & \\
        \mathbf{v}_{1} & \cdots & \mathbf{v}_{r}
        \\ & &
    \end{array}\right] \in \mathbb{R}^{n \times r},
    \qquad
    \widehat{\mathbf{q}}
    = \left[\begin{array}{c}
        \hat{q}_{1}(t) \\ \vdots \\ \hat{q}_{r}(t)
    \end{array}\right] \in \mathbb{R}^{r}.
$$

We call $\mathbf{V}_{r} \in \mathbb{R}^{n \times r}$ the _basis matrix_ and typically require that it have orthonormal columns.
The basis matrix is the link between the high-dimensional state space of the full-order model {eq}`eq:opinf-example-fom` and the low-dimensional state space of the reduced-order model {eq}`eq:opinf-example-rom`.

:::{image} ../../images/basis-projection.svg
:align: center
:width: 80 %
:::

See [Basis Computation](sec-basis-computation) for tools to compute the basis $\mathbf{V}_{r}\in\mathbb{R}^{n \times r}$ and select an appropriate dimension $r$.

<!-- :::{tip}
In the case of finite differences, the time derivative estimation can be done after the data is projected to the low-dimensional subspace defined by the basis (the column space of $\mathbf{V}_{r}$).
Instead of feeding the data matrix $\mathbf{Q}$ to `opinf.pre.ddt()`, consider computing $\widehat{\mathbf{Q}} = \mathbf{V}_{r}^{\mathsf{T}}\mathbf{Q}$ first and using that as the input to `opinf.pre.ddt()`.
You can also use $\widehat{\mathbf{Q}}$ as the input when you fit the reduced-order model object.
::: -->

---

(subsec-opinf-regression)=
## Set up and Solve a Low-dimensional Regression

Operator Inference determines the operators $\widehat{\mathbf{c}}$, $\widehat{\mathbf{A}}$, $\widehat{\mathbf{H}}$, and/or $\widehat{\mathbf{B}}$ by solving the following data-driven regression:

$$
\min_{\widehat{\mathbf{c}},\widehat{\mathbf{A}},\widehat{\mathbf{H}},\widehat{\mathbf{B}}}\sum_{j=0}^{k-1}\left\|
    \widehat{\mathbf{c}}
    + \widehat{\mathbf{A}}\widehat{\mathbf{q}}_{j}
    + \widehat{\mathbf{H}}[\widehat{\mathbf{q}}_{j} \otimes \widehat{\mathbf{q}}_{j}]
    + \widehat{\mathbf{B}}\mathbf{u}_{j}
    - \dot{\widehat{\mathbf{q}}}_{j}
\right\|_{2}^{2}
+ \mathcal{R}(\widehat{\mathbf{c}},\widehat{\mathbf{A}},\widehat{\mathbf{H}},\widehat{\mathbf{B}}),
$$ (eq:opinf-lstsq-residual)

where
- $\widehat{\mathbf{q}}_{j} = \mathbf{V}_{r}^{\mathsf{T}}\mathbf{q}(t_{j})$ is the state at time $t_{j}$ represented in the coordinates of the basis,
- $\dot{\widehat{\mathbf{q}}}_{j} = \frac{\textrm{d}}{\textrm{d}t}\mathbf{V}_{r}^{\mathsf{T}}\mathbf{q}\big|_{t=t_{j}}$ is the time derivative of the state at time $t_{j}$ in the coordinates of the basis,
- $\mathbf{u}_{j} = \mathbf{u}(t_j)$ is the input at time $t_{j}$, and
- $\mathcal{R}$ is a _regularization term_ that penalizes the entries of the learned operators.


The least-squares minimization {eq}`eq:opinf-lstsq-residual` can be written in the more standard form

$$
\min_{\widehat{\mathbf{O}}}\left\|
    \mathbf{D}\widehat{\mathbf{O}}^{\mathsf{T}} - \mathbf{Y}^{\mathsf{T}}
\right\|_{F}^{2} + \mathcal{R}(\widehat{\mathbf{O}}),
$$

where

::::{margin}
:::{note}
The $\odot$ operator is called the [Khatri-Rao product](https://en.wikipedia.org/wiki/Khatri%E2%80%93Rao_product#Column-wise_Kronecker_product) and indicates taking the Kronecker product column by column.
<!-- $$
\left[\begin{array}{cccc}
    \mathbf{z}_{0} & \mathbf{z}_{1} & \cdots & \mathbf{z}_{k-1}
\end{array}\right]
\odot
\left[\begin{array}{cccc}
    \mathbf{w}_{0} & \mathbf{w}_{1} & \cdots & \mathbf{w}_{k-1}
\end{array}\right]
= \left[\begin{array}{cccc}
    \mathbf{z}_{0}\otimes\mathbf{w}_{0} & \mathbf{z}_{1}\otimes\mathbf{w}_{1} & \cdots & \mathbf{z}_{k-1}\otimes\mathbf{w}_{k-1}
\end{array}\right].
$$ -->
:::
::::

\begin{align*}
    \widehat{\mathbf{O}}
    &= \left[~\widehat{\mathbf{c}}~~\widehat{\mathbf{A}}~~\widehat{\mathbf{H}}~~\widehat{\mathbf{B}}~\right]\in\mathbb{R}^{r\times d(r,m)},
    &\text{(unknown operators)}
    \\
    \mathbf{D}
    &= \left[~\mathbf{1}_{k}~~\widehat{\mathbf{Q}}^{\mathsf{T}}~~(\widehat{\mathbf{Q}}\odot\widehat{\mathbf{Q}})^{\mathsf{T}}~~\mathbf{U}^{\mathsf{T}}~\right]\in\mathbb{R}^{k\times d(r,m)},
    &\text{(known data)}
    \\
    \widehat{\mathbf{Q}}
    &= \left[~\widehat{\mathbf{q}}_0~~\widehat{\mathbf{q}}_1~~\cdots~~\widehat{\mathbf{q}}_{k-1}~\right]\in\mathbb{R}^{r\times k}
    &\text{(snapshots)}
    \\
    \mathbf{Y}
    &= \left[~\dot{\widehat{\mathbf{q}}}_0~~\dot{\widehat{\mathbf{q}}}_1~~\cdots~~\dot{\widehat{\mathbf{q}}}_{k-1}~\right]\in\mathbb{R}^{r\times k},
    &\text{(time derivatives)}
    \\
    \mathbf{U}
    &= \left[~\mathbf{u}_0~\mathbf{u}_1~\cdots~\mathbf{u}_{k-1}~\right]\in\mathbb{R}^{m\times k},
    &\text{(inputs)}
\end{align*}

in which $d(r,m) = 1 + r + r(r+1)/2 + m$ and $\mathbf{1}_{k}\in\mathbb{R}^{k}$ is a vector of ones.

:::{dropdown} Derivation
The Frobenius norm of a matrix is the square root of the sum of the squared entries.
If $\mathbf{Z}$ has entries $z_{ij}$, then

$$
\|\mathbf{Z}\|_{F}
= \sqrt{\text{trace}(\mathbf{Z}^{\mathsf{T}}\mathbf{Z})}
= \sqrt{\sum_{i,j}z_{ij}^{2}}.
$$

Writing $\mathbf{Z}$ in terms of its columns,

$$
\mathbf{Z}
= \left[\begin{array}{c|c|c|c}
    &&& \\
    \mathbf{z}_0 & \mathbf{z}_1 & \cdots & \mathbf{z}_{k-1} \\
    &&&
\end{array}\right],
$$

we have

$$
\|\mathbf{Z}\|_F^2 = \sum_{j=0}^{k-1}\|\mathbf{z}_j\|_2^2.
$$

Furthermore, $\|\mathbf{Z}\|_{F} = \|\mathbf{Z}^{\mathsf{T}}\|_{F}$.
Using these two properties, we can rewrite the least-squares residual as follows:

\begin{align*}
    \sum_{j=0}^{k-1}\left\|
        \widehat{\mathbf{c}}
        + \widehat{\mathbf{A}}\widehat{\mathbf{q}}_{j}
        + \widehat{\mathbf{H}}[\widehat{\mathbf{q}}_{j} \otimes \widehat{\mathbf{q}}_{j}]
        + \widehat{\mathbf{B}}\mathbf{u}_{j}
        - \dot{\widehat{\mathbf{q}}}_{j}
    \right\|_{2}^{2}
    &= \left\|
        \widehat{\mathbf{c}}\mathbf{1}^{\mathsf{T}}
        + \widehat{\mathbf{A}}\widehat{\mathbf{Q}}
        + \widehat{\mathbf{H}}[\widehat{\mathbf{Q}} \odot \widehat{\mathbf{Q}}]
        + \widehat{\mathbf{B}}\mathbf{U}
        - \dot{\widehat{\mathbf{Q}}}
    \right\|_{F}^{2}
    \\
    &= \left\|
        \left[\begin{array}{cccc}
            \widehat{\mathbf{c}} & \widehat{\mathbf{A}} & \widehat{\mathbf{H}} & \widehat{\mathbf{B}}
        \end{array}\right]
        \left[\begin{array}{c}
            \mathbf{1}^{\mathsf{T}}
            \\ \widehat{\mathbf{Q}}
            \\ \widehat{\mathbf{Q}} \odot \widehat{\mathbf{Q}}
            \\ \mathbf{U}
        \end{array}\right]
        - \dot{\widehat{\mathbf{Q}}}
    \right\|_{F}^{2}
    \\
    &= \left\|
        \left[\begin{array}{c}
            \mathbf{1}^{\mathsf{T}}
            \\ \widehat{\mathbf{Q}}
            \\ \widehat{\mathbf{Q}} \odot \widehat{\mathbf{Q}}
            \\ \mathbf{U}
        \end{array}\right]^{\mathsf{T}}
        \left[\begin{array}{cccc}
            \widehat{\mathbf{c}} & \widehat{\mathbf{A}} & \widehat{\mathbf{H}} & \widehat{\mathbf{B}}
        \end{array}\right]^{\mathsf{T}}
        - \dot{\widehat{\mathbf{Q}}}^{\mathsf{T}}
    \right\|_{F}^{2}
    \\
    &= \left\|
        \left[\begin{array}{cccc}
            \mathbf{1}
            & \widehat{\mathbf{Q}}^{\mathsf{T}}
            & [\widehat{\mathbf{Q}} \odot \widehat{\mathbf{Q}}]^{\mathsf{T}}
            & \mathbf{U}^{\mathsf{T}}
        \end{array}\right]
        \left[\begin{array}{c}
            \widehat{\mathbf{c}}^{\mathsf{T}}
            \\ \widehat{\mathbf{A}}^{\mathsf{T}}
            \\ \widehat{\mathbf{H}}^{\mathsf{T}}
            \\ \widehat{\mathbf{B}}^{\mathsf{T}}
        \end{array}\right]
        - \dot{\widehat{\mathbf{Q}}}^{\mathsf{T}}
    \right\|_{F}^{2},
\end{align*}

which is the standard form given above.
:::

:::{important}
Writing the problem in standard form reveals an important fact:
for the most common choices of $\mathcal{R}$, the Operator Inference learning problem {eq}`eq:opinf-lstsq-residual` has a unique solution if and only if $\mathbf{D}$ has full column rank.
A necessary condition for this to happen is $k \ge d(r,m)$, that is, the number of training snapshots $k$ should exceed the number of reduced-order operator entries to be learned for each system mode.
If you are experiencing poor performance with Operator Inference reduced models, try decreasing $r$, increasing $k$, or adding a regularization term to improve the conditioning of the learning problem.
:::

:::{note}
Let $\widehat{\mathbf{o}}_{1},\ldots,\widehat{\mathbf{o}}_{r}\in\mathbb{R}^{d(r,m)}$ be the rows of $\widehat{\mathbf{O}}$.
If the regularization can be written as

$$
\mathcal{R}(\widehat{\mathbf{O}})
= \sum_{i=1}^{r}\mathcal{R}_{i}(\widehat{\mathbf{o}}_{i}),
$$

then the Operator Inference regression decouples along the rows of $\widehat{\mathbf{O}}$ into $r$ independent least-squares problems:

$$
\min_{\widehat{\mathbf{O}}}\left\{\left\|
    \mathbf{D}\widehat{\mathbf{O}}^{\mathsf{T}} - \mathbf{Y}^{\mathsf{T}}
\right\|_{F}^{2} + \mathcal{R}(\widehat{\mathbf{O}})\right\}
=
\sum_{i=1}^{r}\min_{\widehat{\mathbf{o}}_{i}}\left\{\left\|
    \mathbf{D}\widehat{\mathbf{o}}_{i} - \mathbf{y}_{i}
\right\|_{2}^{2} + \mathcal{R}_{i}(\widehat{\mathbf{o}}_{i})\right\},
$$

where $\mathbf{y}_{i},\ldots,\mathbf{y}_{r}$ are the rows of $\mathbf{Y}$.
:::

---

(subsec-rom-evaluation)=
## Evaluate the Reduced-order Model

Once the reduced-order operators have been determined, the corresponding reduced-order model {eq}`eq:opinf-example-rom` can be solved rapidly to make predictions.
The computational cost of solving the reduced-order model scales with $r$, the number of degrees of freedom in the low-dimensional representation of the state.

For example, we may use the reduced-order model to obtain approximate solutions of the full-order model {eq}`eq:opinf-example-fom` with
- new initial conditions $\mathbf{q}_{0}$,
- a different input function $\mathbf{u}(t)$,
- a longer time horizon than the training data,
- different system parameters (see [Parametric ROMs](subsec-parametric-roms)).

:::{important}
The accuracy of any data-driven model depends on how well the training data represents the full-order system.
We should not expect a reduced-order model to perform well under conditions that are wildly different than the training data.
The [**Getting Started**](sec-tutorial) tutorial demonstrates this concept in the case of prediction for new initial conditions.
:::

---

## Brief Example

Suppose we have the following variables.

| Variable | Symbol | Description |
| :------- | :----- | :---------- |
| `Q` | $\mathbf{Q}\in\mathbb{R}^{n\times k}$ | State snapshot matrix |
| `U` | $\mathbf{U}\in\mathbb{R}^{m}$ | Input matrix |
| `t` | $t$ | Time domain for snapshot data |

That is, `Q[:, j]` and `U[:, j]` are the state and input, respectively, corresponding to time `t[j]`.
Then the following code learns a reduced-order model of the form

$$
    \frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t)
    = \widehat{\mathbf{A}}\widehat{\mathbf{q}}(t)
    + \widehat{\mathbf{H}}(\widehat{\mathbf{q}}(t)\otimes\widehat{\mathbf{q}}(t))
    + \widehat{\mathbf{B}}\widehat{\mathbf{u}}(t)
$$

from the training data, uses the reduced-order model to reconstruct the training data, and computes the error of the reconstruction.

```python
import opinf

# Compute a rank-10 basis (POD) from the state data.
>>> basis = opinf.pre.PODBasis(Q, r=10)

# Estimate time derivatives of the state with finite differences.
>>> Qdot = opinf.pre.ddt(Q, t)

# Define a reduced-order model with the structure indicated above.
>>> rom = opinf.ContinuousOpInfROM(modelform="AHB")

# Select a least-squares solver with a small amount of regularization.
>>> solver = opinf.lstsq.L2Solver(regularizer=1e-6)

# Fit the model, i.e., construct and solve a linear regression.
>>> rom.fit(basis=basis, states=Q, ddts=Qdot, inputs=U, solver=solver)

# Simulate the learned model over the time domain.
>>> Q_ROM = rom.predict(Q[:, 0], t)

# Compute the error of the ROM prediction.
>>> absolute_error, relative_error = opinf.post.Lp_error(Q, Q_rom)
```

See [**Getting Started**](sec-tutorial) for a tutorial.
