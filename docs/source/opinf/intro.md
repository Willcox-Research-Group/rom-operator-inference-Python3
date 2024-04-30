# What is Operator Inference?

:::{image} ../../images/summary.svg
:align: center
:width: 80 %
:::

The goal of Operator Inference is to construct a low-dimensional, computationally inexpensive system whose solutions are close to those of some high-dimensional system for which we have 1) training data and 2) some knowledge about the system structure.
The main steps are the following.

1. [**Get training data**](subsec-training-data). Gather and [preprocess](../api/pre.ipynb) high-dimensional data to learn a low-dimensional model from. This package has a few common preprocessing tools, but the user must bring the data to the table.
2. [**Compute a low-dimensional representation**](subsec-basis-computation). Approximate the high-dimensional data with only a few degrees of freedom. The simplest approach is to take the SVD of the high-dimensional training data, extract the first few left singular vectors, and use these vectors as a new coordinate basis.
3. [**Set up and solve a low-dimensional regression**](subsec-opinf-regression). Use the low-dimensional representation of the training data to determine a reduced-order model that best fits the data in a minimum-residual sense. This is the core objective of the package.
4. [**Evaluate the reduced-order model**](subsec-rom-evaluation). Use the learned model to make computationally efficient predictions.

This page gives an overview of Operator Inference by walking through each of these steps.

---

## Problem Statement

::::{margin}
:::{note}
We are most often interested in full-order models that are spatial discretizations of partial differential equations (PDEs), but {eq}`eq:opinf-example-fom` does not necessarily have to be related to a PDE.
:::
::::

Consider a system of ordinary differential equations (ODEs) with state $\q(t)\in\RR^{n}$ and inputs $\u(t)\in\RR^{m}$,

$$
    \frac{\text{d}}{\text{d}t}\q(t)
    = \mathbf{F}(t, \q(t), \u(t)).
$$ (eq:opinf-example-fom)

We call {eq}`eq:opinf-example-fom` the _full-order model_.
Given samples of the state $\q(t)$, Operator Inference learns a surrogate system for {eq}`eq:opinf-example-fom` with the much smaller state $\qhat(t) \in \RR^{r}, r \ll n,$ and a polynomial structure, for example:

::::{margin}
:::{note}
The $\otimes$ operator is called the [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product).
:::
::::

$$
    \frac{\text{d}}{\text{d}t}\qhat(t)
    = \chat
    + \Ahat\qhat(t)
    + \Hhat[\qhat(t)\otimes\qhat(t)]
    + \Bhat\u(t).
$$ (eq:opinf-example-rom)

We call {eq}`eq:opinf-example-rom` a _reduced-order model_ for {eq}`eq:opinf-example-fom`.
Our goal is to infer the _reduced-order operators_ $\chat \in \RR^{r}$, $\Ahat\in\RR^{r\times r}$, $\Hhat\in\RR^{r\times r^{2}}$, and/or $\Bhat\in\RR^{r\times m}$ using data from {eq}`eq:opinf-example-fom`.
The user specifies which terms to include in the model.

::::{important}
:name: projection-preserves-structure

The right-hand side of {eq}`eq:opinf-example-rom` is a polynomial with respect to the state $\qhat(t)$:
$\chat$ are the constant terms, $\Ahat\qhat(t)$ are the linear terms, $\Hhat[\qhat(t)\otimes\qhat(t)]$ are the quadratic terms, with input terms $\Bhat\u(t)$.
The user must choose which terms to include in the reduced-order model, and this choice should be motivated by the structure of the full-order model {eq}`eq:opinf-example-fom`.
For example, if the full-order model can be written as

$$
    \frac{\text{d}}{\text{d}t}\q(t)
    = \A\q(t) + \B\u(t),
$$

then the reduced-order model should mirror this structure as

$$
    \frac{\text{d}}{\text{d}t}\qhat(t)
    = \Ahat\qhat(t)
    + \Bhat\u(t).
$$

:::{dropdown} Motivation
**Projection preserves polynomial structure** {cite}`benner2015pmorsurvey,peherstorfer2016opinf`.
The classical (Galerkin) projection-based reduced-order model for the system

$$
    \frac{\text{d}}{\text{d}t}\q(t)
    = \c
    + \A\q(t)
    + \H[\q(t)\otimes\q(t)]
    + \B\u(t)
$$

is obtained by substituting $\q(t)$ with $\mathbf{V}\qhat(t)$ for some orthonormal $\mathbf{V}\in\RR^{n \times r}$ and $\qhat(t)\in \RR^{r}$, then multiplying both sides by $\mathbf{V}\trp$:

$$
    \mathbf{V}\trp\frac{\text{d}}{\text{d}t}\left[\mathbf{V}\qhat(t)\right]
    = \mathbf{V}\trp\left(\c
    + \A\mathbf{V}\qhat(t)
    + \H[(\mathbf{V}\qhat(t))\otimes(\mathbf{V}\qhat(t))]
    + \B\u(t)\right).
$$

Since $\mathbf{V}\trp\mathbf{V}$ is the identity and $(\mathbf{X}\mathbf{Y})\otimes(\mathbf{Z}\mathbf{W}) = (\mathbf{X}\otimes \mathbf{Z})(\mathbf{Y}\otimes\mathbf{W})$, this simplifies to

$$
    \frac{\text{d}}{\text{d}t}\qhat(t)
    =
    \chat
    + \Ahat\qhat(t)
    + \Hhat[\qhat(t)\otimes\qhat(t)]
    + \Bhat\u(t),
$$

where $\chat = \mathbf{V}\trp\c$, $\Ahat = \mathbf{V}\trp\A\mathbf{V}$, $\Hhat = \mathbf{V}\trp\H\left(\mathbf{V}\otimes\mathbf{V}\trp\right)$, and $\Bhat = \mathbf{V}\trp\B$.
Operator Inference learns $\chat$, $\Ahat$, $\Hhat$, and/or $\Bhat$ _from data_ and is therefore useful for situations where $\c$, $\A$, $\H$, and/or $\B$ are not explicitly available for matrix computations.
:::
::::

---

(subsec-training-data)=
## Get Training Data

Operator Inference learns reduced-order models from full-order state/input data.
Start by gathering solution and input data and organizing them columnwise into the _state snapshot matrix_ $\Q$ and _input matrix_ $\U$,

\begin{align*}
    \Q
    &= \left[\begin{array}{cccc}
        & & & \\
        \q_{1} & \q_{2} & \cdots & \q_{k}
        \\ & & &
    \end{array}\right]
    \in \RR^{n \times k},
    &
    \U
    &= \left[\begin{array}{cccc}
        & & & \\
        \u_{1} & \u_{2} & \cdots & \u_{k}
        \\ & & &
    \end{array}\right]
    \in \RR^{m \times k},
\end{align*}

where $n$ is the dimension of the (discretized) state, $m$ is the dimension of the input, $k$ is the number of available data points, and the columns of $\Q$ and $\U$ are the solution to the full-order model at some time $t_j$:

$$
    \frac{\text{d}}{\text{d}t}\q\bigg|_{t = t_j}
    = \mathbf{F}(t_{j}, \q_{j}, \u_{j}).
$$

:::{important}
Raw dynamical systems data often needs to be lightly preprocessed before it can be used in Operator Inference.
Preprocessing can promote stability in the inference of the reduced-order operators and improve the stability and accuracy of the resulting reduced-order model {eq}`eq:opinf-example-rom`.
Common preprocessing steps include
1. Variable transformations / lifting to induce a polynomial structure.
2. Centering or shifting data to account for boundary conditions.
3. Scaling / nondimensionalizing the variables represented in the state.

See [`opinf.pre`](../api/pre.ipynb) for details and examples.
:::

Operator Inference uses a regression problem to compute the reduced-order operators, which requires state data ($\Q$), input data ($\U$), _and_ data for the corresponding time derivatives:

$$
    \dot{\Q}
    = \left[\begin{array}{cccc}
        & & & \\
        \dot{\q}_{1} &
        \dot{\q}_{2} & \cdots &
        \dot{\q}_{k}
        \\ & & &
    \end{array}\right]
    \in \RR^{n \times k},
    \qquad
    \dot{\q}_{j} = \frac{\text{d}}{\text{d}t}\q\bigg|_{t = t_j} \in \RR^{n}.
$$

:::{note}
If these time derivatives cannot be computed directly by evaluating $\mathbf{F}(t_{j}, \q_{j}, \u_{j})$, they may be inferred from the state snapshots.
The simplest approach is to use [finite differences](https://en.wikipedia.org/wiki/Numerical_differentiation) of the state snapshots, implemented in this package as `opinf.pre.ddt()`.
:::

:::{warning}
If you do any preprocessing on the states, be sure to use the time derivatives of the _processed states_, not of the original states.
:::

<!-- :::{note}
Operator Inference can also be used to learn discrete dynamical systems with polynomial structure, for example,

$$
    \q_{j+1}
    = \A\q_{j}
    + \H(\q_{j}\otimes\q_{j})
    + \B\u_{j}.
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
    \q(t)
    \approx \Vr \qhat(t)
    = \sum_{i=1}^{r}\v_{i}\hat{q}_{i}(t),
$$ (eq-opinf-basis-def)

where

$$
    \Vr
    = \left[\begin{array}{ccc}
        & & \\
        \v_{1} & \cdots & \v_{r}
        \\ & &
    \end{array}\right] \in \RR^{n \times r},
    \qquad
    \qhat
    = \left[\begin{array}{c}
        \hat{q}_{1}(t) \\ \vdots \\ \hat{q}_{r}(t)
    \end{array}\right] \in \RR^{r}.
$$

We call $\Vr \in \RR^{n \times r}$ the _basis matrix_ and typically require that it have orthonormal columns.
The basis matrix is the link between the high-dimensional state space of the full-order model {eq}`eq:opinf-example-fom` and the low-dimensional state space of the reduced-order model {eq}`eq:opinf-example-rom`.

:::{image} ../../images/basis-projection.svg
:align: center
:width: 80 %
:::

See {mod}`opinf.basis` for tools to compute the basis $\Vr\in\RR^{n \times r}$ and select an appropriate dimension $r$.

<!-- :::{tip}
In the case of finite differences, the time derivative estimation can be done after the data is projected to the low-dimensional subspace defined by the basis (the column space of $\Vr$).
Instead of feeding the data matrix $\Q$ to `opinf.pre.ddt()`, consider computing $\widehat{\Q} = \Vr\trp\Q$ first and using that as the input to `opinf.pre.ddt()`.
You can also use $\widehat{\Q}$ as the input when you fit the reduced-order model object.
::: -->

---

(subsec-opinf-regression)=
## Set up and Solve a Low-dimensional Regression

Operator Inference determines the operators $\chat$, $\Ahat$, $\Hhat$, and/or $\Bhat$ by solving the following data-driven regression:

$$
\min_{\chat,\Ahat,\Hhat,\Bhat}\sum_{j=0}^{k-1}\left\|
    \chat
    + \Ahat\qhat_{j}
    + \Hhat[\qhat_{j} \otimes \qhat_{j}]
    + \Bhat\u_{j}
    - \dot{\qhat}_{j}
\right\|_{2}^{2}

+ \mathcal{R}(\chat,\Ahat,\Hhat,\Bhat),

$$ (eq:opinf-lstsq-residual)

where
+ $\qhat_{j} = \Vr\trp\q(t_{j})$ is the state at time $t_{j}$ represented in the coordinates of the basis,
+ $\dot{\qhat}_{j} = \ddt\Vr\trp\q\big|_{t=t_{j}}$ is the time derivative of the state at time $t_{j}$ in the coordinates of the basis,
+ $\u_{j} = \u(t_j)$ is the input at time $t_{j}$, and
+ $\mathcal{R}$ is a _regularization term_ that penalizes the entries of the learned operators.

The least-squares minimization {eq}`eq:opinf-lstsq-residual` can be written in the more standard form

$$
\min_{\Ohat}\left\|
    \D\Ohat\trp - \mathbf{Y}\trp
\right\|_{F}^{2} + \mathcal{R}(\Ohat),
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
    \Ohat
    &= \left[~\chat~~\Ahat~~\Hhat~~\Bhat~\right]\in\RR^{r\times d(r,m)},
    &\text{(unknown operators)}
    \\
    \D
    &= \left[~\mathbf{1}_{k}~~\widehat{\Q}\trp~~(\widehat{\Q}\odot\widehat{\Q})\trp~~\U\trp~\right]\in\RR^{k\times d(r,m)},
    &\text{(known data)}
    \\
    \widehat{\Q}
    &= \left[~\qhat_0~~\qhat_1~~\cdots~~\qhat_{k-1}~\right]\in\RR^{r\times k}
    &\text{(snapshots)}
    \\
    \mathbf{Y}
    &= \left[~\dot{\qhat}_0~~\dot{\qhat}_1~~\cdots~~\dot{\qhat}_{k-1}~\right]\in\RR^{r\times k},
    &\text{(time derivatives)}
    \\
    \U
    &= \left[~\u_0~\u_1~\cdots~\u_{k-1}~\right]\in\RR^{m\times k},
    &\text{(inputs)}
\end{align*}

in which $d(r,m) = 1 + r + r(r+1)/2 + m$ and $\mathbf{1}_{k}\in\RR^{k}$ is a vector of ones.

:::{dropdown} Derivation
The Frobenius norm of a matrix is the square root of the sum of the squared entries.
If $\mathbf{Z}$ has entries $z_{ij}$, then

$$
\|\mathbf{Z}\|_{F}
= \sqrt{\text{trace}(\mathbf{Z}\trp\mathbf{Z})}
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

Furthermore, $\|\mathbf{Z}\|_{F} = \|\mathbf{Z}\trp\|_{F}$.
Using these two properties, we can rewrite the least-squares residual as follows:

\begin{align*}
    \sum_{j=0}^{k-1}\left\|
        \chat
        + \Ahat\qhat_{j}
        + \Hhat[\qhat_{j} \otimes \qhat_{j}]
        + \Bhat\u_{j}
        - \dot{\qhat}_{j}
    \right\|_{2}^{2}
    &= \left\|
        \chat\mathbf{1}\trp
        + \Ahat\widehat{\Q}
        + \Hhat[\widehat{\Q} \odot \widehat{\Q}]
        + \Bhat\U
        - \dot{\widehat{\Q}}
    \right\|_{F}^{2}
    \\
    &= \left\|
        \left[\begin{array}{cccc}
            \chat & \Ahat & \Hhat & \Bhat
        \end{array}\right]
        \left[\begin{array}{c}
            \mathbf{1}\trp
            \\ \widehat{\Q}
            \\ \widehat{\Q} \odot \widehat{\Q}
            \\ \U
        \end{array}\right]
        - \dot{\widehat{\Q}}
    \right\|_{F}^{2}
    \\
    &= \left\|
        \left[\begin{array}{c}
            \mathbf{1}\trp
            \\ \widehat{\Q}
            \\ \widehat{\Q} \odot \widehat{\Q}
            \\ \U
        \end{array}\right]\trp
        \left[\begin{array}{cccc}
            \chat & \Ahat & \Hhat & \Bhat
        \end{array}\right]\trp
        - \dot{\widehat{\Q}}\trp
    \right\|_{F}^{2}
    \\
    &= \left\|
        \left[\begin{array}{cccc}
            \mathbf{1}
            & \widehat{\Q}\trp
            & [\widehat{\Q} \odot \widehat{\Q}]\trp
            & \U\trp
        \end{array}\right]
        \left[\begin{array}{c}
            \chat\trp
            \\ \Ahat\trp
            \\ \Hhat\trp
            \\ \Bhat\trp
        \end{array}\right]
        - \dot{\widehat{\Q}}\trp
    \right\|_{F}^{2},
\end{align*}

which is the standard form given above.
:::

:::{important}
Writing the problem in standard form reveals an important fact:
for the most common choices of $\mathcal{R}$, the Operator Inference learning problem {eq}`eq:opinf-lstsq-residual` has a unique solution if and only if $\D$ has full column rank.
A necessary condition for this to happen is $k \ge d(r,m)$, that is, the number of training snapshots $k$ should exceed the number of reduced-order operator entries to be learned for each system mode.
If you are experiencing poor performance with Operator Inference reduced models, try decreasing $r$, increasing $k$, or adding a regularization term to improve the conditioning of the learning problem.
:::

:::{note}
Let $\ohat_{1},\ldots,\ohat_{r}\in\RR^{d(r,m)}$ be the rows of $\Ohat$.
If the regularization can be written as

$$
\begin{align*}
    \mathcal{R}(\Ohat)
    = \sum_{i=1}^{r}\mathcal{R}_{i}(\ohat_{i}),
\end{align*}
$$

then the Operator Inference regression decouples along the rows of $\Ohat$ into $r$ independent least-squares problems:

$$
\begin{align*}
    \min_{\Ohat}\left\{\left\|
        \D\Ohat\trp - \mathbf{Y}\trp
    \right\|_{F}^{2} + \mathcal{R}(\Ohat)\right\}
    =
    \sum_{i=1}^{r}\min_{\ohat_{i}}\left\{\left\|
        \D\ohat_{i} - \mathbf{y}_{i}
    \right\|_{2}^{2} + \mathcal{R}_{i}(\ohat_{i})\right\},
\end{align*}
$$

where $\mathbf{y}_{i},\ldots,\mathbf{y}_{r}$ are the rows of $\mathbf{Y}$.
:::

---

(subsec-rom-evaluation)=
## Solve the Reduced-order Model

Once the reduced-order operators have been determined, the corresponding reduced-order model {eq}`eq:opinf-example-rom` can be solved rapidly to make predictions.
The computational cost of solving the reduced-order model scales with $r$, the number of degrees of freedom in the low-dimensional representation of the state.

For example, we may use the reduced-order model to obtain approximate solutions of the full-order model {eq}`eq:opinf-example-fom` with

+ new initial conditions $\q_{0}$,
+ a different input function $\u(t)$,
+ a longer time horizon than the training data,
+ different system parameters.

:::{important}
The accuracy of any data-driven model depends on how well the training data represents the full-order system.
We should not expect a reduced-order model to perform well under conditions that are wildly different than the training data.
The [**Getting Started**](../tutorials/basics.ipynb) tutorial demonstrates this concept in the case of prediction for new initial conditions.
:::

---

## Brief Example

Suppose we have the following variables.

| Variable | Symbol | Description |
| :------- | :----- | :---------- |
| `Q` | $\Q\in\RR^{n\times k}$ | State snapshot matrix |
| `U` | $\U\in\RR^{m}$ | Input matrix |
| `t` | $t$ | Time domain for snapshot data |

That is, `Q[:, j]` and `U[:, j]` are the state and input, respectively, corresponding to time `t[j]`.
Then the following code learns a reduced-order model of the form

$$
    \frac{\text{d}}{\text{d}t}\qhat(t)
    = \Ahat\qhat(t)
    + \Hhat(\qhat(t)\otimes\qhat(t))
    + \Bhat\widehat{\u}(t)
$$

from the training data, uses the reduced-order model to reconstruct the training data, and computes the error of the reconstruction.

```python
import opinf

# Compute a rank-10 basis (POD) from the state data.
>>> basis = opinf.pre.PODBasis(Q, r=10)

# Compress the state data to a low-dimensional subspace.
>>> Q_compressed = basis.compress(Q)

# Estimate time derivatives of the compressed states with finite differences.
>>> Qdot_compressed = opinf.ddt.ddt(Q_compressed, t)

# Define an ODE model with the structure indicated above.
>>> rom = opinf.models.ContinuousModel("AHB")

# Select a least-squares solver with a small amount of regularization.
>>> solver = opinf.lstsq.L2Solver(regularizer=1e-6)

# Fit the model, i.e., construct and solve a linear regression.
>>> rom.fit(states=Q_compressed, ddts=Qdot_compressed, inputs=U, solver=solver)

# Simulate the learned model over the time domain.
>>> Q_rom_compressed = rom.predict(Q_compressed[:, 0], t)

# Map the reduced-order solutions back to the full state space.
>>> Q_rom = basis.decompress(Q_rom_compressed)

# Compute the error of the ROM prediction.
>>> absolute_error, relative_error = opinf.post.Lp_error(Q, Q_rom)
```

See [**Getting Started**](../tutorials/basics.ipynb) for an introductory tutorial.
