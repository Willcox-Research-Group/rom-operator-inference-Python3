(sec-opinf-overview)=
# What is Operator Inference?

Operator Inference is a projection-based model reduction technique that learns reduced-order models from data.
The goal is to construct a low-dimensional, computationally inexpensive system whose solutions are close to those of some high-dimensional system for which we have 1) training data and 2) some knowledge about the system structure.
The main steps are the following.

1. [**You Provide Training Data**](subsec-training-data). Gather high-dimensional data to learn from and do any [preprocessing](sec-preprocessing) to prepare for model learning.
2. [**Compute a Low-dimensional Basis**](subsec-basis-computation). Take the SVD of the training data and extract the first few left singular vectors.
3. [**Set up and Solve a Low-dimensional Regression**](subsec-opinf-regression). This is the core objective of this package.
4. [**Evaluate the Reduced-order Model**](subsec-rom-evaluation). Use the learned model to do prediction.

This page reviews each of these steps and shows how to do them with this package.

:::{image} ../../images/opinf-summary.svg
:align: center
:width: 80 %
:::

---

## Problem Statement

Consider a system of ODEs with state $\mathbf{q}(t)\in\mathbb{R}^{n}$ and inputs $\mathbf{u}(t)\in\mathbb{R}^{m}$,

$$
    \frac{\text{d}}{\text{d}t}\mathbf{q}(t)
    = \mathbf{f}(t, \mathbf{q}(t), \mathbf{u}(t)).
$$ (eq:opinf-example-fom)

We call {eq}`eq:opinf-example-fom` the _full-order model_, which often represents a PDE after spatial discretization.
Given samples of the state $\mathbf{q}(t)$, Operator Inference learns a surrogate system for {eq}`eq:opinf-example-fom` with the much smaller state $\widehat{\mathbf{q}}(t) \in \mathbb{R}^{r}, r \ll n$, and the following polynomial structure:

$$
    \frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t)
    = \widehat{\mathbf{c}}
    + \widehat{\mathbf{A}}\widehat{\mathbf{q}}(t)
    + \widehat{\mathbf{H}}(\widehat{\mathbf{q}}(t)\otimes\widehat{\mathbf{q}}(t))
    + \widehat{\mathbf{B}}\widehat{\mathbf{u}}(t).
$$ (eq:opinf-example-rom)

We call {eq}`eq:opinf-example-rom` the _reduced-order model_.
Our goal is to infer the _reduced-order operators_ $\widehat{\mathbf{c}}$, $\widehat{\mathbf{A}}$, $\widehat{\mathbf{H}}$, and $\widehat{\mathbf{B}}$ using data from {eq}`eq:opinf-example-fom`.

:::{important}
The right-hand side of {eq}`eq:opinf-example-rom` has a polynomial structure with respect to the state:
$\widehat{\mathbf{c}}$ are constant terms, $\widehat{\mathbf{A}}\widehat{\mathbf{q}}(t)$ are the linear terms, $\widehat{\mathbf{H}}(\widehat{\mathbf{q}}(t)\otimes\widehat{\mathbf{q}}(t))$ are quadratic terms.
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
    + \widehat{\mathbf{B}}\widehat{\mathbf{u}}(t).
$$
:::

---

(subsec-training-data)=
## You Provide Training Data

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

where $n$ is the dimension of the state discretization, $m$ is the dimension of the input, $k$ is the number of available data points, and the columns of $\mathbf{Q}$ and $\mathbf{U}$ are the solution to the full-order model at some time $t_j$:

$$
    \frac{\text{d}}{\text{d}t}\mathbf{q}\bigg|_{t = t_j}
    = \mathbf{f}(t_{j}, \mathbf{q}_{j}, \mathbf{u}_{j}).
$$

:::{note}
Raw dynamical systems data often needs to be lightly preprocessed in order to promote stability in the inference problem for learning the reduced-order operators, and to improve the stability and accuracy of the resulting reduced-order model {eq}`eq:opinf-example-rom`.
Common preprocessing steps include
1. Variable transformations / lifting to induce a polynomial structure.
2. Centering or shifting data to account for boundary conditions.
3. Scaling / nondimensionalizing the variables represented in the state.

See [the preprocessing guide](sec-preprocessing) for details and examples.
:::

Operator Inference uses a regression problem to compute the reduced-order operators, which means we need data for both sides of {eq}`eq:opinf-example-rom`.
The state and input matrices $\mathbf{Q}$ and $\mathbf{U}$ provide data for the right-hand side $\mathbf{f}(t,\mathbf{q}(t),\mathbf{u}(t))$, but we also need data for the time derivative $\frac{\text{d}}{\text{d}t}\mathbf{q}(t)$ corresponding to the state snapshot data:

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
If these time derivatives cannot be computed directly by evaluating $\mathbf{f}(t_{j}, \mathbf{q}_{j}, \mathbf{u}_{j})$, they must be estimated from the state snapshots.
The most common strategy is to use [finite differences](https://en.wikipedia.org/wiki/Numerical_differentiation) of the state snapshots, implemented in this package as `opinf.pre.ddt()`.
See [**the Tutorial**](sec-tutorial) for example usage.
:::

:::{warning}
Take the time derivatives of _processed data_ if you do any preprocessing,
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
## Compute a Low-dimensional Basis

The purpose of learning a reduced-order model is to achieve a computational speedup, which is a result of the dimension reduction from $n$ to $r$.
This is accomplished by introducing the low-dimensional approximation

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

We call $\mathbf{V}_{r} \in \mathbb{R}^{n \times r}$ the _basis matrix_ and typically require that it has orthonormal columns.
The basis matrix is the link between the high-dimensional state space of the full-order model {eq}`eq:opinf-example-fom` and the low-dimensional state space of the reduced-order model {eq}`eq:opinf-example-rom`.

:::{image} ../../images/basis-projection.svg
:align: center
:width: 80 %
:::

See the [Basis Computation](sec-basis-computation) page for tools on computing the basis $\mathbf{V}_{r}\in\mathbb{R}^{n \times r}$ and selecting an appropriate dimension $r$.

<!-- :::{tip}
In the case of finite differences, the time derivative estimation can be done after the data is projected to the low-dimensional subspace defined by the basis (the column space of $\mathbf{V}_{r}$).
Instead of feeding the data matrix $\mathbf{Q}$ to `opinf.pre.ddt()`, consider computing $\widehat{\mathbf{Q}} := \mathbf{V}_{r}^{\mathsf{T}}\mathbf{Q}$ first and using that as the input to `opinf.pre.ddt()`.
You can also use $\widehat{\mathbf{Q}}$ as the input when you fit the reduced-order model object.
::: -->

---

(subsec-opinf-regression)=
## Set up and Solve a Low-dimensional Regression

TODO

---

(subsec-rom-evaluation)=
## Evaluate the Reduced-order Model

TODO

---

## Brief Example

Suppose you have the state snapshot matrix $\mathbf{Q}\in\mathbb{R}^{n\times k}$ stored as the variable `Q` and the input matrix $\mathbf{U}\in\mathbb{R}^{n\times k}$ as the variable `U` and that the time domain corresponding to the data stored as the variable `t`.
That is, `Q[:,j]` and `U[:,j]` are the state and input, respectively, corresponding to time `t[j]`.
Then the following code learns a reduced-order model of the form

$$
    \frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t)
    = \widehat{\mathbf{A}}\widehat{\mathbf{q}}(t)
    + \widehat{\mathbf{H}}(\widehat{\mathbf{q}}(t)\otimes\widehat{\mathbf{q}}(t))
    + \widehat{\mathbf{B}}\widehat{\mathbf{u}}(t)
$$

from the training data.

```python
import rom_operator_inference as opinf

# Compute a rank-10 basis (POD) from the state data.
>>> Vr, svdavls = opinf.pre.pod_basis(Q, 10)

# Estimate time derivatives of the state with finite differences.
>>> Qdot = opinf.pre.ddt(Q, t)

# Define a reduced-order model with the structure indicated above.
>>> rom = opinf.ContinuousOpInfROM(modelform="AHB")

# Fit the model (projection and regression).
>>> rom.fit(basis=Vr, states=Q, ddts=Qdot, inputs=U)

# Simulate the learned model over the time domain.
>>> Q_ROM = rom.predict(Q[:,0], t)

# Compute the error of the ROM prediction.
>>> error = opinf.post.Lp_error(Q, Q_rom)
```

See [**the Tutorial**](sec-tutorial) for a more thorough example.
