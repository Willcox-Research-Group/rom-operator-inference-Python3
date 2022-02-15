(sec-opinf-overview)=
# What is Operator Inference?

Operator Inference is a projection-based model reduction technique that learns reduced-order models from data.
The main steps are the following.

1. [**Prepare Training Data**](subsec-training-data). You provide high-dimensional data to learn from and do any [preprocessing](subsec-preprocessing) to prepare for model learning.
2. [**Compute a Low-dimensional Basis**](subsec-basis-computation). Take the SVD of the training data and extract the first few left singular vectors.
3. [**Set up and Solve Low-dimensional Regression**](subsec-opinf-regression). This is the core objective of this package.
4. [**Evaluate the Reduced-order Model**](subsec-rom-evaluation). Use the learned model to do prediction.

This page reviews each of these steps and shows how to do them with this package.

```{image} ../../images/opinf-summary.svg
:align: center
:width: 80 %
```

---

## Problem Statement

Consider a system of ODEs with state $\mathbf{q}(t)\in\mathbb{R}^{n}$ and inputs $\mathbf{u}(t)\in\mathbb{R}^{m}$ whose nonlinearities are polynomial, for example,

$$
    \frac{\text{d}}{\text{d}t}\mathbf{q}(t)
    = \mathbf{A}\mathbf{q}(t)
    + \mathbf{H}(\mathbf{q}(t)\otimes\mathbf{q}(t))
    + \mathbf{B}\mathbf{u}(t).
$$ (eq:opinf-example-fom)

We call $\mathbf{A}$, $\mathbf{H}$, and $\mathbf{B}$ the _full-order operators_ and {eq}`eq:opinf-example-fom` the _full-order model_.
Given samples of the state $\mathbf{q}(t)$, Operator Inference learns a reduced-order model with the same structure as {eq}`eq:opinf-example-fom`, but with much smaller state $\widehat{\mathbf{q}}(t) \in \mathbb{R}^{r}, r \ll n$:

$$
    \frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t)
    = \widehat{\mathbf{A}}\widehat{\mathbf{q}}(t)
    + \widehat{\mathbf{H}}(\widehat{\mathbf{q}}(t)\otimes\widehat{\mathbf{q}}(t))
    + \widehat{\mathbf{B}}\widehat{\mathbf{u}}(t).
$$

Our goal is to infer the _reduced-order operators_ $\widehat{\mathbf{A}}$, $\widehat{\mathbf{H}}$, and $\widehat{\mathbf{B}}$ **without** direct access to the full-order operators $\mathbf{A}$, $\mathbf{H}$, and $\mathbf{B}$.

---

(subsec-training-data)=
## You Provide Training Data

Operator Inference learns reduced models from data, which must be provided by the user.
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
    = \mathbf{f}(\mathbf{q}_{j}, \mathbf{u}_{j}, t_{j}).
$$

(subsec-preprocessing)=
### Preprocessing

Raw dynamical systems data often needs to be lightly preprocessed in order to promote stability in the model learning problem.
Common preprocessing steps include
1. Variable transformations / lifting to induce a polynomial structure.
2. Centering or shifting data to account for boundary conditions.
3. Scaling / nondimensionalizing the variables represented in the state.

See TODO
and {cite}`QKPW2020LiftAndLearn,MHW2021regOpInfCombustion,SKHW2020ROMCombustion` for examples of step 1.

### Estimating Time Derivatives

TODO

```{tip}
In the case of finite differences, the time derivative estimation can be done after the data is projected to the low-dimensional subspace defined by the basis (the column space of $\mathbf{V}_{r}$).
Instead of feeding the data matrix $\mathbf{Q}$ to `opinf.pre.ddt()`, consider computing $\widehat{\mathbf{Q}} := \mathbf{V}_{r}^{\mathsf{T}}\mathbf{Q}$ first and using that as the input to `opinf.pre.ddt()`.
You can also use $\widehat{\mathbf{Q}}$ as the input when you fit the reduced-order model object.
```

---

(subsec-basis-computation)=
## Basis Computation

The purpose of learning a reduced-order model is to achieve a computational speedup, which is a result of the dimension reduction from $n$ to $r$.
This is accomplished by introducing the low-dimensional representation

$$
    \mathbf{q}(t)
    \approx \mathbf{V}_{r} \widehat{\mathbf{q}}(t)
    = \sum_{i=1}^{r}\mathbf{v}_{i}\hat{q}_{i}(t),
$$

where

\begin{align*}
    \mathbf{V}_{r}
    &= \left[\begin{array}{ccc}
        & & \\
        \mathbf{v}_{1} & \cdots & \mathbf{v}_{r}
        \\ & &
    \end{array}\right],
    &
    \widehat{\mathbf{q}}
    &= \left[\begin{array}{c}
        \hat{q}_{1}(t) \\ \vdots \\ \hat{q}_{r}(t)
    \end{array}\right],
\end{align*}

such that $\mathbf{V}_{r} \in \mathbb{R}^{n \times r}$ has orthonormal columns.

### Proper Orthogonal Decomposition

Any orthonormal basis may be used for $\mathbf{V}_{r}$, but we advocate using the [proper orthogonal decomposition](https://en.wikipedia.org/wiki/Proper_orthogonal_decomposition) (POD), also referred to as the SVD or PCA.

### Choosing the Basis Size

The dimension $r$ is the number of basis vectors used in the low-dimensional representation

---

(subsec-opinf-regression)=
## Operator Learning via Regression

TODO

### Least-squares Problem Setup

TODO

### Least-squares Solvers

TODO

---

(subsec-rom-evaluation)=
## Reconstruction

TODO

### Extracting Reduced-order Operators

TODO

---

## Brief Example

Let's say you have the state matrix $\mathbf{Q}\in\mathbb{R}^{n\times k}$ stored as the variable `Q` and the input matrix $\mathbf{U}\in\mathbb{R}^{n\times k}$ as the variable `U` and that the time domain corresponding to the data stored as the variable `t`.
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
>>> Vr = opinf.pre.pod_basis(Q, 10)

# Estimate time derivatives of the state with finite differences.
>>> Qdot = opinf.pre.ddt(Q, t)

# Define a reduced-order model with the structure indicated above.
>>> rom = opinf.InferredContinuousROM(modelform="AHB")

# Fit the model (projection and regression).
>>> rom.fit(basis=Vr, states=Q, ddts=Qdot, inputs=U)

# Simulate the learned model over the time domain.
>>> Q_ROM = rom.predict(Q[:,0], t)

# Compute the error of the ROM prediction.
>>> error = opinf.post.Lp_error(Q, Q_rom)
```

See [**the Tutorial**](sec-tutorial) for a complete example.
