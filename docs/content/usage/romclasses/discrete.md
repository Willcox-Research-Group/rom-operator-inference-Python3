(sec-discrete)=
# Discrete-time ROMs

:::{warning}
This page is under construction.
:::

The Operator Inference framework can be used to construct reduced-order models for approximating _discrete_ dynamical systems, as may arise from discretizing PDEs in both space and time.
For instance, we can learn a discrete ROM of the form

$$
\begin{align*}
    \widehat{\mathbf{q}}_{j+1}
    = \widehat{\mathbf{c}}
    + \widehat{\mathbf{A}}\widehat{\mathbf{q}}_{j}
    + \widehat{\mathbf{H}}(\widehat{\mathbf{q}}_{j}\otimes\widehat{\mathbf{q}}_{j})
    + \widehat{\mathbf{B}}\mathbf{u}_{j}.
\end{align*}
$$

The procedure is the same as the continuous-time setting, with the exception that the snapshot matrix $\mathbf{Q}\in\mathbb{R}^{n \times k}$ has columns $\mathbf{q}_{j}$ for $j = 0,1,...,k-1$, while the right-hand side matrix $\mathbf{R}$ has columns $\mathbf{q}_{j}$ for $j = 1,2,...,k$.
Thus, the (not yet regularized) least-squares problem to be solved has the form

$$
\begin{align*}
    \min_{\widehat{\mathbf{c}},\widehat{\mathbf{A}},\widehat{\mathbf{H}},\widehat{\mathbf{B}}}\sum_{j=0}^{k-1}\left\|
      \widehat{\mathbf{c}}
    + \widehat{\mathbf{A}}\widehat{\mathbf{q}}_{j}
    + \widehat{\mathbf{H}}(\widehat{\mathbf{q}}_{j}\otimes\widehat{\mathbf{q}}_{j})
    + \widehat{\mathbf{B}}\mathbf{u}_j
    - \widehat{\mathbf{q}}_{j+1}
    \right\|_2^2.
\end{align*}
$$

## DiscreteOpInfROM

This class constructs a reduced-order model for the discrete, nonparametric systems via Operator Inference.

**`DiscreteOpInfROM.fit(basis, states, inputs=None, regularizer=0)`**: Compute the operators of the reduced-order model via Operator Inference.
- **Parameters**
    - `Vr`: $n \times r$ basis for the linear reduced space on which the full-order model will be projected. Each column is a basis vector. The column space of `Vr` should be a good approximation of the column space of the full-order snapshot matrix `Q`. If given as `None`, `Q` is assumed to be the projected snapshot matrix $\mathbf{V}_{r}^{\top}\mathbf{Q}$.
    - `Q`: $n \times k$ snapshot matrix of solutions to the full-order model, or the $r \times k$ projected snapshot matrix $\mathbf{V}_{r}^{\top}\mathbf{Q}$. Each column is one snapshot.
    - `U`: $m \times (k-1)$ input matrix (or a $(k-1)$-vector if $m = 1$). Each column is the input for the corresponding column of `Q`. Only required when `'B'` is in `modelform`.
    - `P`: Tikhonov regularization matrix for the least-squares problem; see [`lstsq`](#least-squares-solvers).
- **Returns**
    - Trained `DiscreteOpInfROM` object.

**`DiscreteOpInfROM.predict(state0, niters, inputs=None)`**: Step forward the learned ROM `niters` steps.
- **Parameters**
    - `state0`: Initial state vector, either full order ($n$-vector) or projected to reduced order ($r$-vector). If `Vr=None` in `fit()`, this must be the projected initial state $\mathbf{V}_{r}^{\top}\mathbf{q}$<sub>0</sub>.
    - `niters`: Number of times to step the system forward.
    - `inputs`: Inputs for the next `niters`-1 time steps, as an $m \times$ `niters`$-1$ matrix (or an (`niters`$-1$)-vector if $m = 1$). This argument is only required if `'B'` is in `modelform`.
- **Returns**
    - `Q_ROM`: $n \times$ `niters` matrix of approximate solutions to the full-order system, including the initial condition; or, if `basis=None` in `fit()`, the $r \times $ `niters` solution in the reduced-order space. Each column is one iteration of the solution.
