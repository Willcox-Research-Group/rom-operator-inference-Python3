(sec-continuous)=
# Continuous-time ROMs

:::{warning}
This page is under construction.
:::

## ContinuousOpInfROM

This class constructs a reduced-order model for continuous, nonparametric systems via Operator Inference.
That is, given snapshot data, a basis, and a form for a reduced model, it computes the reduced model operators by solving an ordinary least-squares problem.

**`ContinuousOpInfROM.fit(Vr, Q, Qdot, U=None, P=0)`**: Compute the operators of the reduced-order model via Operator Inference.
- **Parameters**
    - `basis`: $n \times r$ basis for the linear reduced space on which the full-order model will be projected (for example, a POD basis matrix; see [`pre.pod_basis()`](#preprocessing-tools)). Each column is a basis vector. The column space of `Vr` should be a good approximation of the column space of the full-order snapshot matrix `Q`. If given as `None`, `Q` is assumed to be the projected snapshot matrix $\mathbf{V}_{r}^{\top}\mathbf{Q}$ and `Qdot` is assumed to be the projected time derivative matrix.
    - `states`: An $n \times k$ snapshot matrix of solutions to the full-order model, or the $r \times k$ projected snapshot matrix $\mathbf{V}_{r}^{\top}\mathbf{Q}$. Each column is one snapshot.
    - `ddts`: $n \times k$ snapshot time derivative matrix for the full-order model, or the $r \times k$ projected snapshot time derivative matrix. Each column is the time derivative d**x**/dt for the corresponding column of `Q`. See the [`pre`](#preprocessing-tools) submodule for some simple derivative approximation tools.
    - `inputs`: $m \times k$ input matrix (or a _k_-vector if _m_ = 1). Each column is the input vector for the corresponding column of `Q`. Only required when `'B'` is in `modelform`.
    - `regularizer`: Tikhonov regularization matrix for the least-squares problem; see [`lstsq`](#least-squares-solvers).
- **Returns**
    - Trained `ContinuousOpInfROM` object.

**`ContinuousOpInfROM.predict(x0, t, u=None, **options)`**: Simulate the learned reduced-order model with [`scipy.integrate.solve_ivp()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
- **Parameters**
    - `x0`: Initial state vector, either full order ($n$-vector) or projected to reduced order ($r$-vector). If `Vr=None` in `fit()`, this must be the projected initial state $\mathbf{V}_{r}^{\top}\mathbf{q}_{0}$.
    - `t`: Time domain, an $n_{t}$-vector, over which to integrate the reduced-order model.
    - `u`: Input as a function of time, that is, a function mapping a `float` to an $m$-vector (or to a scalar if $m = 1$). Alternatively, the $m \times n_t$ matrix (or $n_t$-vector if $m = 1$) where column $j$ is the input vector corresponding to time `t[j]`. In this case, $\mathbf{u}(t)$ is approximated by a cubic spline interpolating the given inputs. This argument is only required if `'B'` is in `modelform`.
    - Other keyword arguments for [`scipy.integrate.solve_ivp()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
- **Returns**
    - `Q_ROM`: $n \times n_{t}$ matrix of approximate solution to the full-order system over `t`, or, if `Vr=None` in `fit()`, the $r \times n_{t}$ solution in the reduced-order space. Each column is one snapshot of the solution.
