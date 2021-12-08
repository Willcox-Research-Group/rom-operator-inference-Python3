(sec-discrete)=
# Discrete-time ROMs

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
