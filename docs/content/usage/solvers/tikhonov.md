# Tikhonov Regularization

For $\mathcal{R}\equiv 0$ and a few other common choices of $\mathcal{R}$, the OpInf learning problem is _linear_ and can be solved explicitly.

:::{dropdown} $\mathcal{R} \equiv 0$
If there is no regularization, then the solution to the linear least-squares problem is given by the _normal equations_:

$$
\widehat{\mathbf{O}}^{\mathsf{T}}
= (\mathbf{D}^{\mathsf{T}}\mathbf{D})^{-1}\mathbf{D}^{\mathsf{T}}\mathbf{Y}^{\mathsf{T}}.
$$
:::

:::{dropdown} $\mathcal{R}(\widehat{\mathbf{O}}) = ||\lambda\widehat{\mathbf{O}}||_{F}^{2}$
:::
This choice of regularization is called the $L_{2}$ regularizer, a specific type of Tikhonov regularizer.
The solution is given by the modified normal equations

$$
\widehat{\mathbf{O}}^{\mathsf{T}}
= (\mathbf{D}^{\mathsf{T}}\mathbf{D} + \lambda\mathbf{I})^{-1}\mathbf{D}^{\mathsf{T}}\mathbf{Y}^{\mathsf{T}}.
$$