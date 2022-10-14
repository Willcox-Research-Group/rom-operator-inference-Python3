(sec-lifting)=
# Change of Variables / Lifting

Operator Inference learns models with polynomial terms, for example,

$$
    \frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t)
    = \widehat{\mathbf{c}}
    + \widehat{\mathbf{A}}\widehat{\mathbf{q}}(t)
    + \widehat{\mathbf{H}}[\widehat{\mathbf{q}}(t)\otimes\widehat{\mathbf{q}}(t)]
    + \widehat{\mathbf{B}}\mathbf{u}(t).
$$

The structure of this reduced-order model is [inspired by the structure of the full-order model](projection-preserves-structure).
In some systems with nonpolynomial nonlinearities, a change of variables can induce a polynomial structure.
For example, the incompressible Euler equations for an ideal gas can be written in conservative form as

$$
\begin{align*}
    \frac{\partial}{\partial t}\left[\rho\right]
    &= -\frac{\partial}{\partial x}\left[\rho u\right],
    &
    \frac{\partial}{\partial t}\left[\rho u\right]
    &= -\frac{\partial}{\partial x}\left[\rho u^2 + p\right],
    &
    \frac{\partial}{\partial t}\left[\rho e\right]
    &= -\frac{\partial}{\partial x}\left[(\rho e + p)u\right].
\end{align*}
$$

These equations are nonpolynomially nonlinear in the conservative variables $\vec{q}_{c} = (\rho, \rho u, \rho e)$.
However, by changing to the specific-volume variables $\vec{q} = (u, p, \zeta)$, we have

$$
\begin{align*}
    \frac{\partial u}{\partial t}
    &= -u \frac{\partial u}{\partial x} - \zeta\frac{\partial p}{\partial x},
    &
    \frac{\partial p}{\partial t}
    &= -\gamma p \frac{\partial u}{\partial x} - u\frac{\partial p}{\partial x},
    &
    \frac{\partial \zeta}{\partial t}
    &= -u \frac{\partial\zeta}{\partial x} + \zeta\frac{\partial u}{\partial x},
\end{align*}
$$

which only has quadratic terms in $\vec{q}$.
Hence, a quadratic reduced-order model of the form

$$
    \frac{\text{d}}{\text{d}t}\widehat{\mathbf{q}}(t)
    = \widehat{\mathbf{H}}[\widehat{\mathbf{q}}(t)\otimes\widehat{\mathbf{q}}(t)]
$$

can be learned for this system using data in the variables $\vec{q}$.
See {cite}`qian2020liftandlearn` for more details.

**This package does not contain tools for identifying or transforming to lifting variables**, but once a lifting transformation has been identified, it is usually a straightforward and inexpensive computation.
For example, this is the variable transformation for the Euler equations written above:

```python
def lift(state, gamma=1.4):
    """LIFT from the conservative variables to the learning variables,
    [rho, rho*u, rho*e] -> [u, p, 1/rho].
    """
    rho, rho_u, rho_e = np.split(state, 3)

    u = rho_u / rho
    p = (gamma - 1)*(rho_e - 0.5*rho*u**2)
    zeta = 1 / rho

    return np.concatenate((u, p, zeta))

def unlift(upzeta, gamma=1.4):
    """UNLIFT from the learning variables to the conservative variables,
    [u, p, 1/rho] -> [rho, rho*u, rho*e].
    """
    u, p, zeta = np.split(upzeta, 3)

    rho = 1/zeta
    rho_u = rho*u
    rho_e = p/(gamma - 1) + 0.5*rho*u**2

    return np.concatenate((rho, rho_u, rho_e))
```

:::{admonition} Takeaway
:class: attention
**You are responsible** for ensuring that the structure of the reduced-order model to be learned is appropriate for the problem.
Changing variables can sometimes help to induce a polynomial structure; when needed, do this before any other preprocessing step.
:::

See {cite}`qian2020liftandlearn,swischuk2020combustion,mcquarrie2021combustion,jain2021performance,khodabakhshi2022diffalg` for examples of Operator Inference with lifting, and {cite}`benner2020opinfdeim` for an alternative method to approaching non-linearities via the discrete empirical interpolation method (DEIM).
