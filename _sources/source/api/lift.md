# `opinf.lift`

```{eval-rst}
.. automodule:: opinf.lift
```

Operator Inference learns models with polynomial terms, for example,

$$
    \frac{\text{d}}{\text{d}t}\qhat(t)
    = \chat
    + \Ahat\qhat(t)
    + \Hhat[\qhat(t)\otimes\qhat(t)]
    + \Bhat\u(t).
$$

If training data do not exhibit this kind of polynomial structure, a reduced-order model learned through Operator Inference is not likely to perform well.
In some systems with nonpolynomial nonlinearities, a change of variables can induce a polynomial structure, which can greatly improve the effectiveness of Operator Inference.
Such variable transformations are often called _lifting maps_, especially if the transformation augments the state by introducing additional variables.
This module defines a template class, {class}`LifterTemplate`, for implementing lifting maps in a way that interfaces with the rest of the package.

```{eval-rst}
.. currentmodule:: opinf.lift

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    LifterTemplate
```

## Inheritance Template

To define a custom lifting map, define a class that inherits from {class}`LifterTemplate` and implements a `lift()` and `unlift()` method.
An optional ``lift_ddts()`` method may be implemented to compute the time derivatives of the lifted state variables.
Once implemented, the `verify()` method may be used to test the consistency of these three methods.

```python
import opinf

class MyLifter(opinf.lift.LifterTemplate):
    """Custom lifting map."""

    @staticmethod
    def lift(state):
        """Lift the native state variables to the learning variables."""
        raise NotImplementedError

    @staticmethod
    def unlift(lifted_state):
        """Recover the native state variables from the learning variables."""
        raise NotImplementedError

    @staticmethod
    def lift_ddts(state, ddts):
        """Lift the native state time derivatives to the time derivatives
        of the learning variables.
        """
        raise NotImplementedError
```

## Example: Polynomial Lifting Maps

This example originates from {cite}`qian2021thesis`.
Consider a nonlinear diffusion-reaction equation with a cubic reaction term:

$$
\begin{align*}
    \frac{\partial}{\partial t}q
    = \frac{\partial^{2}}{\partial x^{2}}q - q^3.
\end{align*}
$$

By introducing an auxiliary variable $w = q^{2}$, we have $\frac{\partial}{\partial t}w = 2q\frac{\partial}{\partial t}q$,
hence the previous equation can be expressed as the system

$$
\begin{align*}
    \frac{\partial}{\partial t}q
    &= \frac{\partial^{2}}{\partial x^{2}}q - qw.
    &
    \frac{\partial}{\partial t}2
    &= 2q\frac{\partial^{2}}{\partial x^{2}}q - 2w^2.
\end{align*}
$$

This system is quadratic in the lifted variables $(q, w)$.
The following class implements the lifting map $q \mapsto (q, q^2)$.

```python
class QuadraticLifter(LifterTemplate):
    r"""Quadratic lifting map q -> (q, q^2)."""

    @staticmethod
    def lift(states):
        """Apply the lifting map q -> (q, q^2)."""
        return np.concatenate((states, states**2))


    @staticmethod
    def unlift(lifted_states):
        """Apply the reverse lifting map (q, q^2) -> q."""
        return np.split(lifted_states, 2, axis=0)[0]

    @staticmethod
    def lift_ddts(states, ddts):
        """Get the time derivatives of the lifted variables,
        d / dt (q, q^2) = (q_t, 2qq_t).
        """
        return np.concatenate((ddts, 2 * states * ddts))
```

A more detailed version of this class is included in the package as an example.

```{eval-rst}
.. currentmodule:: opinf.lift

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    QuadraticLifter
    PolynomialLifter
```

## Example: Specific Volume Variables

The compressible Euler equations for an ideal gas can be written in conservative form as

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
However, by changing to the specific-volume variables $\vec{q} = (u, p, \zeta)$ and using the ideal gas law

$$
\begin{align*}
    \rho e = \frac{p}{\gamma - 1} + \frac{\rho u^2}{2},
\end{align*}
$$

we arrive at a _quadratic_ system

$$
\begin{align*}
    \frac{\partial u}{\partial t}
    &= -u \frac{\partial u}{\partial x} - \zeta\frac{\partial p}{\partial x},
    &
    \frac{\partial p}{\partial t}
    &= -\gamma p \frac{\partial u}{\partial x} - u\frac{\partial p}{\partial x},
    &
    \frac{\partial \zeta}{\partial t}
    &= -u \frac{\partial\zeta}{\partial x} + \zeta\frac{\partial u}{\partial x}.
\end{align*}
$$

Hence, a quadratic reduced-order model of the form

$$
    \frac{\text{d}}{\text{d}t}\qhat(t)
    = \Hhat[\qhat(t)\otimes\qhat(t)]
$$

can be learned for this system using data in the variables $\vec{q}$.
See {cite}`qian2020liftandlearn` for details.

The following class defines this the variable transformation using the {class}`LifterTemplate`.

```python
import opinf

class EulerLifter(opinf.lift.LifterTemplate):
    """Lifting map for the Euler equations transforming conservative
    variables to specific volume variables.
    """
    def __init__(self, gamma=1.4):
        """Store the heat capacity ratio, gamma."""
        self.gamma = gamma

    @staticmethod
    def lift(state, gamma=1.4):
        """LIFT from the conservative variables to the learning variables,
        [rho, rho*u, rho*e] -> [u, p, 1/rho].
        """
        rho, rho_u, rho_e = np.split(state, 3)

        u = rho_u / rho
        p = (gamma - 1)*(rho_e - 0.5*rho*u**2)
        zeta = 1 / rho

        return np.concatenate((u, p, zeta))

    @staticmethod
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
Changing variables can sometimes help to induce a polynomial structure; when needed, do this before any other preprocessing steps.
:::

See {cite}`qian2020liftandlearn,swischuk2020combustion,mcquarrie2021combustion,jain2021performance,khodabakhshi2022diffalg` for examples of Operator Inference with lifting, and {cite}`benner2020opinfdeim` for an alternative method to approaching nonlinearities via the discrete empirical interpolation method (DEIM).
