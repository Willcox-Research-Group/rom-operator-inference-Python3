# ddt/_finite_difference.py
"""Finite-difference schemes for estimating snapshot time derivatives."""

__all__ = [
    "fwd1",
    "fwd2",
    "fwd3",
    "fwd4",
    "fwd5",
    "fwd6",
    "bwd1",
    "bwd2",
    "bwd3",
    "bwd4",
    "bwd5",
    "bwd6",
    "ctr2",
    "ctr4",
    "ctr6",
    "ord2",
    "ord4",
    "ord6",
    "ddt_uniform",
    "ddt_nonuniform",
    "ddt",
    "UniformFiniteDifferencer",
    "NonuniformFiniteDifferencer",
]

import types
import warnings
import numpy as np

from .. import errors
from ._base import DerivativeEstimatorTemplate


def _finite_difference(states, coeffs, mode, inputs=None):
    r"""Compute the first time derivative with a finite difference scheme.

    Parameters
    ----------
    states : (r, k) ndarray
        State snapshots: ``states[:, j]`` is the state at time :math:`t_j`.
        It is assumed that the states are uniformly spaced in time, i.e.,
        :math:`t_{j+1} - t_j` is the same for all :math:`j`.
    coeffs : (order + 1,) ndarray
        Finite difference coefficients. The convergence order of the
        estimation is ``len(coeffs) - 1``; for example, four points are
        used to obtain a third-order time derivative estimate.
    mode : str
        Indicator for if this is a forward, backward, or central difference.
        **Options:**

        * ``'fwd'``: forward difference.
        * ``'bwd'``: backward difference.
        * ``'ctr'``: central difference.
    inputs : (m, k) or (k,) ndarray or None
        Inputs corresponding to the states, if applicable.

    Returns
    -------
    _states : (r, k - order) ndarray
        State snapshots, excluding the last ``len(coeffs) - 1`` snapshots.
    ddts : (r, k - order) ndarray
        Time derivative estimates corresponding to the state snapshots.
    _inputs : (m, k - order) ndarray or None
        Inputs corresponding to ``_states``, if applicable.
        **Only returned** if ``inputs`` is provided.
    """
    r, k = states.shape
    order = len(coeffs) - 1

    if mode == "fwd":
        cols = slice(0, -order)
    elif mode == "bwd":
        cols = slice(order, None)
    elif mode == "ctr":
        margin = order // 2
        cols = slice(margin, -margin)
    else:
        raise ValueError(f"invalid finite difference mode '{mode}'")

    ddts = np.zeros((r, k - order))
    for j, coeff in enumerate(coeffs):
        if coeff != 0:
            ddts += coeff * states[:, j : k - order + j]

    if inputs is not None:
        return states[:, cols], ddts, inputs[..., cols]
    return states[:, cols], ddts


# Forward differences =========================================================
def fwd1(states: np.ndarray, dt: float, inputs=None):
    r"""First-order forward difference for estimating the first derivative.

    .. math::
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_j}
       \approx \frac{1}{\delta t}(\q(t_{j+1}) - \q(t_j))

    where :math:`\delta t = t_{j+1} - t_j` for all :math:`j`.

    Parameters
    ----------
    states : (r, k) ndarray
        State snapshots: ``states[:, j]`` is the state at time :math:`t_j`.
    dt : float
        Time step between snapshots.
    inputs : (m, k) or (k,) ndarray or None
        Inputs corresponding to the states, if applicable.

    Returns
    -------
    _states : (r, k - 1) ndarray
        State snapshots, excluding the last snapshot.
    ddts : (r, k - 1) ndarray
        Time derivative estimates corresponding to the state snapshots.
    _inputs : (m, k - 1) or (k - 1,) ndarray or None
        Inputs corresponding to ``_states``, if applicable.
        **Only returned** if ``inputs`` is not ``None``.
    """
    coeffs = np.array([-1, 1]) / dt
    return _finite_difference(states, coeffs, "fwd", inputs)


def fwd2(states: np.ndarray, dt: float, inputs=None):
    r"""Second-order forward difference for estimating the first derivative.

    .. math::
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_j}
       \approx \frac{1}{2\delta t}(-3\q(t_j) + 4\q(t_{j+1}) - \q(t_{j+2}))

    where :math:`\delta t = t_{j+1} - t_j` for all :math:`j`.

    Parameters
    ----------
    states : (r, k) ndarray
        State snapshots: ``states[:, j]`` is the state at time :math:`t_j`.
    dt : float
        Time step between snapshots.
    inputs : (m, k) or (k,) ndarray or None
        Inputs corresponding to the states, if applicable.

    Returns
    -------
    _states : (r, k - 2) ndarray
        State snapshots, excluding the last two snapshots.
    ddts : (r, k - 2) ndarray
        Time derivative estimates corresponding to the state snapshots.
    _inputs : (m, k - 2) or (k - 2,) ndarray or None
        Inputs corresponding to ``_states``, if applicable.
        **Only returned** if ``inputs`` is not ``None``.
    """
    coeffs = np.array([-3, 4, -1]) / (2 * dt)
    return _finite_difference(states, coeffs, "fwd", inputs)


def fwd3(states: np.ndarray, dt: float, inputs=None):
    r"""Third-order forward difference for estimating the first derivative.

    .. math::
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_j}
       \approx \frac{1}{6\delta t}
       (-11\q(t_j) + 18\q(t_{j+1}) - 9\q(t_{j+2}) + 2\q(t_{j+3}))

    where :math:`\delta t = t_{j+1} - t_j` for all :math:`j`.

    Parameters
    ----------
    states : (r, k) ndarray
        State snapshots: ``states[:, j]`` is the state at time :math:`t_j`.
    dt : float
        Time step between snapshots.
    inputs : (m, k) or (k,) ndarray or None
        Inputs corresponding to the states, if applicable.

    Returns
    -------
    _states : (r, k - 3) ndarray
        State snapshots, excluding the last three snapshots.
    ddts : (r, k - 3) ndarray
        Time derivative estimates corresponding to the state snapshots.
    _inputs : (m, k - 3) or (k - 3,) ndarray or None
        Inputs corresponding to ``_states``, if applicable.
        **Only returned** if ``inputs`` is not ``None``.
    """
    coeffs = np.array([-11, 18, -9, 2]) / (6 * dt)
    return _finite_difference(states, coeffs, "fwd", inputs)


def fwd4(states: np.ndarray, dt: float, inputs=None):
    r"""Fourth-order forward difference for estimating the first derivative.

    .. math::
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_j}
       \approx \frac{1}{12\delta t}(-25\q(t_j) + 48\q(t_{j+1}) - 36\q(t_{j+2})
       + 16\q(t_{j+3}) - 3\q(t_{j+4}))

    where :math:`\delta t = t_{j+1} - t_j` for all :math:`j`.

    Parameters
    ----------
    states : (r, k) ndarray
        State snapshots: ``states[:, j]`` is the state at time :math:`t_j`.
    dt : float
        Time step between snapshots.
    inputs : (m, k) or (k,) ndarray or None
        Inputs corresponding to the states, if applicable.

    Returns
    -------
    _states : (r, k - 4) ndarray
        State snapshots, excluding the last four snapshots.
    ddts : (r, k - 4) ndarray
        Time derivative estimates corresponding to the state snapshots.
    _inputs : (m, k - 4) or (k - 4,) ndarray or None
        Inputs corresponding to ``_states``, if applicable.
        **Only returned** if ``inputs`` is not ``None``.
    """
    coeffs = np.array([-25, 48, -36, 16, -3]) / (12 * dt)
    return _finite_difference(states, coeffs, "fwd", inputs)


def fwd5(states: np.ndarray, dt: float, inputs=None):
    r"""Fifth-order forward difference for estimating the first derivative.

    .. math::
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_j}
       \approx \frac{1}{60\delta t}(-137\q(t_j) + 300\q(t_{j+1})
       - 300\q(t_{j+2}) + 200\q(t_{j+3}) - 75\q(t_{j+4}) + 12\q(t_{j+5}))

    where :math:`\delta t = t_{j+1} - t_j` for all :math:`j`.

    Parameters
    ----------
    states : (r, k) ndarray
        State snapshots: ``states[:, j]`` is the state at time :math:`t_j`.
    dt : float
        Time step between snapshots.
    inputs : (m, k) or (k,) ndarray or None
        Inputs corresponding to the states, if applicable.

    Returns
    -------
    _states : (r, k - 5) ndarray
        State snapshots, excluding the last five snapshots.
    ddts : (r, k - 5) ndarray
        Time derivative estimates corresponding to the state snapshots.
    _inputs : (m, k - 5) or (k - 5,) ndarray or None
        Inputs corresponding to ``_states``, if applicable.
        **Only returned** if ``inputs`` is not ``None``.
    """
    coeffs = np.array([-137, 300, -300, 200, -75, 12]) / (60 * dt)
    return _finite_difference(states, coeffs, "fwd", inputs)


def fwd6(states: np.ndarray, dt: float, inputs=None):
    r"""Sixth-order forward difference for estimating the first derivative.

    .. math::
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_j}
       \approx \frac{1}{60\delta t}(
       -147\q(t_j) + 360\q(t_{j+1}) - 450\q(t_{j+2}) + 400\q(t_{j+3})
       - 225\q(t_{j+4}) + 72\q(t_{j+5}) - 10\q(t_{j+6}))

    where :math:`\delta t = t_{j+1} - t_j` for all :math:`j`.

    Parameters
    ----------
    states : (r, k) ndarray
        State snapshots: ``states[:, j]`` is the state at time :math:`t_j`.
    dt : float
        Time step between snapshots.
    inputs : (m, k) or (k,) ndarray or None
        Inputs corresponding to the states, if applicable.

    Returns
    -------
    _states : (r, k - 6) ndarray
        State snapshots, excluding the last six snapshots.
    ddts : (r, k - 6) ndarray
        Time derivative estimates corresponding to the state snapshots.
    _inputs : (m, k - 6) or (k - 6,) ndarray or None
        Inputs corresponding to ``_states``, if applicable.
        **Only returned** if ``inputs`` is not ``None``.
    """
    coeffs = np.array([-147, 360, -450, 400, -225, 72, -10]) / (60 * dt)
    return _finite_difference(states, coeffs, "fwd", inputs)


# Backwards differences =======================================================
def bwd1(states: np.ndarray, dt: float, inputs=None):
    r"""First-order backward difference for estimating the first derivative.

    .. math::
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_j}
       \approx \frac{1}{\delta t}(\q(t_j) - \q(t_{j-1}))

    where :math:`\delta t = t_{j+1} - t_j` for all :math:`j`.

    Parameters
    ----------
    states : (r, k) ndarray
        State snapshots: ``states[:, j]`` is the state at time :math:`t_j`.
    dt : float
        Time step between snapshots.
    inputs : (m, k) or (k,) ndarray or None
        Inputs corresponding to the states, if applicable.

    Returns
    -------
    _states : (r, k - 1) ndarray
        State snapshots, excluding the first snapshot.
    ddts : (r, k - 1) ndarray
        Time derivative estimates corresponding to the state snapshots.
    _inputs : (m, k - 1) or (k - 1,) ndarray or None
        Inputs corresponding to ``_states``, if applicable.
        **Only returned** if ``inputs`` is not ``None``.
    """
    coeffs = np.array([-1, 1]) / dt
    return _finite_difference(states, coeffs, "bwd", inputs)


def bwd2(states: np.ndarray, dt: float, inputs=None):
    r"""Second-order backward difference for estimating the first derivative.

    .. math::
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_j}
       \approx \frac{1}{2\delta t}(3\q(t_j) - 4\q(t_{j-1}) + \q(t_{j-2}))

    where :math:`\delta t = t_{j+1} - t_j` for all :math:`j`.

    Parameters
    ----------
    states : (r, k) ndarray
        State snapshots: ``states[:, j]`` is the state at time :math:`t_j`.
    dt : float
        Time step between snapshots.
    inputs : (m, k) or (k,) ndarray or None
        Inputs corresponding to the states, if applicable.

    Returns
    -------
    _states : (r, k - 2) ndarray
        State snapshots, excluding the first two snapshots.
    ddts : (r, k - 2) ndarray
        Time derivative estimates corresponding to the state snapshots.
    _inputs : (m, k - 2) or (k - 2,) ndarray or None
        Inputs corresponding to ``_states``, if applicable.
        **Only returned** if ``inputs`` is not ``None``.
    """
    coeffs = np.array([1, -4, 3]) / (2 * dt)
    return _finite_difference(states, coeffs, "bwd", inputs)


def bwd3(states: np.ndarray, dt: float, inputs=None):
    r"""Third-order backward difference for estimating the first derivative.

    .. math::
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_j}
       \approx \frac{1}{6\delta t}
       (11\q(t_j) - 18\q(t_{j-1}) + 9\q(t_{j-2}) - 2\q(t_{j-3}))

    where :math:`\delta t = t_{j+1} - t_j` for all :math:`j`.

    Parameters
    ----------
    states : (r, k) ndarray
        State snapshots: ``states[:, j]`` is the state at time :math:`t_j`.
    dt : float
        Time step between snapshots.
    inputs : (m, k) or (k,) ndarray or None
        Inputs corresponding to the states, if applicable.

    Returns
    -------
    _states : (r, k - 3) ndarray
        State snapshots, excluding the first three snapshots.
    ddts : (r, k - 3) ndarray
        Time derivative estimates corresponding to the state snapshots.
    _inputs : (m, k - 3) or (k - 3,) ndarray or None
        Inputs corresponding to ``_states``, if applicable.
        **Only returned** if ``inputs`` is not ``None``.
    """
    coeffs = np.array([-2, 9, -18, 11]) / (6 * dt)
    return _finite_difference(states, coeffs, "bwd", inputs)


def bwd4(states: np.ndarray, dt: float, inputs=None):
    r"""Fourth-order backward difference for estimating the first derivative.

    .. math::
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_j}
       \approx \frac{1}{12\delta t}(25\q(t_j) - 48\q(t_{j-1}) + 36\q(t_{j-2})
       - 16\q(t_{j-3}) + 3\q(t_{j-4}))

    where :math:`\delta t = t_{j+1} - t_j` for all :math:`j`.

    Parameters
    ----------
    states : (r, k) ndarray
        State snapshots: ``states[:, j]`` is the state at time :math:`t_j`.
    dt : float
        Time step between snapshots.
    inputs : (m, k) or (k,) ndarray or None
        Inputs corresponding to the states, if applicable.

    Returns
    -------
    _states : (r, k - 4) ndarray
        State snapshots, excluding the first four snapshots.
    ddts : (r, k - 4) ndarray
        Time derivative estimates corresponding to the state snapshots.
    _inputs : (m, k - 4) or (k - 4,) ndarray or None
        Inputs corresponding to ``_states``, if applicable.
        **Only returned** if ``inputs`` is not ``None``.
    """
    coeffs = np.array([3, -16, 36, -48, 25]) / (12 * dt)
    return _finite_difference(states, coeffs, "bwd", inputs)


def bwd5(states: np.ndarray, dt: float, inputs=None):
    r"""Fifth-order backward difference for estimating the first derivative.

    .. math::
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_j}
       \approx \frac{1}{60\delta t}(137\q(t_j) - 300\q(t_{j-1})
       + 300\q(t_{j-2}) - 200\q(t_{j-3}) + 75\q(t_{j-4}) - 12\q(t_{j-5}))

    where :math:`\delta t = t_{j+1} - t_j` for all :math:`j`.

    Parameters
    ----------
    states : (r, k) ndarray
        State snapshots: ``states[:, j]`` is the state at time :math:`t_j`.
    dt : float
        Time step between snapshots.
    inputs : (m, k) or (k,) ndarray or None
        Inputs corresponding to the states, if applicable.

    Returns
    -------
    _states : (r, k - 5) ndarray
        State snapshots, excluding the first five snapshots.
    ddts : (r, k - 5) ndarray
        Time derivative estimates corresponding to the state snapshots.
    _inputs : (m, k - 5) or (k - 5,) ndarray or None
        Inputs corresponding to ``_states``, if applicable.
        **Only returned** if ``inputs`` is not ``None``.
    """
    coeffs = np.array([-12, 75, -200, 300, -300, 137]) / (60 * dt)
    return _finite_difference(states, coeffs, "bwd", inputs)


def bwd6(states: np.ndarray, dt: float, inputs=None):
    r"""Sixth-order backward difference for estimating the first derivative.

    .. math::
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_j}
       \approx \frac{1}{60\delta t}(
       147\q(t_j) - 360\q(t_{j-1}) + 450\q(t_{j-2}) - 400\q(t_{j-3})
       + 225\q(t_{j-4}) - 72\q(t_{j-5}) + 10\q(t_{j-6})
       )

    where :math:`\delta t = t_{j+1} - t_j` for all :math:`j`.

    Parameters
    ----------
    states : (r, k) ndarray
        State snapshots: ``states[:, j]`` is the state at time :math:`t_j`.
    dt : float
        Time step between snapshots.
    inputs : (m, k) or (k,) ndarray or None
        Inputs corresponding to the states, if applicable.

    Returns
    -------
    _states : (r, k - 6) ndarray
        State snapshots, excluding the first six snapshots.
    ddts : (r, k - 6) ndarray
        Time derivative estimates corresponding to the state snapshots.
    _inputs : (m, k - 6) or (k - 6,) ndarray or None
        Inputs corresponding to ``_states``, if applicable.
        **Only returned** if ``inputs`` is not ``None``.
    """
    coeffs = np.array([10, -72, 225, -400, 450, -360, 147]) / (60 * dt)
    return _finite_difference(states, coeffs, "bwd", inputs)


# Central differences ========================================================
def ctr2(states: np.ndarray, dt: float, inputs=None):
    r"""Second-order central difference for estimating the first derivative.

    .. math::
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_j}
       \approx \frac{1}{2\delta t}(\q(t_{j+1}) - \q(t_{j-1}))

    where :math:`\delta t = t_{j+1} - t_j` for all :math:`j`.

    Parameters
    ----------
    states : (r, k) ndarray
        State snapshots: ``states[:, j]`` is the state at time :math:`t_j`.
    dt : float
        Time step between snapshots.
    inputs : (m, k) or (k,) ndarray or None
        Inputs corresponding to the states, if applicable.

    Returns
    -------
    _states : (r, k - 2) ndarray
        State snapshots, excluding the first and last snapshots.
    ddts : (r, k - 2) ndarray
        Time derivative estimates corresponding to the state snapshots.
    _inputs : (m, k - 2) or (k - 2,) ndarray or None
        Inputs corresponding to ``_states``, if applicable.
        **Only returned** if ``inputs`` is not ``None``.
    """
    coeffs = np.array([-1, 0, 1]) / (2 * dt)
    return _finite_difference(states, coeffs, "ctr", inputs)


def ctr4(states: np.ndarray, dt: float, inputs=None):
    r"""Fourth-order central difference for estimating the first derivative.

    .. math::
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_j}
       \approx \frac{1}{12\delta t}
       (\q(t_{j-2}) - 8\q(t_{j-1}) + 8\q(t_{j+1}) - \q(t_{j+2}))

    where :math:`\delta t = t_{j+1} - t_j` for all :math:`j`.

    Parameters
    ----------
    states : (r, k) ndarray
        State snapshots: ``states[:, j]`` is the state at time :math:`t_j`.
    dt : float
        Time step between snapshots.
    inputs : (m, k) or (k,) ndarray or None
        Inputs corresponding to the states, if applicable.

    Returns
    -------
    _states : (r, k - 4) ndarray
        State snapshots, excluding the first two and last two snapshots.
    ddts : (r, k - 4) ndarray
        Time derivative estimates corresponding to the state snapshots.
    _inputs : (m, k - 4) or (k - 4,) ndarray or None
        Inputs corresponding to ``_states``, if applicable.
        **Only returned** if ``inputs`` is not ``None``.
    """
    coeffs = np.array([1, -8, 0, 8, -1]) / (12 * dt)
    return _finite_difference(states, coeffs, "ctr", inputs)


def ctr6(states: np.ndarray, dt: float, inputs=None):
    r"""Sixth-order central difference for estimating the first derivative.

    .. math::
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_j}
       \approx \frac{1}{60\delta t}
       (-\q(t_{j-3}) + 9\q(t_{j-2}) - 45\q(t_{j-1})
       + 45\q(t_{j+1})) - 9\q(t_{j+2}) + \q(t_{j+3})

    where :math:`\delta t = t_{j+1} - t_j` for all :math:`j`.

    Parameters
    ----------
    states : (r, k) ndarray
        State snapshots: ``states[:, j]`` is the state at time :math:`t_j`.
    dt : float
        Time step between snapshots.
    inputs : (m, k) or (k,) ndarray or None
        Inputs corresponding to the states, if applicable.

    Returns
    -------
    _states : (r, k - 6) ndarray
        State snapshots, excluding the first three and last three snapshots.
    ddts : (r, k - 6) ndarray
        Time derivative estimates corresponding to the state snapshots.
    _inputs : (m, k - 6) or (k - 6,) ndarray or None
        Inputs corresponding to ``_states``, if applicable.
        **Only returned** if ``inputs`` is not ``None``.
    """
    coeffs = np.array([-1, 9, -45, 0, 45, -9, 1]) / (60 * dt)
    return _finite_difference(states, coeffs, "ctr", inputs)


# Mixed differences ===========================================================
def _mixed_differences(states, dt, order, inputs):
    """Mixed finite differences on a uniform time domain."""
    ddts = ddt_uniform(states, dt, order=order)
    if inputs is not None:
        return states, ddts, inputs
    return states, ddts


def ord2(states: np.ndarray, dt: float, inputs=None):
    r"""Second-order forward, central, and backward differences for estimating
    the first derivative.

    Central differences are used where possible; forward differences are used
    for the first point and backward differences are used for the last point:

    .. math::
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_0}
       &\approx \frac{1}{2\delta t}(-3\q(t_0) + 4\q(t_{1}) - \q(t_{2})),
       \\ \\
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_j}
       &\approx \frac{1}{2\delta t}(-\q(t_{j-1}) + \q(t_{j+1})),
       \quad j = 2,\ldots,k-2,
       \\ \\
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_{k-1}}
       &\approx \frac{1}{2\delta t}(\q(t_{k-3}) - 4\q(t_{k-2}) + 3\q(t_{k-1}))

    where :math:`\delta t = t_{j+1} - t_j` for all :math:`j`.

    Parameters
    ----------
    states : (r, k) ndarray
        State snapshots: ``states[:, j]`` is the state at time :math:`t_j`.
    dt : float
        Time step between snapshots.
    inputs : (m, k) or (k,) ndarray or None
        Inputs corresponding to the states, if applicable.

    Returns
    -------
    states : (r, k) ndarray
        State snapshots.
    ddts : (r, k) ndarray
        Time derivative estimates corresponding to the state snapshots.
    inputs : (m, k) or (k,) ndarray or None
        Inputs corresponding to ``_states``, if applicable.
        **Only returned** if ``inputs`` is not ``None``.
    """
    return _mixed_differences(states, dt, 2, inputs)


def ord4(states: np.ndarray, dt: float, inputs=None):
    r"""Fourth-order forward, central, and backward differences for estimating
    the first derivative.

    Central differences are used where possible; forward differences are used
    for the first two points and backward differences are used for the last
    two points:

    .. math::
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_0}
       &\approx \frac{1}{12\delta t}(-25\q(t_0) + 48\q(t_{1}) - 36\q(t_{2})
       + 16\q(t_{3}) - 3\q(t_{4})),
       \\
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_1}
       &\approx \frac{1}{12\delta t}(-3\q(t_0) - 10\q(t_{1}) + 18\q(t_{2})
       - 6\q(t_{3}) + \q(t_{4})),
       \\ \\
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_j}
       &\approx \frac{1}{12\delta t}
       (\q(t_{j-2}) - 8\q(t_{j-1}) + 8\q(t_{j+1}) - \q(t_{j+2})),
       \quad j = 2,\ldots,k-3,
       \\ \\
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_{k-2}}
       &\approx \frac{1}{12\delta t}(-\q(t_{k-5}) + 6\q(t_{k-4})
       - 18\q(t_{k-3}) + 10\q(t_{k-2}) + 3\q(t_{k-1})),
       \\
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_{k-1}}
       &\approx \frac{1}{12\delta t}(3\q(t_{k-5}) - 16\q(t_{k-4})
       + 36\q(t_{k-3}) - 48\q(t_{k-2}) + 25\q(t_{k-1})),

    where :math:`\delta t = t_{j+1} - t_j` for all :math:`j`.


    Parameters
    ----------
    states : (r, k) ndarray
        State snapshots: ``states[:, j]`` is the state at time :math:`t_j`.
    dt : float
        Time step between snapshots.
    inputs : (m, k) or (k,) ndarray or None
        Inputs corresponding to the states, if applicable.

    Returns
    -------
    states : (r, k) ndarray
        State snapshots.
    ddts : (r, k) ndarray
        Time derivative estimates corresponding to the state snapshots.
    inputs : (m, k) or (k,) ndarray or None
        Inputs corresponding to ``_states``, if applicable.
        **Only returned** if ``inputs`` is not ``None``.
    """
    return _mixed_differences(states, dt, 4, inputs)


def ord6(states: np.ndarray, dt: float, inputs=None):
    r"""Sixth-order forward, central, and backward differences for estimating
    the first derivative.

    Central differences are used where possible; forward differences are used
    for the first three points and backward differences are used for the last
    three points:

    .. math::
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_0}
       &\approx \frac{1}{60\delta t}(
       -147\q(t_0) + 360\q(t_{1}) - 450\q(t_{2}) + 400\q(t_{3})
       - 225\q(t_{4}) + 72\q(t_{5}) - 10\q(t_{6})),
       \\
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_1}
       &\approx \frac{1}{60\delta t}(
       -10\q(t_0) - 77\q(t_{1}) + 150\q(t_{2}) - 100\q(t_{3})
       + 50\q(t_{4}) - 15\q(t_{5}) + 2\q(t_{6})),
       \\
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_2}
       &\approx \frac{1}{60\delta t}(
       2\q(t_0) - 24\q(t_{1}) - 35\q(t_{2}) + 80\q(t_{3})
       - 30\q(t_{4}) + 8\q(t_{5}) - \q(t_{6})),
       \\ \\
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_j}
       &\approx \frac{1}{60\delta t}
       (-\q(t_{j-3}) + 9\q(t_{j-2}) - 45\q(t_{j-1})
       + 45\q(t_{j+1})) - 9\q(t_{j+2}) + \q(t_{j+3}),
       \quad j = 3, \ldots, k - 4,
       \\ \\
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_{k-3}}
       &\approx \frac{1}{60\delta t}(
       \q(t_{k-7}) - 8\q(t_{k-6}) + 30\q(t_{k-5}) - 80\q(t_{k-4})
       + 35\q(t_{k-3}) + 24\q(t_{k-2}) - 2\q(t_{k-1})),
       \\
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_{k-2}}
       &\approx \frac{1}{60\delta t}(
       -2\q(t_{k-7}) + 15\q(t_{k-6}) - 50\q(t_{k-5}) + 100\q(t_{k-4})
       - 150\q(t_{k-3}) + 77\q(t_{k-2}) + 10\q(t_{k-1})),
       \\
       \frac{\textup{d}}{\textup{d}t}\q(t)\bigg|_{t = t_{k-1}}
       &\approx \frac{1}{60\delta t}(
       10\q(t_{k-7}) - 72\q(t_{k-6}) + 225\q(t_{k-5}) - 400\q(t_{k-4})
       + 450\q(t_{k-3}) - 360\q(t_{k-2}) + 147\q(t_{k-1})),

    where :math:`\delta t = t_{j+1} - t_j` for all :math:`j`.

    Parameters
    ----------
    states : (r, k) ndarray
        State snapshots: ``states[:, j]`` is the state at time :math:`t_j`.
    dt : float
        Time step between snapshots.
    inputs : (m, k) or (k,) ndarray or None
        Inputs corresponding to the states, if applicable.

    Returns
    -------
    states : (r, k) ndarray
        State snapshots.
    ddts : (r, k) ndarray
        Time derivative estimates corresponding to the state snapshots.
    inputs : (m, k) or (k,) ndarray or None
        Inputs corresponding to ``_states``, if applicable.
        **Only returned** if ``inputs`` is not ``None``.
    """
    return _mixed_differences(states, dt, 6, inputs)


# Class interface ============================================================
class UniformFiniteDifferencer(DerivativeEstimatorTemplate):
    r"""Time derivative estimation with finite differences for state snapshots
    spaced uniformly in time.

    Parameters
    ----------
    time_domain : (k,) ndarray
        Time domain corresponding to the snapshot data.
        This class requires uniformly spaced time domains, see
        :class:`NonuniformFiniteDifferencer` for non-uniform domains.
    scheme : str or callable
        Finite difference scheme to use.
        **Options:**

        * ``'fwd1'``: first-order forward differences, see :func:`fwd1`.
        * ``'fwd2'``: second-order forward differences, see :func:`fwd2`.
        * ``'fwd3'``: third-order forward differences, see :func:`fwd3`.
        * ``'fwd4'``: fourth-order forward differences, see :func:`fwd4`.
        * ``'fwd5'``: fifth-order forward differences, see :func:`fwd5`.
        * ``'fwd6'``: sixth-order forward differences, see :func:`fwd6`.
        * ``'bwd1'``: first-order backward differences, see :func:`bwd1`.
        * ``'bwd2'``: second-order backward differences, see :func:`bwd2`.
        * ``'bwd3'``: third-order backward differences, see :func:`bwd3`.
        * ``'bwd4'``: fourth-order backward differences, see :func:`bwd4`.
        * ``'bwd5'``: fifth-order backward differences, see :func:`bwd5`.
        * ``'bwd6'``: sixth-order backward differences, see :func:`bwd6`.
        * ``'ctr2'``: second-order backward differences, see :func:`ctr2`.
        * ``'ctr4'``: fourth-order backward differences, see :func:`ctr4`.
        * ``'ctr6'``: sixth-order backward differences, see :func:`ctr6`.
        * ``'ord2'``: second-order differences, see :func:`ord2`.
        * ``'ord4'``: fourth-order differences, see :func:`ord4`.
        * ``'ord6'``: sixth-order differences, see :func:`ord6`.

        If ``scheme`` is a callable function, its signature must match the
        following syntax.

        .. code-block:: python

           _states, ddts = scheme(states, dt)
           _states, ddts, _inputs = scheme(states, dt, inputs)

        Here ``dt`` is a positive float, the uniform time step.
        Each output should have the same number of columns.
    """

    _schemes = types.MappingProxyType(
        {
            "fwd1": fwd1,
            "fwd2": fwd2,
            "fwd3": fwd3,
            "fwd4": fwd4,
            "fwd5": fwd5,
            "fwd6": fwd6,
            "bwd1": bwd1,
            "bwd2": bwd2,
            "bwd3": bwd3,
            "bwd4": bwd4,
            "bwd5": bwd5,
            "bwd6": bwd6,
            "ctr2": ctr2,
            "ctr4": ctr4,
            "ctr6": ctr6,
            "ord2": ord2,
            "ord4": ord4,
            "ord6": ord6,
        }
    )

    def __init__(self, time_domain, scheme="ord4"):
        """Store the time domain and set the finite difference scheme."""
        DerivativeEstimatorTemplate.__init__(self, time_domain)

        # Check for uniform spacing.
        diffs = np.diff(time_domain)
        if not np.allclose(diffs, diffs[0]):
            raise ValueError("time domain must be uniformly spaced")

        # Set the finite difference scheme.
        if not callable(scheme):
            if scheme not in self._schemes:
                raise ValueError(
                    f"invalid finite difference scheme '{scheme}'"
                )
            scheme = self._schemes[scheme]
        self.__scheme = scheme

    # Properties --------------------------------------------------------------
    @property
    def dt(self):
        """Time step."""
        t = self.time_domain
        return t[1] - t[0]

    @property
    def scheme(self):
        """Finite difference engine."""
        return self.__scheme

    def __str__(self):
        """String representation: class name, time domain."""
        head = DerivativeEstimatorTemplate.__str__(self)
        tail = [f"time step: {self.dt:.2e}"]
        tail.append(f"finite difference scheme: {self.scheme.__name__}()")
        return f"{head}\n  " + "\n  ".join(tail)

    # Main routine ------------------------------------------------------------
    def estimate(self, states, inputs=None):
        r"""Estimate the first time derivatives of the states using
        finite differences.

        The stencil and order of the method are determined by the ``scheme``
        attribute.

        Parameters
        ----------
        states : (r, k) ndarray
            State snapshots, either full or (preferably) reduced.
        inputs : (m, k) ndarray or None
            Inputs corresponding to the state snapshots, if applicable.

        Returns
        -------
        _states : (r, k') ndarray
            Subset of the state snapshots.
        ddts : (r, k') ndarray
            First time derivatives corresponding to ``_states``.
        _inputs : (m, k') ndarray or None
            Inputs corresponding to ``_states``, if applicable.
            **Only returned** if ``inputs`` is provided.
        """
        states, inputs = self._check_dimensions(states, inputs, False)
        return self.scheme(states, self.dt, inputs)


class NonuniformFiniteDifferencer(DerivativeEstimatorTemplate):
    """Time derivative estimation with finite differences for state snapshots
    that are **not** spaced uniformly in time.

    This class essentially wraps :func:`numpy.gradient()`, which uses
    second-order finite differences.

    Parameters
    ----------
    time_domain : (k,) ndarray
        Time domain corresponding to the snapshot data.
        This class is for time domains that are not uniformly spaced, see
        :class:`UniformFiniteDifferencer` for uniformly spaced time domains.
    """

    def __init__(self, time_domain):
        """Set the time domain."""
        DerivativeEstimatorTemplate.__init__(self, time_domain)

        # Warn if time_domain in not uniform.
        if np.allclose(diffs := np.diff(time_domain), diffs[0]):
            warnings.warn(
                "time_domain is uniformly spaced, consider using "
                "UniformFiniteDifferencer",
                errors.OpInfWarning,
            )

    def __str__(self):
        """String representation: class name, time domain."""
        head = DerivativeEstimatorTemplate.__str__(self)
        tail = "finite difference engine: np.gradient(edge_order=2)"
        return f"{head}\n  {tail}"

    # Main routine ------------------------------------------------------------
    def estimate(self, states, inputs=None):
        r"""Estimate the first time derivatives of the states using
        second-order finite differences.

        Parameters
        ----------
        states : (r, k) ndarray
            State snapshots, either full or (preferably) reduced.
        inputs : (m, k) ndarray or None
            Inputs corresponding to the state snapshots, if applicable.

        Returns
        -------
        _states : (r, k) ndarray
            State snapshots.
        ddts : (r, k) ndarray
            First time derivatives corresponding to ``_states``.
        _inputs : (m, k) ndarray or None
            Inputs corresponding to ``_states``, if applicable.
            **Only returned** if ``inputs`` is provided.
        """
        states, inputs = self._check_dimensions(states, inputs)

        # Do the computation.
        ddts = np.gradient(states, self.time_domain, edge_order=2, axis=-1)

        if inputs is not None:
            return states, ddts, inputs
        return states, ddts


# Old API =====================================================================
def ddt_uniform(states, dt, order=2):
    """Forward, central, and backward differences for estimating the first
    derivative.

    Parameters
    ----------
    states : (r, k) ndarray
        State snapshots: ``states[:, j]`` is the state at time :math:`t_j`.
    dt : float
        Time step between snapshots.
    order : int {2, 4, 6}
        The order of the derivative approximation.

    Returns
    -------
    ddts : (r, k) ndarray
        Time derivative estimates corresponding to the state snapshots.
    """
    # Check dimensions and input types.
    if states.ndim != 2:
        raise errors.DimensionalityError("states must be two-dimensional")
    if not np.isscalar(dt):
        raise TypeError("time step dt must be a scalar (e.g., float)")

    ddts = np.empty_like(states)

    if order == 2:
        # TODO: Does this next line do what the rest does?
        # return np.gradient(states, dt, edge_order=2, axis=1)
        coeffs0 = np.array([-3, 4, -1]) / (2 * dt)
        # Forward differences on the front.
        ddts[:, 0] = states[:, :3] @ coeffs0
        # Central differences on the interior.
        _, ddts[:, 1:-1] = ctr2(states, dt, None)
        # Backward differences on the end.
        ddts[:, -1] = states[:, -3:] @ -coeffs0[::-1]

    elif order == 4:
        coeffs0 = np.array([-25, 48, -36, 16, -3]) / (12 * dt)
        coeffs1 = np.array([-3, -10, 18, -6, 1]) / (12 * dt)
        # Forward differences on the front.
        ddts[:, 0] = states[:, :5] @ coeffs0
        ddts[:, 1] = states[:, :5] @ coeffs1
        # Central differences on interior.
        _, ddts[:, 2:-2] = ctr4(states, dt, None)
        # Backward differences on the end.
        ddts[:, -2] = states[:, -5:] @ -coeffs1[::-1]
        ddts[:, -1] = states[:, -5:] @ -coeffs0[::-1]

    elif order == 6:
        coeffs0 = np.array([-147, 360, -450, 400, -225, 72, -10]) / (60 * dt)
        coeffs1 = np.array([-10, -77, 150, -100, 50, -15, 2]) / (60 * dt)
        coeffs2 = np.array([2, -24, -35, 80, -30, 8, -1]) / (60 * dt)
        # Forward differences on the front.
        ddts[:, 0] = states[:, :7] @ coeffs0
        ddts[:, 1] = states[:, :7] @ coeffs1
        ddts[:, 2] = states[:, :7] @ coeffs2
        # Central differences on interior.
        _, ddts[:, 3:-3] = ctr6(states, dt, None)
        # Backward differences on the end.
        ddts[:, -3] = states[:, -7:] @ -coeffs2[::-1]
        ddts[:, -2] = states[:, -7:] @ -coeffs1[::-1]
        ddts[:, -1] = states[:, -7:] @ -coeffs0[::-1]

    else:
        raise NotImplementedError(
            f"invalid order '{order}'; " "valid options: {2, 4, 6}"
        )

    return ddts


def ddt_nonuniform(states, t):
    """Second-order finite differences for estimating the first derivative.

    Parameters
    ----------
    states : (r, k) ndarray
        State snapshots: ``states[:, j]`` is the state at time :math:`t_j`.
    t : (k,) ndarray
        Time domain corresponding to the state snapshots.

    Returns
    -------
    ddts : (r, k) ndarray
        Time derivative estimates corresponding to the state snapshots.
    """
    # Check dimensions.
    if states.ndim != 2:
        raise errors.DimensionalityError("states must be two-dimensional")
    if t.ndim != 1:
        raise errors.DimensionalityError("time t must be one-dimensional")
    if states.shape[-1] != t.shape[0]:
        raise errors.DimensionalityError("states not aligned with time t")

    # Compute the derivative with a second-order difference scheme.
    return np.gradient(states, t, edge_order=2, axis=-1)


def ddt(states, *args, **kwargs):
    """Finite differences for estimating the first derivative.

    This is a convenience function that calls either :func:`ddt_uniform()` or
    :func:`ddt_nonuniform()`, depending on the arguments.

    Parameters
    ----------
    states : (r, k) ndarray
        State snapshots: ``states[:, j]`` is the state at time :math:`t_j`.
    dt : float
        Time step between snapshots (in the case of a time domain with
        uniform spacing).
    order : int {2, 4, 6} (optional)
        The order of the derivative approximation.
    t : (k,) ndarray
       Time domain corresponding to the state snapshots.
       May or may not be uniformly spaced.

    Returns
    -------
    ddts : (r, k) ndarray
        Time derivative estimates corresponding to the state snapshots.
    """
    n_args = len(args)  # Number of other positional args.
    n_kwargs = len(kwargs)  # Number of keyword args.
    n_total = n_args + n_kwargs  # Total number of other args.

    if n_total == 0:
        raise TypeError("at least one other argument required (dt or t)")
    elif n_total == 1:  # There is only one other argument.
        if n_kwargs == 1:  # It is a keyword argument.
            arg_name = list(kwargs.keys())[0]
            if arg_name == "dt":
                func = ddt_uniform
            elif arg_name == "t":
                func = ddt_nonuniform
            elif arg_name == "order":
                raise TypeError(
                    "keyword argument 'order' requires float " "argument dt"
                )
            else:
                raise TypeError(
                    f"ddt() got unexpected keyword argument '{arg_name}'"
                )
        elif n_args == 1:  # It is a positional argument.
            arg = args[0]
            if isinstance(arg, float):  # arg = dt.
                func = ddt_uniform
            elif isinstance(arg, np.ndarray):  # arg = t.
                func = ddt_nonuniform
            else:
                raise TypeError(f"invalid argument type '{type(arg)}'")
    elif n_total == 2:  # There are two other arguments: dt, order.
        func = ddt_uniform
    else:
        raise TypeError(
            "ddt() takes 2 or 3 positional arguments "
            f"but {n_total+1} were given"
        )

    return func(states, *args, **kwargs)


# Helper functions ============================================================
def _fdcoeffs(s):  # pragma: no cover
    r"""Vandermonde solve for finite difference coefficients.

    Parameters
    ----------
    s : (p,) ndarray
        Finite difference stencil. For example, ``s=[-2, -1, 0, 1]`` means
        we want coefficients :math:`c_{-2},c_{-1},c_{0},c_{1}` such that
        :math:`f'(x) \approx
        c_{-2} f(x - 2) + c_{-1} f(x - 1) + c_{0} f(x) + c_{1} f(x + 1)`.

    Returns
    -------
    (p,) ndarray
        Finite difference coefficients (the :math:`c_{j}`'s).
    """
    V = np.vander(s, increasing=True).T
    e = np.zeros_like(s)
    e[1] = 1
    return np.linalg.solve(V, e)


def _fdcoeffs2(s):  # pragma: no cover
    r"""Stable solve for finite difference coefficients (LeVeque).

    Parameters
    ----------
    s : (p,) ndarray
        Finite difference stencil. For example, ``[-2, -1, 0, 1]`` means
        get coefficients :math:`c_{-2},c_{-1},c_{0},c_{1}` such that
        :math:`f'(x) \approx
        c_{-2} f(x - 2) + c_{-1} f(x - 1) + c_{0} f(x) + c_{1} f(x + 1)`.

    Returns
    -------
    (p,) ndarray
        Finite difference coefficients (the :math:`c_{j}`'s).
    """
    pass
