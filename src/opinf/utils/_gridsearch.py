# _gridsearch.py
"""Routine for performing a grid search, followed by a derivative-free
optimization.
"""

__all__ = [
    "gridsearch",
]

import typing
import warnings
import numpy as np
import scipy.optimize

from .. import errors


__MAXOPTVAL = 1e20


def gridsearch(
    func: typing.Callable,
    candidates: np.ndarray,
    gridsearch_only: bool = False,
    label: str = "parameter",
    verbose: bool = False,
) -> typing.Union[float, np.ndarray]:
    r"""Minimize a function by first checking a collection of candidates, then
    following up with a derivative-free optimization routine.

    If the candidates are one-dimensional, meaning ``func()`` is a scalar map
    :math:`f:\RR_+\to\RR`, the optimization is carried out via
    :func:`scipy.optimize.minimize_scalar` with ``method='brent'``.

    Otherwise (:math:`f:\RR_+^p\to\RR` for some :math:`p > 1`), the
    optimization is carried out via :func:`scipy.optimize.minimize` with
    ``method='Nelder-Mead'``.

    Parameters
    ----------
    func : callable(candidate) -> float
        Function to minimize.
    candidates : list of floats or ndarrays
        Candidates to check before starting the optimization routine.
    gridsearch_only : bool
        If ``True``, stop after checking all ``candidates`` and do not follow
        up with optimization.
    label : str
        Label for the types of candidates being checked.
    verbose : bool
        If ``True``, print information at each iteration.

    Returns
    -------
    winner : float or ndarray
        Optimized candidate.

    Raises
    ------
    RuntimeError
        If the grid search fails because ``func()`` returns ``np.inf`` for
        each candidate.

    Warns
    -----
    OpInfWarning
        1) If the ``candidates`` are one-dimensional and the grid search winner
        is the smallest or largest candidate, or 2) If the minimization fails
        or does not result in a solution that improves upon the grid search.

    Notes
    -----
    The optimization minimizes ``func(x)`` by varying ``x`` logarithmically.
    Unless ``gridsearch_only=True``, it is assumed that all entries of the
    argument ``x`` are positive.

    If only one scalar candidate is given, the (one-dimensional) optimization
    sets the bounds to ``[candidate / 100, candidate * 100]``.
    """
    # Process the candidates.
    candidates = np.atleast_1d(candidates)
    if np.shape(candidates[0]) == (1,):
        candidates = [reg[0] for reg in candidates]
    scalar_regularization = np.ndim(candidates[0]) == 0

    def pstr(params):
        if np.isscalar(params):
            return f"{params:.4e}"
        else:
            return "[" + ", ".join([f"{p:.3e}" for p in params]) + "]"

    def linfunc(params):
        if verbose:
            print(f"{label.title()} {pstr(params)}...", end="", flush=True)
        out = func(params)
        if verbose:
            if out == np.inf:
                print("UNSTABLE")
            else:
                print(f"{out:.8%} error")
        return out

    def logfunc(log10params):
        params = 10**log10params
        return min(__MAXOPTVAL, linfunc(params))

    # Grid search.
    num_tests = len(candidates)
    if verbose:
        print(f"\nGRID SEARCH ({num_tests} candidates)")
        ndigits = len(str(num_tests))
    winning_error, winner_index = np.inf, None
    if scalar_regularization:
        candidates = np.sort(candidates)
    for i, candidate in enumerate(candidates):
        if verbose:
            print(f"({i+1: >{ndigits}d}/{num_tests:d}) ", end="")
        if (error := linfunc(candidate)) < winning_error:
            winning_error = error
            winner_index = i
    if winner_index is None:
        raise RuntimeError(f"{label} grid search failed")
    gridsearch_winner = candidates[winner_index]
    if verbose:
        print(
            f"Best {label} candidate via grid search:",
            pstr(gridsearch_winner),
        )
    if gridsearch_only:
        return gridsearch_winner

    # Post-process results if the candidates are scalars.
    if scalar_regularization:
        if num_tests == 1:
            search_bounds = [gridsearch_winner / 1e2, 1e2 * gridsearch_winner]
        elif winner_index == 0:
            warnings.warn(
                f"smallest {label} candidate won grid search, "
                f"consider using smaller candidates",
                errors.OpInfWarning,
            )
            search_bounds = [gridsearch_winner / 1e2, candidates[1]]
        elif winner_index == num_tests - 1:
            warnings.warn(
                f"largest {label} candidate won grid search, "
                f"consider using larger candidates",
                errors.OpInfWarning,
            )
            search_bounds = [candidates[-2], 1e2 * gridsearch_winner]
        else:
            search_bounds = [
                candidates[winner_index - 1],
                candidates[winner_index + 1],
            ]

    # Follow up grid search with minimization-based search.
    if verbose:
        print("OPTIMIZATION")
    if scalar_regularization:
        opt_result = scipy.optimize.minimize_scalar(
            logfunc,
            method="bounded",
            bounds=np.log10(search_bounds),
        )
    else:
        opt_result = scipy.optimize.minimize(
            logfunc,
            x0=np.log10(gridsearch_winner),
            method="Nelder-Mead",
        )

    # Report results.
    success = opt_result.success and opt_result.fun != __MAXOPTVAL
    if not success or (winning_error < opt_result.fun):
        warnings.warn(
            f"{label} grid search performed better than optimization, "
            "falling back on grid search solution",
            errors.OpInfWarning,
        )
        return gridsearch_winner

    optimization_winner = 10**opt_result.x

    if verbose:
        print(
            f"Best {label} candidate via optimization:",
            pstr(optimization_winner),
        )
    return optimization_winner
