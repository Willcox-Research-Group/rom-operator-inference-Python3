# model_selection/_tikhonov.py
"""Tikhonov regularization selection."""

import inspect
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection

from .._core._base import _BaseROM
from .._core._inferred import _InferredMixin


__MAXFUN = 1000                 # Artificial ceiling for optimization routine.


def _plot_Lcurve(regs, residuals, norms, discrete=False):

    fig, ax = plt.gcf(), plt.gca()

    if discrete:        # As discrete points
        colors = plt.cm.Spectral(np.linspace(0, 1, len(regs)))
        for reg, resid, norm, c in zip(regs, residuals, norms, colors):
            ax.loglog([resid], [norm], '.',
                      color=c, ms=20, label=fr"$\lambda = {reg:e}$")
        ax.legend()

    else:               # As continuous colored line
        x, y, z = np.array(residuals), np.array(norms), np.array(regs)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='Spectral_r',
                            norm=LogNorm(z.min(),z.max()))
        lc.set_array(z)
        lc.set_linewidth(3)
        line = ax.add_collection(lc)
        cbar = fig.colorbar(line, ax=ax, label=r"regularization $\lambda$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$||\mathbf{D}\mathbf{O}^\top - \dot{\hat{\mathbf{X}}}||_F^2$")
    ax.set_ylabel(r"$||\mathbf{O}^\top||_F^2$")
    ax.grid()


def Lcurve(rom, regs, fit_args, discrete=False):
    """Plot the L-curve (solution norm vs residual norm) for an Operator
    Inference ROM.

    Parameters
    ----------
    rom : Operator Inference ROM object
        Instance of a class inheriting from _BaseROM and _InferredMixin,
        e.g., InferredContinuousROM.

    regs : list(float)
        Regularization parameters to include in the L-curve.

    fit_args : list
        Ordered arguments for rom.fit() EXCEPT the regularization parameter.

    discrete : bool
        If True, make a scatter plot. If False, make a colored line plot.
    """
    # Validate inputs.
    if not isinstance(rom, _BaseROM) or not isinstance(rom, _InferredMixin):
        raise TypeError("rom must be Operator Inference ROM instance")

    # Get the residuals and norms for each regularization parameter.
    residuals, norms = [], []
    for reg in regs:
        rom.fit(*fit_args, reg, compute_extras=True)
        residuals.append(rom.solver_.misfit_)
        norms.append(np.sum(rom.O_**2))

    return _plot_Lcurve(regs, residuals, norms, discrete)


def best_bounded_reg(rom, B, errornorm, reg_bounds, fit_args, predict_args,
                     verbose=False):
    """Search for the regularization parameter that yields the ROM with the
    least training error while also satisfying a bound on the integrated POD
    coefficients.

    Parameters
    ----------
    rom : Operator Inference ROM object
        Instance of a class inheriting from _BaseROM and _InferredMixin,
        e.g., InferredContinuousROM.

    B : float > 0
        Bound that the integrated ROM states must satisfy.

    errornorm : callable -> float
        Function computing the error of the integrated ROM.

    reg_bounds : 2-tuple (float,float)
        Initial guess for the regularization parameter.

    fit_args : list
        Ordered arguments for rom.fit() EXCEPT the regularization parameter.

    predict_args : list
        Ordered arguments for rom.predict().

    verbose : bool
        If True, print progress on the optimization.

    Returns
    -------
    best_reg : float
        The best regularization parameter according to the minimization.

    rom : Operator Inference ROM object
        The ROM object, now trained with the best regularization parameter.

    Examples
    --------
    # Given POD basis Vr, snapshots X, time derivatives Xdot, inputs U,
    # an input function u(t), and a target time domain t.
    >>> import numpy as np
    >>> import scipy.linalg as la
    >>> import rom_operator_inference as roi
    >>> fit_args = [Vr, X, Xdot, U]
    >>> predict_args = [X[:,0], t, u]
    >>> Xnorm = la.norm(X)
    >>> def norm(Y):
    ...     return la.norm(X - Y[:,:X_.shape[1]]) / Xnorm
    >>> rom = roi.InferredContinuousROM("cAB")
    >>> B = 1.5 * np.abs(X_).max()
    >>> best_reg, rom = best_bounded_reg(rom, B, norm, (1e2,1e4),
    ...                                  fit_args, predict_args)

    # Given projected snapshots X_, projected time derivatives Xdot_, inputs U,
    # an input function u(t), and a target time domain t.
    >>> import numpy as np
    >>> import rom_operator_inference as roi
    >>> fit_args = [None, X_, Xdot_, U]
    >>> predict_args = [X_[:,0], t, u]
    >>> k = X_.shape[1]
    >>> def norm(Y_):
    ...     return roi.post.Lp_error(X_, Y_[:,:k], t[:k])[1]
    >>> rom = roi.InferredContinuousROM("cAB")
    >>> B = 1.5 * np.abs(X_).max()
    >>> best_reg, rom = best_bounded_reg(rom, B, norm, (1e2,1e4),
    ...                                  fit_args, predict_args)
    """
    # Validate inputs.
    if not isinstance(rom, _BaseROM) or not isinstance(rom, _InferredMixin):
        raise TypeError("rom must be Operator Inference ROM instance")
    if not callable(errornorm):
        raise TypeError("errornorm must be callable")

    # Buffer *_args up to final argument.
    nargs_fit = len(inspect.signature(rom.fit).parameters)
    nargs_predict = len(inspect.signature(rom.predict).parameters)
    while len(fit_args) < nargs_fit - 2:
        fit_args += [None]
    while len(predict_args) < nargs_predict - 1:
        predict_args += [None]

    # Define the subroutine to optimize.
    def _training_error(log10reg):
        """Return the training error of the ROM trained with regularization
        parameter reg.
        """
        reg = 10**log10reg
        if verbose:
            print(f"Processing reg={reg:e}...", end='')

        # Train the ROM with the given regularization parameter.
        try:
            rom.fit(*fit_args, P=reg)
        except (np.linalg.LinAlgError, ValueError) as e:    # pragma: nocover
            if e.args[0] in [               # Near-singular data matrix.
                "SVD did not converge in Linear Least Squares",
                "On entry to DLASCL parameter number 4 had an illegal value"
            ]:
                return __MAXFUN
            else:
                raise

        # Simulate the ROM over the specified time domain.
        with np.warnings.catch_warnings():
            np.warnings.simplefilter("ignore")
            x_rom = rom.predict(*predict_args)

        # Check for boundedness of solution.
        if np.abs(x_rom).max() > B:                         # pragma: nocover
            if verbose:
                print("ROM violates bound")
            return __MAXFUN
        elif verbose:
            print("done")

        # Return the error of the prediction.
        return errornorm(x_rom)

    # Do the optimization and unpack the results.
    opt_result = opt.minimize_scalar(_training_error,
                                     method="bounded",
                                     bounds=reg_bounds)
    if opt_result.success and opt_result.fun != __MAXFUN:
        best_reg = 10 ** opt_result.x
        rom.fit(*fit_args, P=best_reg)
        return best_reg, rom
    else:                                                   # pragma: nocover
        print(f"Regularization optimization FAILED")
