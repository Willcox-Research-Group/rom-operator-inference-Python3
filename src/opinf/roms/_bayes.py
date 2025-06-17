# rom/_bayes.py
"""Classes supporting Bayesian operator inference."""

__all__ = [
    "OperatorPosterior",
    "BayesianROM",
]

import warnings
import numpy as np
import scipy.stats
import scipy.linalg

from .. import errors, lstsq, post, utils
from ..models import _utils as modutils
from ._base import _identity
from ._nonparametric import ROM


VALID_SOLVERS = (
    lstsq.L2Solver,
    lstsq.L2DecoupledSolver,
    lstsq.TikhonovSolver,
    lstsq.TikhonovDecoupledSolver,
)


# Posterior ===================================================================
class OperatorPosterior:
    r"""Posterior distribution for operator matrices.

    Operator inference models are uniquely determined by operator matrices
    :math:`\Ohat\in\RR^{r\times d}` that concatenate the entries of all
    operators in the model. For example, the time-continuous model

    .. math::
       \ddt\qhat(t) = \chat + \Ahat\qhat(t) + \Hhat[\qhat(t)\otimes\qhat(t)]

    is uniquely determined by the operator matrix

    .. math::
       \Ohat = [~\chat~~\Ahat~~\Hhat~] \in \RR^{r \times d}.

    Typical *deterministic* operator inference learns a single operator matrix
    :math:`\Ohat` from state measurements, while *probabilistic* or *Bayesian*
    operator inference constructs a distribution of operator matrices,
    :math:`p(\Ohat)`. This class implements an operator matrix distribution
    where the rows of :math:`\Ohat` are multivariate Normal (Gaussian) random
    variables, i.e.,

    .. math::
       p(\ohat_{i}) = \mathcal{N}(\ohat_i\mid\bfmu_i,\bfSigma_i),
       \\
       \bfmu_i \in \RR^{d},
       \quad
       \bfSigma_i \in \RR^{d\times d},
       \quad
       i = 0, \ldots, r-1,

    where :math:`\ohat_i \in \RR^{d}` is the :math:`i`-th row of :math:`\Ohat`.

    The :class:`BayesianROM` class has a ``posterior`` attribute that is an
    ``OperatorPosterior`` object.

    Parameters
    ----------
    means : list of r (d,) ndarrays
        Mean values for each row of the operator matrix.
    precisions : list of r (d, d) ndarrays
        **INVERSE** covariance matrices for each row of the operator matrix.
    alreadyinverted : bool
        If ``True``, assume ``precisions`` is the collection of covariance
        matrices, not their inverses.
    """

    def __init__(self, means, precisions, *, alreadyinverted=False):
        """Store and pre-process the distribution parameters."""
        if (r := len(means)) != (_r2 := len(precisions)):
            raise ValueError(f"len(means) = {r} != {_r2} = len(precisions)")

        self.__r = r
        self.__randomvariables = []

        for i in range(self.__r):
            # Verify dimensions.
            mean_i, cov_i = means[i], precisions[i]
            if not isinstance(mean_i, np.ndarray) or mean_i.ndim != 1:
                raise ValueError(f"means[{i}] should be a 1D ndarray")
            if not isinstance(cov_i, np.ndarray) or cov_i.ndim != 2:
                raise ValueError(f"precisions[{i}] should be a 2D ndarray")
            d = mean_i.shape[0]
            if cov_i.shape != (d, d):
                raise ValueError(f"means[{i}] and precisions[{i}] not aligned")

            # Make a multivariate Normal distribution for this operator row.
            if not alreadyinverted:
                cov_i = scipy.stats.Covariance.from_precision(cov_i)
            self.__randomvariables.append(
                scipy.stats.multivariate_normal(mean=mean_i, cov=cov_i)
            )

        # If operator rows are all the same size, wrap rvs() output as array.
        self.__rvsasarray = False
        d = means[0].size
        if all(mean.size == d for mean in means):
            self.__rvsasarray = True

    # Properties --------------------------------------------------------------
    @property
    def nrows(self) -> int:
        """Number of rows :math:`r` in the data matrix. This is also the state
        dimension of the corresponding model.
        """
        return self.__r

    @property
    def randomvariables(self) -> list:
        """Multivariate normal random variables for the rows of the operator
        matrix.
        """
        return self.__randomvariables

    @property
    def means(self) -> list:
        r"""Mean vectors :math:`\bfmu_0,\ldots,\bfmu_{r-1}\in\RR^{d}` for the
        rows of the operator matrix.
        """
        return [rv.mean for rv in self.randomvariables]

    @property
    def covs(self) -> list:
        r"""Covariance matrices
        :math:`\bfSigma_0,\ldots,\bfSigma_{r-1}\in\RR^{d\times d}`
        for the rows of the operator matrix.
        """
        return [rv.cov for rv in self.randomvariables]

    def __eq__(self, other):
        if self.nrows != other.nrows:
            return False
        for m1, m2 in zip(self.means, other.means):
            if m1.shape != m2.shape or not np.all(m1 == m2):
                return False
        for C1, C2 in zip(self.covs, other.covs):
            if C1.shape != C2.shape or not np.all(C1 == C2):
                return False
        return True

    # Random draws ------------------------------------------------------------
    def rvs(self):
        r"""Draw a random operator matrix from the posterior operator
        distribution.

        Returns
        -------
        Ohat : (r, d) ndarray
            Operator matrix sampled from :math:`p(\Ohat)`.
        """
        ohats = [rv.rvs()[0] for rv in self.randomvariables]
        return np.array(ohats) if self.__rvsasarray else ohats

    # Model persistance -------------------------------------------------------
    def save(self, savefile, overwrite=True):
        """Save the posterior operator distribution.

        Parameters
        ----------
        savefile : str
            File to save data to.
        overwrite : bool
            If False and ``savefile`` exists, raise an exception.
        """
        with utils.hdf5_savehandle(savefile, overwrite) as hf:
            hf.create_dataset("state_dimension", data=[self.nrows])
            for i, (mean_i, cov_i) in enumerate(zip(self.means, self.covs)):
                hf.create_dataset(f"means_{i}", data=mean_i)
                hf.create_dataset(f"covs_{i}", data=cov_i)

    @classmethod
    def load(cls, loadfile):
        """Load a previously saved posterior operator distribution.

        Parameters
        ----------
        loadfile : str
            File to load data from.
        """
        with utils.hdf5_loadhandle(loadfile) as hf:
            r = int(hf["state_dimension"][0])
            means = [hf[f"means_{i}"][:] for i in range(r)]
            covs = [hf[f"covs_{i}"][:] for i in range(r)]

        return cls(means, covs, alreadyinverted=True)


# Bayesian ROMs ===============================================================
class _BayesianROMMixin:
    """Mixin for ROM classes with a Bayesian operator posterior."""

    def __init__(self):
        self.__posterior = None
        self._validate_model_solver(self.model)

    @staticmethod
    def _validate_model_solver(model):
        if modutils.is_interpolatory(model):
            raise AttributeError(
                "Fully interpolatory parametric models are not supported "
                "for Bayesian ROMs"
            )
        if not hasattr(model, "solver") or not isinstance(
            model.solver, VALID_SOLVERS
        ):
            types = ", ".join(f"lstsq.{s.__name__}" for s in VALID_SOLVERS)
            raise AttributeError(
                "'model' must have a 'solver' attribute "
                f"of one of the following types: {types}"
            )

    @property
    def posterior(self) -> OperatorPosterior:
        """Posterior distribution for the operator matrices."""
        return self.__posterior

    def _initialize_posterior(self):
        """Set the operator posterior if numerically possible."""
        try:
            means, precisions = self.model.solver.posterior()
            self.__posterior = OperatorPosterior(means, precisions)
        except np.linalg.LinAlgError:
            self.__posterior = None

    def draw_operators(self):
        """Set the :attr:`model` operators to a new random draw from the
        :attr:`posterior` operator distribution.
        """
        self.model._extract_operators(self.posterior.rvs())

    def fit_regselect_continuous(
        self,
        candidates: list,
        train_time_domains: list,
        parameters: list,
        states: list,
        ddts: list = None,
        input_functions: list = None,
        fit_transformer: bool = True,
        fit_basis: bool = True,
        regularizer_factory=None,
        gridsearch_only: bool = False,
        test_time_length: float = 0,
        stability_margin: float = 5.0,
        num_posterior_draws: int = 20,
        test_cases: list = None,
        verbose: bool = False,
        **predict_options: dict,
    ):
        if not self._iscontinuous:
            raise AttributeError(
                "this method is for time-continuous models only, "
                "use fit_regselect_discrete()"
            )

        if parameters is None:
            states, ddts, input_functions = self._fix_single_trajectory(
                states, ddts, input_functions
            )

        # Validate arguments.
        if np.isscalar(train_time_domains[0]):
            train_time_domains = [train_time_domains] * len(states)
        for t, Q in zip(train_time_domains, states):
            if t.shape != (Q.shape[1],):
                raise errors.DimensionalityError(
                    "train_time_domains and states not aligned"
                )
        if input_functions is not None:
            if callable(input_functions):  # one global input function.
                input_functions = [input_functions] * len(states)
            if not callable(input_functions[0]):
                raise TypeError(
                    "argument 'input_functions' must be sequence of callables"
                )
            inputs = [  # evaluate the inputs over the time domain.
                np.column_stack([u(tt) for tt in t])
                for u, t in zip(input_functions, train_time_domains)
            ]
        else:
            inputs = None
        if test_time_length < 0:
            raise ValueError("argument 'test_time_length' must be nonnegative")
        if regularizer_factory is None:
            regularizer_factory = _identity

        # Fit the model for the first time.
        if hasattr(self.model.solver, "reset"):
            self.model.solver.reset()
        self._fit_model(
            parameters=parameters,
            states=states,
            lhs=ddts,
            inputs=inputs,
            fit_transformer=fit_transformer,
            fit_basis=fit_basis,
            solver_only=True,
        )

        # Set up the regularization selection.
        states_ = [self.encode(Q) for Q in states]
        shifts, limits = self._get_stability_limits(states_, stability_margin)
        processed_test_cases = self._process_test_cases(
            test_cases, utils.ContinuousRegTest
        )

        def unstable(_Q, ell, size):
            """Return ``True`` if the solution is unstable."""
            if _Q.shape[-1] != size:
                return True
            return np.abs(_Q - shifts[ell]).max() > limits[ell]

        # Extend the training time domains by the testing time length.
        if test_time_length > 0:
            time_domains = []
            for t_train in train_time_domains:
                dt = np.mean(np.diff(t_train))
                t_test = t_train[-1] + np.linspace(
                    dt,
                    dt + test_time_length,
                    int(test_time_length / dt),
                )
                time_domains.append(np.concatenate(((t_train, t_test))))
        else:
            time_domains = train_time_domains

        if input_functions is None:
            input_functions = [None] * len(states_)
        loop_collections = [states_, input_functions, time_domains]
        if is_parametric := parameters is not None:
            loop_collections.insert(0, parameters)

        def update_model(reg_params):
            """Reset the regularizer and refit the model operators."""
            self.model.solver.regularizer = regularizer_factory(reg_params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", scipy.linalg.LinAlgWarning)
                self._initialize_posterior()

        def training_error(reg_params):
            """Compute the training error for a single regularization
            candidate by solving the model, checking for stability, and
            comparing to available training data.
            """
            update_model(reg_params)
            if self.posterior is None:
                return np.inf

            # Pass stability checks.
            for tcase in processed_test_cases:
                for _ in range(num_posterior_draws):
                    self.draw_operators()
                    if not tcase.evaluate(self.model, **predict_options):
                        return np.inf

            # Compute training error.
            error = 0
            for ell, entries in enumerate(zip(*loop_collections)):
                if is_parametric:
                    params, Q, input_func, t = entries
                    predict_args = (params, Q[:, 0], t, input_func)
                else:
                    Q, input_func, t = entries
                    predict_args = (Q[:, 0], t, input_func)
                draws = []
                trainsize = Q.shape[-1]
                for _ in range(num_posterior_draws):
                    self.draw_operators()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        solution = self.model.predict(
                            *predict_args, **predict_options
                        )
                    if unstable(solution, ell, t.size):
                        return np.inf
                    draws.append(solution[:, :trainsize])
                solution_train = np.mean(draws, axis=0)
                error += post.Lp_error(Q, solution_train, t[:trainsize])[1]
            return error / len(states)

        best_regularization = utils.gridsearch(
            training_error,
            candidates,
            gridsearch_only=gridsearch_only,
            label="regularization",
            verbose=verbose,
        )

        update_model(best_regularization)
        return self

    def fit_regselect_discrete(
        self,
        candidates: list,
        parameters: list,
        states: list,
        inputs: list = None,
        fit_transformer: bool = True,
        fit_basis: bool = True,
        regularizer_factory=None,
        gridsearch_only: bool = False,
        num_test_iters: int = 0,
        stability_margin: float = 5.0,
        num_posterior_draws: int = 20,
        test_cases: list = None,
        verbose: bool = False,
    ):
        if self._iscontinuous:
            raise AttributeError(
                "this method is for fully discrete models only, "
                "use fit_regselect_continuous()"
            )

        if parameters is None:
            states, _, inputs = self._fix_single_trajectory(
                states, None, inputs
            )

        # Validate arguments.
        if num_test_iters < 0:
            raise ValueError(
                "argument 'num_test_iters' must be a nonnegative integer"
            )
        if inputs is not None:
            if len(inputs) != len(states):
                raise errors.DimensionalityError(
                    f"{len(states)} state trajectories but "
                    f"{len(inputs)} input trajectories detected"
                )
            for Q, U in zip(states, inputs):
                if U.shape[-1] < Q.shape[1] + num_test_iters:
                    raise ValueError(
                        "argument 'inputs' must contain enough data for "
                        f"{num_test_iters} iterations after the training data"
                    )
        if regularizer_factory is None:
            regularizer_factory = _identity

        # Fit the model for the first time.
        if hasattr(self.model.solver, "reset"):
            self.model.solver.reset()
        states_ = self._fit_model(
            parameters=parameters,
            states=states,
            lhs=None,
            inputs=inputs,
            fit_transformer=fit_transformer,
            fit_basis=fit_basis,
            solver_only=True,
        )

        # Set up the regularization selection.
        shifts, limits = self._get_stability_limits(states_, stability_margin)
        processed_test_cases = self._process_test_cases(
            test_cases, utils.DiscreteRegTest
        )

        def unstable(_Q, ell):
            """Return ``True`` if the solution is unstable."""
            if np.isnan(_Q).any() or np.isinf(_Q).any():
                return True
            return np.any(np.abs(_Q - shifts[ell]).max() > limits[ell])

        # Extend the iteration counts by the number of testing iterations.
        num_iters = [Q.shape[-1] for Q in states_]
        if num_test_iters > 0:
            num_iters = [n + num_test_iters for n in num_iters]

        if inputs is None:
            inputs = [None] * len(states_)
        loop_collections = [states_, inputs, num_iters]
        if is_parametric := parameters is not None:
            loop_collections.insert(0, parameters)

        def update_model(reg_params):
            """Reset the regularizer and refit the model operators."""
            self.model.solver.regularizer = regularizer_factory(reg_params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", scipy.linalg.LinAlgWarning)
                self._initialize_posterior()

        def training_error(reg_params):
            """Compute the mean training error for a single regularization
            candidate by solving the model, checking for stability, and
            comparing to available training data.
            """
            update_model(reg_params)
            if self.posterior is None:
                return np.inf

            # Pass stability checks.
            for tcase in processed_test_cases:
                for _ in range(num_posterior_draws):
                    self.draw_operators()
                    if not tcase.evaluate(self.model):
                        return np.inf

            # Solve the model, check for stability, and compute training error.
            error = 0
            for ell, entries in enumerate(zip(*loop_collections)):
                if is_parametric:
                    params, Q, U, niter = entries
                    predict_args = (params, Q[:, 0], niter, U)
                else:
                    Q, U, niter = entries
                    predict_args = (Q[:, 0], niter, U)
                draws = []
                for _ in range(num_posterior_draws):
                    self.draw_operators()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        solution = self.model.predict(*predict_args)
                    if unstable(solution, ell):
                        return np.inf
                    draws.append(solution[:, : Q.shape[-1]])
                error += post.frobenius_error(Q, np.mean(draws, axis=0))[1]
            return error / len(states_)

        best_regularization = utils.gridsearch(
            training_error,
            candidates,
            gridsearch_only=gridsearch_only,
            label="regularization",
            verbose=verbose,
        )

        update_model(best_regularization)
        return self


class BayesianROM(ROM, _BayesianROMMixin):
    r"""Probabilistic nonparametric reduced-order model.

    This class connects classes from the various submodules to form a complete
    reduced-order modeling workflow for probabilistic models.

    High-dimensional data
    :math:`\to` transformed / preprocessed data
    :math:`\to` compressed data
    :math:`\to` low-dimensional probabilistic model.

    Operator inference models are uniquely determined by operator matrices
    :math:`\Ohat\in\RR^{r\times d}` that concatenate the entries of all
    operators in the model. For example, the time-continuous model

    .. math::
       \ddt\qhat(t) = \chat + \Ahat\qhat(t) + \Hhat[\qhat(t)\otimes\qhat(t)]

    is uniquely determined by the operator matrix

    .. math::
       \Ohat = [~\chat~~\Ahat~~\Hhat~] \in \RR^{r \times d}.

    Typical *deterministic* operator inference learns a single operator matrix
    :math:`\Ohat` from state measurements, while *probabilistic* or *Bayesian*
    operator inference constructs a distribution of operator matrices,
    :math:`p(\Ohat)`. This class solves a Bayesian linear inference to define
    an :class:`OperatorPosterior` and facilitates sampling from the posterior.
    See :cite:`guo2022bayesopinf`.

    Parameters
    ----------
    model : :mod:`opinf.models` object
        Nonparametric system model, an instance of one of the following:

        * :class:`opinf.models.ContinuousModel`
        * :class:`opinf.models.DiscreteModel`

        The model must have a ``solver`` of one of the following types:

        * :class:`opinf.lstsq.L2Solver`
        * :class:`opinf.lstsq.L2DecoupledSolver`
        * :class:`opinf.lstsq.TikhonovSolver`
        * :class:`opinf.lstsq.TikhonovDecoupledSolver`

    lifter : :mod:`opinf.lift` object or None
        Lifting transformation.
    transformer : :mod:`opinf.pre` object or None
        Preprocesser.
    basis : :mod:`opinf.basis` object or None
        Dimensionality reducer.
    ddt_estimator : :mod:`opinf.ddt` object or None
        Time derivative estimator.
        Ignored if ``model`` is not time continuous.

    Notes
    -----
    The ``operators`` attribute of the :attr:`model` represents a single draw
    from the operator distribution and is modified every time
    :meth:`draw_operators` or :meth:`predict` are called.
    """

    def __init__(
        self,
        model,
        *,
        lifter=None,
        transformer=None,
        basis=None,
        ddt_estimator=None,
    ):
        ROM.__init__(
            self,
            model,
            lifter=lifter,
            transformer=transformer,
            basis=basis,
            ddt_estimator=ddt_estimator,
        )
        _BayesianROMMixin.__init__(self)

    def fit(
        self,
        states,
        lhs=None,
        inputs=None,
        fit_transformer: bool = True,
        fit_basis: bool = True,
    ):
        ROM.fit(
            self,
            states=states,
            lhs=lhs,
            inputs=inputs,
            fit_transformer=fit_transformer,
            fit_basis=fit_basis,
        )
        self._initialize_posterior()
        return self

    def fit_regselect_continuous(
        self,
        candidates: list,
        train_time_domains: list,
        states: list,
        ddts: list = None,
        input_functions: list = None,
        fit_transformer: bool = True,
        fit_basis: bool = True,
        regularizer_factory=None,
        gridsearch_only: bool = False,
        test_time_length: float = 0,
        stability_margin: float = 5,
        num_posterior_draws: int = 20,
        test_cases: list = None,
        verbose: bool = False,
        **predict_options: dict,
    ):
        """Calibrate the time-continuous model to training data, selecting the
        regularization hyperparameter(s) that minimize the sample mean training
        error while maintaining stability over the testing regime.

        This method requires the :attr:`model` to be time-continuous; use
        :meth:`fit_regselect_discrete` for discrete models. The
        ``model.solver.regularizer`` is repeatedly adjusted, and the operator
        :attr:`posterior` is recalibrated, until a best regularization is
        selected. Training error is measured by comparing training data to the
        sample mean of ``num_posterior_draws`` model predictions. Stability is
        required for each of the individual model predictions.
        See :cite:`mcquarrie2021combustion,guo2022bayesopinf`.

        Parameters
        ----------
        candidates : list of regularization hyperparameters
            Regularization hyperparameters to check before carrying out a
            derivative-free optimization.
        train_time_domains : list of s (k_i,) ndarrays
            Time domain corresponding to the training states.
        states : list of s (n, k_i) ndarrays
            State snapshots in the original state space. Each array
            ``states[i]`` is data corresponding to a different trajectory;
            each column ``states[i][:, j]`` is one snapshot.
        ddts : list of s (n, k_i) ndarrays or None
            Snapshot time derivative data. Each array ``ddts[i]`` are the time
            derivatives of ``states[i]``; each column ``ddts[i][:, j]``
            corresponds to the snapshot ``states[i][:, j]``. If ``None``
            (default), these are estimated using :attr:`ddt_estimator`.
        input_functions : list of s callables or None
            Input functions mapping time to input vectors. Only required if the
            :attr:`model` takes external inputs. Each ``input_functions[i]``
            is the function corresponding to ``states[i]``, and
            ``input_functions[i](train_time_domains[i][j])`` is the input
            vector corresponding to the snapshot ``states[i][:, j]``.
        fit_transformer : bool
            If ``True`` (default), calibrate the preprocessing transformation
            using the ``states``.
            If ``False``, assume the transformer is already calibrated.
        fit_basis : bool
            If ``True`` (default), calibrate the high-to-low dimensional
            mapping using the ``states``.
            If ``False``, assume the basis is already calibrated.
        regularizer_factory : callable or None
            Function mapping regularization hyperparameters to the full
            regularizer. Specifically, ``regularizer_factory(candidates[i])``
            will be assigned to ``model.solver.regularizer`` for each ``i``.
            If ``None`` (default), set ``regularizer_factory()`` to the
            identity function.
        gridsearch_only : bool
            If ``True``, stop after checking all regularization ``candidates``
            and do not follow up with optimization.
        test_time_length : float or None
            Amount of time after the training regime in which to require model
            stability.
        stability_margin : float
            Factor by which the predicted reduced states may deviate from the
            range of the training reduced states without the trajectory being
            classified as unstable.
        num_posterior_draws : int
            Number of draws from the operator :attr:`posterior` for stability
            checks and for estimating the sample mean of model predictions.
        test_cases : list of ContinuousRegTest objects
            Additional test cases for which the model is required to be stable.
            See :class:`opinf.utils.ContinuousRegTest`.
        verbose : bool
            If ``True``, print information during the regularization selection.
        predict_options : dict or None
            Extra arguments for :meth:`opinf.models.ContinuousModel.predict`.

        Notes
        -----
        If there is only one trajectory of training data (s = 1), ``states``
        may be provided as an (n, k) ndarray. In this case, it is assumed that
        ``ddts`` (if provided) is an (n, k) ndarray.

        The ``train_time_domains`` may be a single one-dimensional array, in
        which case it is assumed that each trajectory ``states[i]`` corresponds
        to the same time domain. Similarly, if ``input_functions`` is a single
        callable, it is assumed to be the input function for each trajectory.
        """
        return _BayesianROMMixin.fit_regselect_continuous(
            self,
            candidates=candidates,
            train_time_domains=train_time_domains,
            parameters=None,
            states=states,
            ddts=ddts,
            input_functions=input_functions,
            fit_transformer=fit_transformer,
            fit_basis=fit_basis,
            regularizer_factory=regularizer_factory,
            gridsearch_only=gridsearch_only,
            test_time_length=test_time_length,
            stability_margin=stability_margin,
            num_posterior_draws=num_posterior_draws,
            test_cases=test_cases,
            verbose=verbose,
            **predict_options,
        )

    def fit_regselect_discrete(
        self,
        candidates: list,
        states: list,
        inputs: list = None,
        fit_transformer: bool = True,
        fit_basis: bool = True,
        regularizer_factory=None,
        gridsearch_only: bool = False,
        num_test_iters: int = 0,
        stability_margin: float = 5,
        num_posterior_draws: int = 20,
        test_cases: list = None,
        verbose: bool = False,
    ):
        """Calibrate the fully discrete model to training data, selecting the
        regularization hyperparameter(s) that minimize the sample mean training
        error while maintaining stability over the testing regime.

        This method requires the :attr:`model` to be fully discrete; use
        :meth:`fit_regselect_continuous` for time-continuous models. The
        ``model.solver.regularizer`` is repeatedly adjusted, and the operator
        :attr:`posterior` is recalibrated, until a best regularization is
        selected. Training error is measured by comparing training data to the
        sample mean of ``num_posterior_draws`` model predictions. Stability is
        required for each of the individual model predictions.
        See :cite:`mcquarrie2021combustion,guo2022bayesopinf`.

        Parameters
        ----------
        candidates : list of regularization hyperparameters
            Regularization hyperparameters to check. If a single hyperparameter
            is given, use it as the start of an optimization-based search.
        states : list of s (n, k_i) ndarrays
            State snapshots in the original state space. Each array
            ``states[i]`` is data corresponding to a different trajectory;
            each column ``states[i][:, j]`` is one snapshot. This method
            assumes the snapshots are sequential, i.e., the model maps
            ``states[i][:, j]`` to ``states[i][:, j+1]``.
        states : list of s (r, k_i) ndarrays
            State snapshots in the reduced state space. This method assumes
            the snapshots are sequential, i.e., the model maps
            ``states[i][:, j]`` to ``states[i][:, j+1]``.
        inputs : list of s (m, k_i + num_test_iters) ndarrays
            Inputs corresponding to the training data, together with inputs
            for the testing regime. Only required if the :attr:`model` takes
            external inputs.
        regularizer_factory : callable or None
            Function mapping regularization hyperparameters to the full
            regularizer. Specifically, ``regularizer_factory(candidates[i])``
            will be assigned to ``model.solver.regularizer`` for each ``i``.
        gridsearch_only : bool
            If ``True``, stop after checking all regularization ``candidates``
            and do not follow up with optimization.
        num_test_iters : int
            Number of iterations after the training data in which to require
            model stability.
        stability_margin : float,
            Factor by which the reduced states may deviate from the range of
            the training data without being flagged as unstable.
        num_posterior_draws : int
            Number of draws from the operator :attr:`posterior` for stability
            checks and for estimating the sample mean of model predictions.
        test_cases : list of DiscreteRegTest objects
            Additional test cases for which the model is required to be stable.
            See :class:`opinf.utils.DiscreteRegTest`.
        verbose : bool
            If ``True``, print information during the regularization selection.

        Notes
        -----
        If there is only one trajectory of training data (s = 1), ``states``
        may be provided as an (n, k) ndarray. In this case, it is assumed that
        ``inputs`` (if provided) is a single (m, k + num_test_iters) ndarray.
        """
        return _BayesianROMMixin.fit_regselect_discrete(
            self,
            candidates=candidates,
            parameters=None,
            states=states,
            inputs=inputs,
            fit_transformer=fit_transformer,
            fit_basis=fit_basis,
            regularizer_factory=regularizer_factory,
            gridsearch_only=gridsearch_only,
            num_test_iters=num_test_iters,
            stability_margin=stability_margin,
            num_posterior_draws=num_posterior_draws,
            test_cases=test_cases,
            verbose=verbose,
        )

    def predict(self, state0, *args, **kwargs):
        """Draw from the operator posterior and evaluate the resulting model.

        Arguments are the same as the ``predict()`` method of :attr:`model`.

        Parameters
        ----------
        state0 : (n,) ndarray
            Initial state, expressed in the original state space.
        *args : list
            Other positional arguments to the ``predict()`` method of
            :attr:`model`.
        **kwargs : dict
            Keyword arguments to the ``predict()`` method of :attr:`model`.

        Returns
        -------
        states: (n, k) ndarray
            Solution to the drawn model, expressed in the original state space.
        """
        self.draw_operators()
        return ROM.predict(self, state0, *args, **kwargs)
