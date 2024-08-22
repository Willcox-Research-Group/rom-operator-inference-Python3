# operators/_affine.py
"""Classes for parametric OpInf operators where the parametric dependence is
expressed as an affine expansion.
"""

__all__ = [
    "AffineConstantOperator",
    "AffineLinearOperator",
    "AffineQuadraticOperator",
    "AffineCubicOperator",
    "AffineInputOperator",
    "AffineStateInputOperator",
]

import h5py
import numpy as np
import scipy.sparse as sparse

from .. import utils
from ._base import ParametricOpInfOperator, InputMixin
from ._nonparametric import (
    ConstantOperator,
    LinearOperator,
    QuadraticOperator,
    CubicOperator,
    InputOperator,
    StateInputOperator,
)


# Base class ==================================================================
class _AffineOperator(ParametricOpInfOperator):
    r"""Base class for parametric operators where the parameter dependence
    can be written as an affine expansion with known scalar coefficients
    which are a function of the parameter vector.

    This type of operator can be written as

    .. math::
       \Ophat_{\ell}(\qhat,\u;\bfmu) = \left(\sum_{a=0}^{A_{\ell}-1}
       \theta_{\ell}^{(0)}\!(\bfmu)\Ohat_{\ell}^{(a)}
       \right)\d_{\ell}(\qhat, \u)

    where each :math:`\theta_{\ell}^{(a)}:\RR^{p}\to\RR` is a scalar-valued
    function of the parameter vector, each
    :math:`\Ohat_{\ell}^{(a)}\in\RR^{r\times d}` is a constant matrix, and
    :math:`\d:\RR^{r}\times\RR^{m}\to\RR^{d}.`

    Parent class: :class:`opinf.operators.ParametricOpInfOperator`

    Parameters
    ----------
    coefficient_functions : (iterable of callables) or int
        Scalar-valued coefficient functions for each term of the affine
        expansion, i.e.,
        :math:`\theta_{\ell}^{(0)},\ldots,\theta_{\ell}^{(A_{\ell}-1)}.`
        If an integer :math:`p` is provided, set :math:`A_{\ell} = p` and
        define :math:`\theta_{\ell}^{(i)}\!(\bfmu) = \mu_i`.
    entries : list of ndarrays, or None
        Operator matrices for each term of the affine expansion, i.e.,
        :math:`\Ohat_{\ell}^{(0)},\ldots,\Ohat_{\ell}^{(A_{\ell}-1)}.`
        If not provided in the constructor, use :meth:`set_entries` later.
    fromblock : bool
        If ``True``, interpret ``entries`` as a horizontal concatenation
        of arrays; if ``False`` (default), interpret ``entries`` as a list
        of arrays.

    Warnings
    --------
    A common choice for the ``coefficient_functions`` is for the :math:`i`-th
    coefficient function to return the :math:`i`-th component of the parameter
    vector, i.e., :math:`\theta_{\ell}^{(i)}\!(\bfmu) = \mu_{i}`. The following
    implementation for this choice results in a subtle but serious error:

    .. code-block:: python

       coefficient_functions = [lambda mu: mu[i] for i in range(nterms)]

    Due to the late binding behavior of closures in Python, the ``lambda``
    functions do not capture the value of the variable ``i`` at each iteration.
    When any of the ``lambda`` functions are called, they use the value of
    ``i`` at the time of the call, which, after the loop, is always
    ``nterms - 1``. To avoid this issue, use ``coefficient_functions=p`` in the
    constructor, where ``p`` is the dimension of the parameter vector. For
    related scenarios, avoid this pitfall by writing `lambda` function with
    the index given explicitly, or by using a function factory.

    .. code-block:: python

       coefficient_functions = [
           lambda mu: mu[0],
           lambda mu: mu[1],
           lambda mu: mu[2],
           # ...
       ]

       # Alternatively, define a function factory.
       def coeff_function(i : int):
           return lambda mu: mu[i]
       coefficient_functions = [coeff_function(i) for i in range(nterms)]
    """

    # Initialization ----------------------------------------------------------
    def __init__(
        self,
        coefficient_functions,
        entries=None,
        fromblock: bool = False,
    ):
        """Set coefficient functions and (if given) operator matrices."""
        ParametricOpInfOperator.__init__(self)

        # Shortcut: theta[i](mu) = mu[i].
        if isinstance(coefficient_functions, int):
            self.parameter_dimension = coefficient_functions

            def componentgetter(i: int):
                """Make a function that returns the i-th value of its input."""
                return lambda mu: mu[i]

            coefficient_functions = [
                componentgetter(i) for i in range(self.parameter_dimension)
            ]

        # Ensure that the coefficient functions are callable.
        if any(not callable(theta) for theta in coefficient_functions):
            raise TypeError(
                "coefficient_functions must be collection of callables"
            )
        self.__thetas = tuple(coefficient_functions)

        if entries is not None:
            self.set_entries(entries, fromblock=fromblock)

    # Properties --------------------------------------------------------------
    @property
    def coefficient_functions(self) -> tuple:
        r"""Scalar-valued coefficient functions for each term of the affine
        expansion, i.e.,
        :math:`\theta_{\ell}^{(0)},\ldots,\theta_{\ell}^{(A_{\ell}-1)}.`
        """
        return self.__thetas

    @property
    def entries(self) -> list:
        r"""Operator matrices for each term of the affine expansion, i.e.,
        :math:`\Ohat_{\ell}^{(0)},\ldots,\Ohat_{\ell}^{(A_{\ell}-1)}.`
        """
        return ParametricOpInfOperator.entries.fget(self)

    @property
    def nterms(self):
        r"""Number of terms :math:`A_{\ell}` in the affine expansion."""
        return len(self.coefficient_functions)

    def set_entries(self, entries, fromblock: bool = False) -> None:
        r"""Set the operator matrices for each term of the affine expansion.

        Parameters
        ----------
        entries : list of s (r, d) ndarrays, or (r, sd) ndarray
            Operator matrices, either as a list of arrays
            (``fromblock=False``, default)
            or as a horizontal concatenatation of arrays (``fromblock=True``).
        fromblock : bool
            If ``True``, interpret ``entries`` as a horizontal concatenation
            of arrays; if ``False`` (default), interpret ``entries`` as a list
            of arrays.
        """
        # Extract / verify the entries.
        nterms = self.nterms
        if fromblock:
            if not isinstance(entries, np.ndarray) or (
                entries.ndim not in (1, 2)
            ):
                raise ValueError(
                    "entries must be a 1- or 2-dimensional ndarray "
                    "when fromblock=True"
                )
            entries = np.split(entries, nterms, axis=-1)
        if np.ndim(entries) > 1:
            self._check_shape_consistency(entries, "entries")
        if (n_arrays := len(entries)) != nterms:
            raise ValueError(
                f"{nterms} = len(coefficient_functions) "
                f"!= len(entries) = {n_arrays}"
            )

        ParametricOpInfOperator.set_entries(
            self,
            [self.OperatorClass(A).entries for A in entries],
        )

    # Evaluation --------------------------------------------------------------
    @utils.requires("entries")
    def evaluate(self, parameter):
        r"""Evaluate the operator at the given parameter value.

        Parameters
        ----------
        parameter : (p,) ndarray or float
            Parameter value :math:`\bfmu` at which to evalute the operator.

        Returns
        -------
        op : :mod:`opinf.operators` operator of type :attr:`OperatorClass`
            Nonparametric operator corresponding to the parameter value.
        """
        if self.parameter_dimension is None:
            self._set_parameter_dimension_from_values([parameter])
        self._check_parametervalue_dimension(parameter)
        entries = sum(
            [
                theta(parameter) * A
                for theta, A in zip(self.coefficient_functions, self.entries)
            ]
        )
        return self.OperatorClass(entries)

    # Dimensionality reduction ------------------------------------------------
    @utils.requires("entries")
    def galerkin(self, Vr, Wr=None):
        r"""Project this operator to a low-dimensional linear space.

        Consider an affine operator

        .. math::
            \Op_{\ell}(\q,\u;\bfmu)
            = \sum_{a=0}^{A_{\ell}-1}\theta_{\ell}^{(a)}\!(\bfmu)\,
            \Op_{\ell}^{(a)}\!(\q, \u)

        where

        * :math:`\q\in\RR^n` is the full-order state,
        * :math:`\u\in\RR^m` is the input,
        * :math:`\bfmu\in\RR^p` is the parameter vector, and
        * each :math:`\Op_{\ell}^{(a)}\!(\q,\u)` is a nonparametric operator.

        Given a *trial basis* :math:`\Vr\in\RR^{n\times r}` and a *test basis*
        :math:`\Wr\in\RR^{n\times r}`, the corresponding *intrusive projection*
        of :math:`\f` is the affine operator

        .. math::
           \fhat_{\ell}(\qhat,\u;\bfmu)
           = \sum_{a=0}^{A_{\ell}-1}\theta_{\ell}^{(a)}\!(\bfmu)\,
           (\Wr\trp\Vr)^{-1}\Wr\trp\Op_{\ell}^{(a)}\!(\V\qhat, \u)
           = \sum_{a=0}^{A_{\ell}-1}\theta_{\ell}^{(a)}\!(\bfmu)\,
           \Ophat_{\ell}^{(a)}\!(\qhat, \u),

        where :math:`\Ophat_{\ell}^{(a)}\!(\qhat, \u)
        = (\Wr\trp\Vr)^{-1}\Wr\trp\Op_{\ell}^{(a)}\!(\V\qhat, \u)`
        is the intrusive projection of :math:`\Op_{\ell}^{(a)}.`
        Here, :math:`\qhat\in\RR^r` is the reduced-order state, which enables
        the low-dimensional state approximation :math:`\q = \Vr\qhat.`
        If :math:`\Wr = \Vr`, the result is called a *Galerkin projection*.
        If :math:`\Wr \neq \Vr`, it is called a *Petrov-Galerkin projection*.

        Parameters
        ----------
        Vr : (n, r) ndarray
            Basis for the trial space.
        Wr : (n, r) ndarray or None
            Basis for the test space. If ``None``, defaults to ``Vr``.

        Returns
        -------
        op : operator
            New object of the same class as ``self``.
        """
        return self.__class__(
            coefficient_functions=self.coefficient_functions,
            entries=[
                self.OperatorClass(A).galerkin(Vr, Wr).entries
                for A in self.entries
            ],
            fromblock=False,
        )

    # Operator inference ------------------------------------------------------
    def operator_dimension(self, s: int, r: int, m: int) -> int:
        r"""Number of columns in the concatenated operator matrix.

        For affine operators, this is :math:`A_{\ell}\cdot d(r,m)`,
        where :math:`A_{\ell}` is the number of terms in the affine expansion
        and :math:`d(r,m)` is the dimension of the function
        :math:`\d(\qhat,\u)`.

        Parameters
        ----------
        s : int
            Number of training parameter values.
        r : int
            State dimension.
        m : int or None
            Input dimension.

        Returns
        -------
        d : int
            Number of columns in the concatenated operator matrix.
        """
        return self.nterms * self.OperatorClass.operator_dimension(r, m)

    def datablock(self, parameters, states, inputs=None) -> np.ndarray:
        r"""Return the data matrix block corresponding to the operator.

        For affine operators :math:`\Ophat_{\ell}(\qhat,\u;\bfmu)
        = \Ohat_{\ell}(\bfmu)\d_{\ell}(\qhat,\u)` with
        :math:`\Ohat_{\ell}(\bfmu)\in\RR^{r\times d}` and
        :math:`\d_{\ell}(\qhat,\u)\in\RR^{d}`, this is the block matrix

        .. math::
           \D_{\ell}\trp
           = \left[\begin{array}{ccc}
               \theta_{\ell}^{(0)}\!(\bfmu_{0})\,
               \d_{\ell}(\Qhat_{0},\U_{0})
               & \cdots &
               \theta_{\ell}^{(0)}\!(\bfmu_{s-1})\,
               \d_{\ell}(\Qhat_{s-1},\U_{s-1})
               \\ \vdots & & \vdots \\
               \theta_{\ell}^{(A_{\ell})}\!(\bfmu_{0})\,
               \d_{\ell}(\Qhat_{0},\U_{0})
               & \cdots &
               \theta_{\ell}^{(A_{\ell})}\!(\bfmu_{s-1})\,
               \d_{\ell}(\Qhat_{s-1},\U_{s-1})
           \end{array}\right]
           \in \RR^{A_{\ell}d \times \sum_{i=0}^{s-1}k_i}

        where :math:`\Qhat_{i} =
        [~\qhat_{i,0}~~\cdots~~\qhat_{i,k_i-1}] \in \RR^{r \times k_i}`
        and :math:`\U_{i} =
        [~\u_{i,0}~~\cdots~~\u_{i,k_i-1}] \in \RR^{m\times k_i}`
        are the state snapshots and inputs corresponding to training parameter
        value :math:`\bfmu_i\in\RR^{p}`, :math:`i = 0, \ldots, s-1`, where
        :math:`s` is the number of training parameter values. The notation
        :math:`\d_{\ell}(\Qhat_{i},\U_{i})` is shorthand for the matrix

        .. math::
           \d(\Qhat_{i},\U_{i})
           = \left[\begin{array}{ccc}
               \d_{\ell}(\qhat_{i,0},\u_{i,0})
               & \cdots &
               \d_{\ell}(\qhat_{i,k_i-1},\u_{i,k_i-1})
           \end{array}\right]
           \in \RR^{d \times k_i}.

        Parameters
        ----------
        parameters : (s, p) ndarray
            Traning parameter values :math:`\bfmu_{0},\ldots,\bfmu_{s-1}.`
        states : list of s (r, k) ndarrays
            State snapshots for each of the :math:`s` training parameter
            values, i.e., :math:`\Qhat_{0},\ldots,\Qhat_{s-1}.`
        inputs : list of s (m, k)-or-(k,) ndarrays or None
            Inputs corresponding to the state snapshots, i.e.,
            :math:`\U_{0},\ldots,\U_{s-1}.`
            If each input matrix is 1D, it is assumed that :math:`m = 1.`

        Returns
        -------
        block : (D, K) ndarray
            Data block for the affine operator. Here,
            :math:`D = A_{\ell}d(r,m)` and :math:`K = \sum_{i=0}^{s-1}k_i`
            is the total number of snapshots.
        """
        if not isinstance(self, InputMixin):
            inputs = [None] * len(parameters)
        blockcolumns = []
        for mu, Q, U in zip(parameters, states, inputs):
            Di = self.OperatorClass.datablock(Q, U)
            blockcolumns.append(
                np.vstack(
                    [theta(mu) * Di for theta in self.coefficient_functions]
                )
            )
        return np.hstack(blockcolumns)

    # Model persistence -------------------------------------------------------
    def copy(self):
        """Return a copy of the operator. Only the operator matrices are
        copied, not the coefficient functions.
        """
        As = None
        if self.entries is not None:
            As = [A.copy() for A in self.entries]
        op = self.__class__(
            coefficient_functions=self.coefficient_functions,
            entries=As,
            fromblock=False,
        )
        if self.parameter_dimension is not None:
            op.parameter_dimension = self.parameter_dimension
        return op

    def save(self, savefile: str, overwrite: bool = False) -> None:
        """Save the operator to an HDF5 file.

        Since the :attr:`coefficient_functions` are callables, they cannot be
        serialized, and are therefore an argument to :meth:`load()`.

        Parameters
        ----------
        savefile : str
            Path of the file to save the basis in.
        overwrite : bool
            If ``True``, overwrite the file if it already exists. If ``False``
            (default), raise a ``FileExistsError`` if the file already exists.
        """
        with utils.hdf5_savehandle(savefile, overwrite) as hf:
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["class"] = self.__class__.__name__
            if (p := self.parameter_dimension) is not None:
                meta.attrs["parameter_dimension"] = p
            if self.entries is not None:
                group = hf.create_group("entries")
                for i, Ai in enumerate(self.entries):
                    name = f"A{i:d}"
                    if sparse.issparse(Ai):
                        utils.save_sparray(group.create_group(name), Ai)
                    else:
                        group.create_dataset(name, data=Ai)

    @classmethod
    def load(cls, loadfile: str, coefficient_functions):
        """Load an affine parametric operator from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            Path to the file where the operator was stored via :meth:`save()`.
        coefficient_functions : iterable of callables
            Scalar-valued coefficient functions for each term of the affine
            expansion.
        Returns
        -------
        op : _AffineOperator
            Initialized operator object.
        """
        with utils.hdf5_loadhandle(loadfile) as hf:
            ClassName = hf["meta"].attrs["class"]
            if ClassName != cls.__name__:
                raise TypeError(
                    f"file '{loadfile}' contains '{ClassName}' "
                    f"object, use '{ClassName}.load()'"
                )

            entries = None
            if "entries" in hf:
                entries = []
                group = hf["entries"]
                for i in range(len(group)):
                    obj = group[f"A{i:d}"]
                    if isinstance(obj, h5py.Dataset):
                        entries.append(obj[:])
                    else:
                        entries.append(utils.load_sparray(obj))

            op = cls(
                coefficient_functions=coefficient_functions,
                entries=entries,
                fromblock=False,
            )

            if (key := "parameter_dimension") in hf["meta"].attrs:
                op.parameter_dimension = int(hf["meta"].attrs[key])
            return op


# Public affine operator classes ==============================================
class AffineConstantOperator(_AffineOperator):
    r"""Affine-parametric constant operator
    :math:`\Ophat_{\ell}(\qhat,\u;\bfmu)
    = \chat_{\ell}(\bfmu)
    = \sum_{a=0}^{A_{\ell}-1}\theta_\ell^{(a)}\!(\bfmu)\,\chat_{\ell}^{(a)}.`

    Here, each :math:`\theta_\ell^{(a)}:\RR^{p}\to\RR` is a scalar-valued
    function of the parameter vector
    and each :math:`\chat_{\ell}^{(a)} \in \RR^r` is a constant vector,
    see :class:`opinf.operators.ConstantOperator`.

    Parameters
    ----------
    coefficient_functions : (iterable of callables) or int
        Scalar-valued coefficient functions for each term of the affine
        expansion, i.e.,
        :math:`\theta_{\ell}^{(0)},\ldots,\theta_{\ell}^{(A_{\ell}-1)}.`
        If an integer :math:`p` is provided, set :math:`A_{\ell} = p` and
        define :math:`\theta_{\ell}^{(i)}\!(\bfmu) = \mu_i`.
    entries : list of ndarrays, or None
        Operator matrices for each term of the affine expansion, i.e.,
        :math:`\Ohat_{\ell}^{(0)},\ldots,\Ohat_{\ell}^{(A_{\ell}-1)}.`
        If not provided in the constructor, use :meth:`set_entries` later.
    fromblock : bool
        If ``True``, interpret ``entries`` as a horizontal concatenation
        of arrays; if ``False`` (default), interpret ``entries`` as a list
        of arrays.

    Warnings
    --------
    A common choice for the ``coefficient_functions`` is for the :math:`i`-th
    coefficient function to return the :math:`i`-th component of the parameter
    vector, i.e., :math:`\theta_{\ell}^{(i)}\!(\bfmu) = \mu_{i}`. The following
    implementation for this choice results in a subtle but serious error:

    .. code-block:: python

       coefficient_functions = [lambda mu: mu[i] for i in range(nterms)]

    Due to the late binding behavior of closures in Python, the ``lambda``
    functions do not capture the value of the variable ``i`` at each iteration.
    When any of the ``lambda`` functions are called, they use the value of
    ``i`` at the time of the call, which, after the loop, is always
    ``nterms - 1``. To avoid this issue, use ``coefficient_functions=p`` in the
    constructor, where ``p`` is the dimension of the parameter vector. For
    related scenarios, avoid this pitfall by writing `lambda` function with
    the index given explicitly, or by using a function factory.

    .. code-block:: python

       coefficient_functions = [
           lambda mu: mu[0],
           lambda mu: mu[1],
           lambda mu: mu[2],
           # ...
       ]

       # Alternatively, define a function factory.
       def coeff_function(i : int):
           return lambda mu: mu[i]
       coefficient_functions = [coeff_function(i) for i in range(nterms)]
    """

    _OperatorClass = ConstantOperator


class AffineLinearOperator(_AffineOperator):
    r"""Affine-parametric linear operator
    :math:`\Ophat_{\ell}(\qhat,\u;\bfmu)
    = \Ahat_{\ell}(\bfmu)\qhat = \left(
    \sum_{a=0}^{A_{\ell}-1}\theta_{\ell}^{(a)}\!(\bfmu)\,\Ahat_{\ell}^{(a)}
    \right)\qhat.`

    Here, each :math:`\theta_\ell^{(a)}:\RR^{p}\to\RR` is a scalar-valued
    function of the parameter vector
    and each :math:`\Ahat_{\ell}^{(a)} \in \RR^{r\times r}` is a constant
    matrix, see :class:`opinf.operators.LinearOperator`.

    Parameters
    ----------
    coefficient_functions : (iterable of callables) or int
        Scalar-valued coefficient functions for each term of the affine
        expansion, i.e.,
        :math:`\theta_{\ell}^{(0)},\ldots,\theta_{\ell}^{(A_{\ell}-1)}.`
        If an integer :math:`p` is provided, set :math:`A_{\ell} = p` and
        define :math:`\theta_{\ell}^{(i)}\!(\bfmu) = \mu_i`.
    entries : list of ndarrays, or None
        Operator matrices for each term of the affine expansion, i.e.,
        :math:`\Ahat_{\ell}^{(0)},\ldots,\Ahat_{\ell}^{(A_{\ell}-1)}.`
        If not provided in the constructor, use :meth:`set_entries` later.
    fromblock : bool
        If ``True``, interpret ``entries`` as a horizontal concatenation
        of arrays; if ``False`` (default), interpret ``entries`` as a list
        of arrays.

    Warnings
    --------
    A common choice for the ``coefficient_functions`` is for the :math:`i`-th
    coefficient function to return the :math:`i`-th component of the parameter
    vector, i.e., :math:`\theta_{\ell}^{(i)}\!(\bfmu) = \mu_{i}`. The following
    implementation for this choice results in a subtle but serious error:

    .. code-block:: python

       coefficient_functions = [lambda mu: mu[i] for i in range(nterms)]

    Due to the late binding behavior of closures in Python, the ``lambda``
    functions do not capture the value of the variable ``i`` at each iteration.
    When any of the ``lambda`` functions are called, they use the value of
    ``i`` at the time of the call, which, after the loop, is always
    ``nterms - 1``. To avoid this issue, use ``coefficient_functions=p`` in the
    constructor, where ``p`` is the dimension of the parameter vector. For
    related scenarios, avoid this pitfall by writing `lambda` function with
    the index given explicitly, or by using a function factory.

    .. code-block:: python

       coefficient_functions = [
           lambda mu: mu[0],
           lambda mu: mu[1],
           lambda mu: mu[2],
           # ...
       ]

       # Alternatively, define a function factory.
       def coeff_function(i : int):
           return lambda mu: mu[i]
       coefficient_functions = [coeff_function(i) for i in range(nterms)]
    """

    _OperatorClass = LinearOperator


class AffineQuadraticOperator(_AffineOperator):
    r"""Affine-parametric quadratic operator
    :math:`\Ophat_{\ell}(\qhat,\u;\bfmu)
    = \Hhat_{\ell}(\bfmu)[\qhat\otimes\qhat] = \left(
    \sum_{a=0}^{A_{\ell}-1}\theta_{\ell}^{(a)}\!(\bfmu)\,\Hhat_{\ell}^{(a)}
    \right)[\qhat\otimes\qhat].`

    Here, each :math:`\theta_\ell^{(a)}:\RR^{p}\to\RR` is a scalar-valued
    function of the parameter vector
    and each :math:`\Hhat_{\ell}^{(a)} \in \RR^{r\times r^2}` is a constant
    matrix, see :class:`opinf.operators.QuadraticOperator`.

    Parameters
    ----------
    coefficient_functions : (iterable of callables) or int
        Scalar-valued coefficient functions for each term of the affine
        expansion, i.e.,
        :math:`\theta_{\ell}^{(0)},\ldots,\theta_{\ell}^{(A_{\ell}-1)}.`
        If an integer :math:`p` is provided, set :math:`A_{\ell} = p` and
        define :math:`\theta_{\ell}^{(i)}\!(\bfmu) = \mu_i`.
    entries : list of ndarrays, or None
        Operator matrices for each term of the affine expansion, i.e.,
        :math:`\Hhat_{\ell}^{(0)},\ldots,\Hhat_{\ell}^{(A_{\ell}-1)}.`
        If not provided in the constructor, use :meth:`set_entries` later.
    fromblock : bool
        If ``True``, interpret ``entries`` as a horizontal concatenation
        of arrays; if ``False`` (default), interpret ``entries`` as a list
        of arrays.

    Warnings
    --------
    A common choice for the ``coefficient_functions`` is for the :math:`i`-th
    coefficient function to return the :math:`i`-th component of the parameter
    vector, i.e., :math:`\theta_{\ell}^{(i)}\!(\bfmu) = \mu_{i}`. The following
    implementation for this choice results in a subtle but serious error:

    .. code-block:: python

       coefficient_functions = [lambda mu: mu[i] for i in range(nterms)]

    Due to the late binding behavior of closures in Python, the ``lambda``
    functions do not capture the value of the variable ``i`` at each iteration.
    When any of the ``lambda`` functions are called, they use the value of
    ``i`` at the time of the call, which, after the loop, is always
    ``nterms - 1``. To avoid this issue, use ``coefficient_functions=p`` in the
    constructor, where ``p`` is the dimension of the parameter vector. For
    related scenarios, avoid this pitfall by writing `lambda` function with
    the index given explicitly, or by using a function factory.

    .. code-block:: python

       coefficient_functions = [
           lambda mu: mu[0],
           lambda mu: mu[1],
           lambda mu: mu[2],
           # ...
       ]

       # Alternatively, define a function factory.
       def coeff_function(i : int):
           return lambda mu: mu[i]
       coefficient_functions = [coeff_function(i) for i in range(nterms)]
    """

    _OperatorClass = QuadraticOperator


class AffineCubicOperator(_AffineOperator):
    r"""Affine-parametric cubic operator
    :math:`\Ophat_{\ell}(\qhat,\u;\bfmu)
    = \Ghat_{\ell}(\bfmu)[\qhat\otimes\qhat\otimes\qhat] = \left(
    \sum_{a=0}^{A_{\ell}-1}\theta_{\ell}^{(a)}\!(\bfmu)\,\Ghat_{\ell}^{(a)}
    \right)[\qhat\otimes\qhat\otimes\qhat].`

    Here, each :math:`\theta_\ell^{(a)}:\RR^{p}\to\RR` is a scalar-valued
    function of the parameter vector
    and each :math:`\Ghat_{\ell}^{(a)} \in \RR^{r\times r^3}` is a constant
    matrix, see :class:`opinf.operators.CubicOperator`.

    Parameters
    ----------
    coefficient_functions : (iterable of callables) or int
        Scalar-valued coefficient functions for each term of the affine
        expansion, i.e.,
        :math:`\theta_{\ell}^{(0)},\ldots,\theta_{\ell}^{(A_{\ell}-1)}.`
        If an integer :math:`p` is provided, set :math:`A_{\ell} = p` and
        define :math:`\theta_{\ell}^{(i)}\!(\bfmu) = \mu_i`.
    entries : list of ndarrays, or None
        Operator matrices for each term of the affine expansion, i.e.,
        :math:`\Ghat_{\ell}^{(0)},\ldots,\Ghat_{\ell}^{(A_{\ell}-1)}.`
        If not provided in the constructor, use :meth:`set_entries` later.
    fromblock : bool
        If ``True``, interpret ``entries`` as a horizontal concatenation
        of arrays; if ``False`` (default), interpret ``entries`` as a list
        of arrays.

    Warnings
    --------
    A common choice for the ``coefficient_functions`` is for the :math:`i`-th
    coefficient function to return the :math:`i`-th component of the parameter
    vector, i.e., :math:`\theta_{\ell}^{(i)}\!(\bfmu) = \mu_{i}`. The following
    implementation for this choice results in a subtle but serious error:

    .. code-block:: python

       coefficient_functions = [lambda mu: mu[i] for i in range(nterms)]

    Due to the late binding behavior of closures in Python, the ``lambda``
    functions do not capture the value of the variable ``i`` at each iteration.
    When any of the ``lambda`` functions are called, they use the value of
    ``i`` at the time of the call, which, after the loop, is always
    ``nterms - 1``. To avoid this issue, use ``coefficient_functions=p`` in the
    constructor, where ``p`` is the dimension of the parameter vector. For
    related scenarios, avoid this pitfall by writing `lambda` function with
    the index given explicitly, or by using a function factory.

    .. code-block:: python

       coefficient_functions = [
           lambda mu: mu[0],
           lambda mu: mu[1],
           lambda mu: mu[2],
           # ...
       ]

       # Alternatively, define a function factory.
       def coeff_function(i : int):
           return lambda mu: mu[i]
       coefficient_functions = [coeff_function(i) for i in range(nterms)]
    """

    _OperatorClass = CubicOperator


class AffineInputOperator(_AffineOperator, InputMixin):
    r"""Affine-parametric input operator
    :math:`\Ophat_{\ell}(\qhat,\u;\bfmu)
    = \Bhat_{\ell}(\bfmu)\u = \left(
    \sum_{a=0}^{A_{\ell}-1}\theta_{\ell}^{(a)}\!(\bfmu)\,\Bhat_{\ell}^{(a)}
    \right)\u.`

    Here, each :math:`\theta_\ell^{(a)}:\RR^{p}\to\RR` is a scalar-valued
    function of the parameter vector
    and each :math:`\Bhat_{\ell}^{(a)} \in \RR^{r\times m}` is a constant
    matrix, see :class:`opinf.operators.InputOperator`.

    Parameters
    ----------
    coefficient_functions : (iterable of callables) or int
        Scalar-valued coefficient functions for each term of the affine
        expansion, i.e.,
        :math:`\theta_{\ell}^{(0)},\ldots,\theta_{\ell}^{(A_{\ell}-1)}.`
        If an integer :math:`p` is provided, set :math:`A_{\ell} = p` and
        define :math:`\theta_{\ell}^{(i)}\!(\bfmu) = \mu_i`.
    entries : list of ndarrays, or None
        Operator matrices for each term of the affine expansion, i.e.,
        :math:`\Bhat_{\ell}^{(0)},\ldots,\Bhat_{\ell}^{(A_{\ell}-1)}.`
        If not provided in the constructor, use :meth:`set_entries` later.
    fromblock : bool
        If ``True``, interpret ``entries`` as a horizontal concatenation
        of arrays; if ``False`` (default), interpret ``entries`` as a list
        of arrays.

    Warnings
    --------
    A common choice for the ``coefficient_functions`` is for the :math:`i`-th
    coefficient function to return the :math:`i`-th component of the parameter
    vector, i.e., :math:`\theta_{\ell}^{(i)}\!(\bfmu) = \mu_{i}`. The following
    implementation for this choice results in a subtle but serious error:

    .. code-block:: python

       coefficient_functions = [lambda mu: mu[i] for i in range(nterms)]

    Due to the late binding behavior of closures in Python, the ``lambda``
    functions do not capture the value of the variable ``i`` at each iteration.
    When any of the ``lambda`` functions are called, they use the value of
    ``i`` at the time of the call, which, after the loop, is always
    ``nterms - 1``. To avoid this issue, use ``coefficient_functions=p`` in the
    constructor, where ``p`` is the dimension of the parameter vector. For
    related scenarios, avoid this pitfall by writing `lambda` function with
    the index given explicitly, or by using a function factory.

    .. code-block:: python

       coefficient_functions = [
           lambda mu: mu[0],
           lambda mu: mu[1],
           lambda mu: mu[2],
           # ...
       ]

       # Alternatively, define a function factory.
       def coeff_function(i : int):
           return lambda mu: mu[i]
       coefficient_functions = [coeff_function(i) for i in range(nterms)]
    """

    _OperatorClass = InputOperator

    @property
    def input_dimension(self):
        r"""Dimension of the input :math:`\u` that the operator acts on."""
        return None if self.entries is None else self.shape[1]


class AffineStateInputOperator(_AffineOperator, InputMixin):
    r"""Affine-parametric state-input operator
    :math:`\Ophat_{\ell}(\qhat,\u;\bfmu)
    = \Nhat_{\ell}(\bfmu)\qhat = \left(
    \sum_{a=0}^{A_{\ell}-1}\theta_{\ell}^{(a)}\!(\bfmu)\,\Nhat_{\ell}^{(a)}
    \right)[\u\otimes\qhat].`

    Here, each :math:`\theta_\ell^{(a)}:\RR^{p}\to\RR` is a scalar-valued
    function of the parameter vector
    and each :math:`\Nhat_{\ell}^{(a)} \in \RR^{r\times rm}` is a constant
    matrix, see :class:`opinf.operators.StateInputOperator`.

    Parameters
    ----------
    coefficient_functions : (iterable of callables) or int
        Scalar-valued coefficient functions for each term of the affine
        expansion, i.e.,
        :math:`\theta_{\ell}^{(0)},\ldots,\theta_{\ell}^{(A_{\ell}-1)}.`
        If an integer :math:`p` is provided, set :math:`A_{\ell} = p` and
        define :math:`\theta_{\ell}^{(i)}\!(\bfmu) = \mu_i`.
    entries : list of ndarrays, or None
        Operator matrices for each term of the affine expansion, i.e.,
        :math:`\Nhat_{\ell}^{(0)},\ldots,\Nhat_{\ell}^{(A_{\ell}-1)}.`
        If not provided in the constructor, use :meth:`set_entries` later.
    fromblock : bool
        If ``True``, interpret ``entries`` as a horizontal concatenation
        of arrays; if ``False`` (default), interpret ``entries`` as a list
        of arrays.

    Warnings
    --------
    A common choice for the ``coefficient_functions`` is for the :math:`i`-th
    coefficient function to return the :math:`i`-th component of the parameter
    vector, i.e., :math:`\theta_{\ell}^{(i)}\!(\bfmu) = \mu_{i}`. The following
    implementation for this choice results in a subtle but serious error:

    .. code-block:: python

       coefficient_functions = [lambda mu: mu[i] for i in range(nterms)]

    Due to the late binding behavior of closures in Python, the ``lambda``
    functions do not capture the value of the variable ``i`` at each iteration.
    When any of the ``lambda`` functions are called, they use the value of
    ``i`` at the time of the call, which, after the loop, is always
    ``nterms - 1``. To avoid this issue, use ``coefficient_functions=p`` in the
    constructor, where ``p`` is the dimension of the parameter vector. For
    related scenarios, avoid this pitfall by writing `lambda` function with
    the index given explicitly, or by using a function factory.

    .. code-block:: python

       coefficient_functions = [
           lambda mu: mu[0],
           lambda mu: mu[1],
           lambda mu: mu[2],
           # ...
       ]

       # Alternatively, define a function factory.
       def coeff_function(i : int):
           return lambda mu: mu[i]
       coefficient_functions = [coeff_function(i) for i in range(nterms)]
    """

    _OperatorClass = StateInputOperator

    @property
    def input_dimension(self):
        r"""Dimension of the input :math:`\u` that the operator acts on."""
        if self.entries is None:
            return None
        r, rm = self.shape
        return rm // r


# Utilities ===================================================================
def is_affine(obj) -> bool:
    """Return ``True`` if ``obj`` is a interpolated operator object."""
    return isinstance(obj, _AffineOperator)


def nonparametric_to_affine(OpClass: type) -> type:
    """Get the affine operator class corresponding to a nonparametric
    operator class.

    """
    for AffineClassName in __all__:
        AffineClass = eval(AffineClassName)
        if not isinstance(AffineClass, type) or not issubclass(
            AffineClass, _AffineOperator
        ):  # pragma: no cover
            continue
        if AffineClass._OperatorClass is OpClass:
            return AffineClass
    raise TypeError(
        f"_AffineOperator for class '{OpClass.__name__}' not found"
    )
