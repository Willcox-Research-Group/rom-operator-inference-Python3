# pre/_shiftscale.py
"""Preprocessing transformations based on elementary shifts and scalings."""

__all__ = [
    "shift",
    "scale",
    "SnapshotTransformer",
    "SnapshotTransformerMulti",
]

import warnings
import numpy as np

from .. import errors, utils
from ._base import TransformerTemplate, _MultivarMixin


# Functional paradigm =========================================================
def shift(states: np.ndarray, shift_by: np.ndarray = None):
    """Shift the columns of a snapshot matrix by a vector.

    Parameters
    ----------
    states : (n, k) ndarray
        Matrix of k snapshots. Each column is a single snapshot.
    shift_by : (n,) ndarray
        Vector that is the same size as a single snapshot. If ``None``
        (default), set to the mean of the columns of ``states``.

    Returns
    -------
    states_shifted : (n, k) ndarray
        Shifted state matrix, i.e.,
        ``states_shifted[:, j] = states[:, j] - shift_by``.
    shift_by : (n,) ndarray
        Shift factor, returned only if ``shift_by=None``.
        Since this is a one-dimensional array, it must be reshaped to be
        applied to a matrix, for example,
        ``states_shifted = states - shift_by.reshape(-1, 1)``.

    Examples
    --------
    >>> import opinf
    # Shift Q by its mean, then shift Y by the same mean.
    >>> Q_shifted, qbar = opinf.pre.shift(Q)
    >>> Y_shifted = opinf.pre.shift(Y, qbar)
    # Shift Q by its mean, then undo the transformation by an inverse shift.
    >>> Q_shifted, qbar = opinf.pre.shift(Q)
    >>> Q_again = opinf.pre.shift(Q_shifted, -qbar)
    """
    # Check dimensions.
    if states.ndim != 2:
        raise ValueError("'states' must be two-dimensional")

    # If not shift_by factor is provided, compute the mean column.
    learning = shift_by is None
    if learning:
        shift_by = np.mean(states, axis=1)
    if shift_by.ndim != 1:
        if shift_by.ndim == 2 and shift_by.shape[1] == 1:
            shift_by = shift_by[:, 0]
        else:
            raise ValueError("'shift_by' must be one-dimensional")

    # Shift the columns by the mean.
    states_shifted = states - shift_by.reshape((-1, 1))

    return (states_shifted, shift_by) if learning else states_shifted


def scale(states: np.ndarray, scale_to: tuple, scale_from: tuple = None):
    r"""Scale the entries of a snapshot matrix to a specified interval
    :math:`[a, b]`.

    This scaling follows ``sklearn.preprocessing.MinMaxScaler``.
    If the entries of the snapshot matrix are contained in the interval
    :math:`[\bar{a}, \bar{b}]`, then the transformation is given by

    .. math::
       q' = \frac{q - \bar{a}}{\bar{b} - \bar{a}}(b - a) + a,

    where :math:`q` is the original variable and :math:`q'` is the transformed
    variable.

    Parameters
    ----------
    states : (n, k) ndarray
        Matrix of k snapshots to be scaled. Each column is a single snapshot.
    scale_to : (float, float)
        Desired minimum and maximum of the scaled data, i.e., :math:`[a, b]`.
    scale_from : (float, float)
        Minimum and maximum of the snapshot data. If None, learn the scaling:
        scale_from[0] = min(states); scale_from[1] = max(states).

    Returns
    -------
    states_scaled : (n, k) ndarray
        Scaled snapshot matrix.
    scaled_to : (float, float)
        Bounds that the snapshot matrix was scaled to, i.e.,
        ``scaled_to[0] = min(states_scaled)``;
        ``scaled_to[1] = max(states_scaled)``.
        Only returned if ``scale_from = None``.
    scaled_from : (float, float)
        Minimum and maximum of the snapshot data, i.e., the bounds that
        the data was scaled from. Only returned if ``scale_from = None``.

    Examples
    --------
    >>> import opinf
    # Scale Q to [-1, 1] and then scale Y with the same transformation.
    >>> Qscaled, scaled_to, scaled_from = opinf.pre.scale(Q, (-1, 1))
    >>> Yscaled = opinf.pre.scale(Y, scaled_to, scaled_from)
    # Scale Q to [0, 1], then undo the transformation by an inverse scaling.
    >>> Qscaled, scaled_to, scaled_from = opinf.pre.scale(Q, (0, 1))
    >>> Q_again = opinf.pre.scale(Qscaled, scaled_from, scaled_to)
    """
    # If no scale_from bounds are provided, learn them.
    learning = scale_from is None
    if learning:
        scale_from = np.min(states), np.max(states)

    # Check scales.
    if len(scale_to) != 2:
        raise ValueError("scale_to must have exactly 2 elements")
    if len(scale_from) != 2:
        raise ValueError("scale_from must have exactly 2 elements")

    # Do the scaling.
    mini, maxi = scale_to
    xmin, xmax = scale_from
    scl = (maxi - mini) / (xmax - xmin)
    states_scaled = states * scl + (mini - xmin * scl)

    return (states_scaled, scale_to, scale_from) if learning else states_scaled


# Object-oriented paradigm ====================================================
class SnapshotTransformer(TransformerTemplate):
    r"""Process snapshots by centering and/or scaling (in that order).

    Transformations with this class are notated below as

    .. math::
       \Q \mapsto \Q'
       ~\text{(centered)}~
       \mapsto \Q''
       ~\text{(centered/scaled)},

    where :math:`\Q\in\RR^{n \times k}` is the snapshot matrix to be
    transformed and :math:`\Q''\in\RR^{n \times k}` is the transformed snapshot
    matrix.

    All transformations with this class are `affine` and hence can be written
    componentwise as :math:`\Q_{i,j}'' = \alpha_{i,j} \Q_{i,j} + \beta_{i,j}`
    for some choice of :math:`\alpha_{i,j},\beta_{i,j}\in\RR`.

    Parameters
    ----------
    centering : bool
        If ``True``, shift the snapshots by the mean training snapshot, i.e.,

        .. math:: \Q'_{:,j} = \Q_{:,j} - \frac{1}{k}\sum_{j=0}^{k-1}\Q_{:,j}.

        Otherwise, :math:`\Q' = \Q` (default).

    scaling : str or None
        If given, scale (non-dimensionalize) the centered snapshot entries.
        Otherwise, :math:`\Q'' = \Q'` (default).

        **Options:**

        * ``'standard'``: standardize to zero mean and unit standard deviation.

          .. math:: \Q'' = \Q' - \frac{\mean(\Q')}{\std(\Q')}.
          Guarantees :math:`\mean(\Q'') = 0` and :math:`\std(\Q'') = 1`.

          If ``byrow=True``, then :math:`\mean_{j}(\Q_{i,j}'') = 0` and
          :math:`\std_j(\Q_{i,j}'') = 1` for each row index :math:`i`.

        * ``'minmax'``: minmax scaling to :math:`[0, 1]`.

          .. math:: \Q'' = \frac{\Q' - \min(\Q')}{\max(\Q') - \min(\Q')}.
          Guarantees :math:`\min(\Q'') = 0` and :math:`\max(\Q'') = 1`.

          If ``byrow=True``, then :math:`\min_{j}(\Q_{i,j}'') = 0` and
          :math:`\max_{j}(\Q_{i,j}'') = 1` for each row index :math:`i`.

        * ``'minmaxsym'``: minmax scaling to :math:`[-1, 1]`.

          .. math:: \Q'' = 2\frac{\Q' - \min(\Q')}{\max(\Q') - \min(\Q')} - 1.
          Guarantees :math:`\min(\Q'') = -1` and :math:`\max(\Q'') = 1`.

          If ``byrow=True``, then :math:`\min_{j}(\Q_{i,j}'') = -1` and
          :math:`\max_{j}(\Q_{i,j}'') = 1` for each row index :math:`i`.

        * ``'maxabs'``: maximum absolute scaling to :math:`[-1, 1]`
          `without` scalar mean shift.

          .. math:: \Q'' = \frac{1}{\max(\text{abs}(\Q'))}\Q'.
          Guarantees
          :math:`\mean(\Q'') = \frac{\mean(\Q')}{\max(\text{abs}(\Q'))}`
          and :math:`\max(\text{abs}(\Q'')) = 1`.

          If ``byrow=True``, then
          :math:`\mean_{j}(\Q_{i,j}'')
          = \frac{\mean_j(\Q_{i,j}')}{\max_j(\text{abs}(\Q_{i,j}'))}`
          and :math:`\max_{j}(\text{abs}(\Q_{i,j}'')) = 1` for each
          row index :math:`i`.

        * ``'maxabssym'``: maximum absolute scaling to :math:`[-1, 1]`
          `with` scalar mean shift.

          .. math::
             \Q''
             = \frac{\Q' - \mean(\Q')}{\max(\text{abs}(\Q' - \mean(\Q')))}.
          Guarantees :math:`\mean(\Q'') = 0` and
          :math:`\max(\text{abs}(\Q'')) = 1`.

          If ``byrow=True``, then :math:`\mean_j(\Q_{i,j}'') = 0` and
          :math:`\max_j(\text{abs}(\Q_{i,j}'')) = 1` for each row index
          :math:`i`.

    byrow : bool
        If ``True``, scale each row of the snapshot matrix separately when a
        scaling is specified. Otherwise, scale the entire matrix at once.

    verbose : bool
        If ``True``, print information upon learning a transformation.
    """
    _VALID_SCALINGS = frozenset(
        (
            "standard",
            "minmax",
            "minmaxsym",
            "maxabs",
            "maxabssym",
        )
    )

    _table_header = (
        "    |     min    |    mean    |     max    |    std\n"
        "----|------------|------------|------------|------------"
    )

    def __init__(
        self,
        centering: bool = False,
        scaling: str = None,
        byrow: bool = False,
        verbose: bool = False,
    ):
        """Set transformation hyperparameters."""
        self.__centering = bool(centering)
        self.__scaling = None
        self.__byrow = bool(byrow)
        self.__verbose = bool(verbose)
        self.__n = None

        self._clear()
        self.scaling = scaling

    def _clear_scaling(self):
        """Reset the numerical attributes that define the scaling."""
        self.__alpha = None
        self.__beta = None

    def _clear(self):
        """Reset the numerical attributes that define the transformation."""
        self.__qbar = None
        self._clear_scaling()

    # Properties --------------------------------------------------------------
    @property
    def centering(self) -> bool:
        """If ``True``, center the snapshots by the mean training snapshot."""
        return self.__centering

    @centering.setter
    def centering(self, ctr):
        """Set the centering directive, resetting the transformation."""
        if (yesno := bool(ctr)) is not self.__centering:
            self._clear()
            self.__centering = yesno

    @property
    def scaling(self) -> str:
        """Type of scaling (non-dimensionalization)."""
        return self.__scaling

    @scaling.setter
    def scaling(self, scl):
        """Set the scaling strategy, resetting the transformation."""
        if scl is None:
            self._clear_scaling()
            self.__scaling = scl
            return
        if not isinstance(scl, str):
            raise TypeError("'scaling' must be None or of type 'str'")
        if scl not in self._VALID_SCALINGS:
            opts = ", ".join([f"'{v}'" for v in self._VALID_SCALINGS])
            raise ValueError(
                f"invalid scaling '{scl}'; valid options are {opts}"
            )
        if scl != self.__scaling:
            self._clear_scaling()
            self.__scaling = scl

    @property
    def byrow(self) -> bool:
        """If ``True``, scale each row of the snapshot matrix separately."""
        return self.__byrow

    @byrow.setter
    def byrow(self, by):
        """Set the row-wise scaling directive, resetting the transformation."""
        if (yesno := bool(by)) is not self.byrow:
            if yesno and self.scaling is None:
                warnings.warn(
                    "scaling=None, byrow=True will have no effect",
                    errors.UsageWarning,
                )
            self._clear_scaling()
            self.__byrow = yesno

    @property
    def verbose(self) -> bool:
        """If ``True``, print information upon learning a transformation."""
        return self.__verbose

    @verbose.setter
    def verbose(self, vbs):
        """Set the verbosity."""
        self.__verbose = bool(vbs)

    @property
    def state_dimension(self):
        r"""Dimension :math:`n` of the state snapshots."""
        return self.__n

    @state_dimension.setter
    def state_dimension(self, n):
        """Set the state dimension."""
        if (dim := int(n)) != self.__n:
            self._clear()
        self.__n = dim

    @property
    def mean_(self):
        """Mean training snapshot. ``None`` unless ``centering = True``."""
        return self.__qbar

    @mean_.setter
    @utils.requires("state_dimension")
    def mean_(self, mean):
        """Set the mean vector."""
        if not self.centering:
            raise AttributeError("cannot set mean_ (centering=False)")
        if np.shape(mean) != (self.state_dimension,):
            raise ValueError(
                f"expected mean_ to be ({self.state_dimension:d},) ndarray"
            )
        self.__qbar = mean

    @property
    def scale_(self):
        r"""Multiplicative factor of the scaling, the :math:`\alpha` of
        :math:`q'' = \alpha q' + \beta`.
        """
        return self.__alpha

    @scale_.setter
    @utils.requires("state_dimension")
    def scale_(self, alpha):
        """Set the multiplicative factor of the scaling."""
        if self.scaling is None:
            raise AttributeError("cannot set scale_ (scaling=None)")
        if self.byrow and np.shape(alpha) != (self.state_dimension,):
            raise ValueError(
                f"expected scale_ to be ({self.state_dimension:d},) ndarray"
            )
        self.__alpha = alpha

    @property
    def shift_(self):
        r"""Additive factor of the scaling, the :math:`\beta` of
        :math:`q'' = \alpha q' + \beta`.
        """
        return self.__beta

    @shift_.setter
    @utils.requires("state_dimension")
    def shift_(self, beta):
        """Set the multiplicative factor of the scaling."""
        if self.scaling is None:
            raise AttributeError("cannot set shift_ (scaling=None)")
        if self.byrow and np.shape(beta) != (self.state_dimension,):
            raise ValueError(
                f"expected shift_ to be ({self.state_dimension:d},) ndarray"
            )
        self.__beta = beta

    def __eq__(self, other):
        """Test two SnapshotTransformers for equality."""
        if not isinstance(other, self.__class__):
            # print("WHY 1")
            return False
        for attr in ("centering", "scaling", "byrow"):
            if getattr(self, attr) != getattr(other, attr):
                # print("WHY 2")
                return False
        if self.state_dimension != other.state_dimension:
            # print("WHY 3")
            return False
        if self.centering and self.mean_ is not None:
            if other.mean_ is None:
                # print("WHY 4")
                return False
            if not np.all(self.mean_ == other.mean_):
                # print("WHY 5")
                return False
        if self.scaling and self.scale_ is not None:
            for attr in ("scale_", "shift_"):
                if (oat := getattr(other, attr)) is None:
                    # print("WHY 6")
                    return False
                if not np.all(getattr(self, attr) == oat):
                    # print("WHY 7")
                    return False
        # print("WHY 8")
        return True

    # Printing ----------------------------------------------------------------
    @staticmethod
    def _statistics_report(Q):
        """Return a string of basis statistics about a data set."""
        return " | ".join(
            [f"{f(Q):>10.3e}" for f in (np.min, np.mean, np.max, np.std)]
        )

    def __str__(self):
        """String representation: scaling type + centering bool."""
        out = ["Snapshot transformer"]
        trained = self._is_trained()
        if trained:
            out.append(f"(state dimension n = {self.state_dimension:d})")
        if self.centering:
            out.append("with mean-snapshot centering")
            if self.scaling:
                out.append(f"and '{self.scaling}' scaling")
        elif self.scaling:
            out.append(f"with '{self.scaling}' scaling")
        if not trained:
            out.append("(call fit() or fit_transform() to train)")
        return " ".join(out)

    def __repr__(self):
        """Unique ID + string representation."""
        uniqueID = f"<{self.__class__.__name__} object at {hex(id(self))}>"
        return f"{uniqueID}\n{str(self)}"

    # Main routines -----------------------------------------------------------
    def _check_shape(self, Q):
        """Verify the shape of the snapshot set Q."""
        if Q.shape[0] != self.state_dimension:
            raise ValueError(
                f"states.shape[0] = {Q.shape[0]:d} "
                f"!= {self.state_dimension} = state dimension n"
            )

    def _is_trained(self):
        """Return True if transform() and inverse_transform() are ready."""
        if self.state_dimension is None:
            return False
        if self.centering and self.mean_ is None:
            return False
        if self.scaling and any(
            getattr(self, attr) is None for attr in ("scale_", "shift_")
        ):
            return False
        return True

    def _check_is_trained(self):
        """Raise an exception if the transformer is not trained."""
        if not self._is_trained():
            raise AttributeError(
                "transformer not trained (call fit() or fit_transform())"
            )

    def fit_transform(self, states, inplace=False):
        """Learn and apply the transformation.

        Parameters
        ----------
        states : (n, k) ndarray
            Matrix of k snapshots. Each column is a snapshot of dimension n.
        inplace : bool
            If ``True``, overwrite the input data during transformation.
            If ``False``, create a copy of the data to transform.

        Returns
        -------
        states_transformed: (n, k) ndarray
            Matrix of k transformed snapshots of dimension n.
        """
        if states.ndim != 2:
            raise ValueError("2D array required to fit transformer")
        self.state_dimension = states.shape[0]

        Y = states if inplace else states.copy()
        axis = 1 if self.byrow else None

        # Record statistics of the training data.
        if self.verbose:
            report = ["No transformation learned"]
            report.append(self._table_header)
            report.append(f"Q   | {self._statistics_report(Y)}")

        # Center the snapshots by the mean training snapshot.
        if self.centering:
            self.mean_ = np.mean(Y, axis=1)
            Y -= self.mean_.reshape((-1, 1))

            if self.verbose:
                report[0] = "Learned mean centering Q -> Q'"
                report.append(f"Q'  | {self._statistics_report(Y)}")

        # Scale (non-dimensionalize) the centered snapshot entries.
        if self.scaling:
            # Standard: Q' = (Q - mu)/sigma
            if self.scaling == "standard":
                mu = np.mean(Y, axis=axis)
                sigma = np.std(Y, axis=axis)
                self.scale_ = 1 / sigma
                self.shift_ = -mu * self.scale_

            # Min-max: Q' = (Q - min(Q))/(max(Q) - min(Q))
            elif self.scaling == "minmax":
                Ymin = np.min(Y, axis=axis)
                Ymax = np.max(Y, axis=axis)
                self.scale_ = 1 / (Ymax - Ymin)
                self.shift_ = -Ymin * self.scale_

            # Symmetric min-max: Q' = (Q - min(Q))*2/(max(Q) - min(Q)) - 1
            elif self.scaling == "minmaxsym":
                Ymin = np.min(Y, axis=axis)
                Ymax = np.max(Y, axis=axis)
                self.scale_ = 2 / (Ymax - Ymin)
                self.shift_ = -Ymin * self.scale_ - 1

            # MaxAbs: Q' = Q / max(abs(Q))
            elif self.scaling == "maxabs":
                self.scale_ = 1 / np.max(np.abs(Y), axis=axis)
                self.shift_ = (
                    0 if axis is None else np.zeros(self.state_dimension)
                )

            # maxabssym: Q' = (Q - mean(Q)) / max(abs(Q - mean(Q)))
            elif self.scaling == "maxabssym":
                mu = np.mean(Y, axis=axis)
                Y -= mu if axis is None else mu.reshape((-1, 1))
                self.scale_ = 1 / np.max(np.abs(Y), axis=axis)
                self.shift_ = -mu * self.scale_
                Y += mu if axis is None else mu.reshape((-1, 1))

            else:  # pragma nocover
                raise RuntimeError(f"invalid scaling '{self.scaling}'")

            # Apply the scaling.
            Y *= self.scale_ if axis is None else self.scale_.reshape((-1, 1))
            Y += self.shift_ if axis is None else self.shift_.reshape((-1, 1))

            if self.verbose:
                if self.centering:
                    report[0] += f" and {self.scaling} scaling Q' -> Q''"
                else:
                    report[0] = f"Learned {self.scaling} scaling Q -> Q''"
                report.append(f"Q'' | {self._statistics_report(Y)}")

        if self.verbose:
            print("\n".join(report) + "\n")

        return Y

    def transform(self, states, inplace=False):
        """Apply the learned transformation.

        Parameters
        ----------
        states : (n, k) or (n,) ndarray
            Matrix of k snapshots where each column is a snapshot of dimension
            n, or a single snapshot of dimension n.
        inplace : bool
            If ``True``, overwrite the input data during transformation.
            If ``False``, create a copy of the data to transform.

        Returns
        -------
        states_transformed: (n, k) ndarray
            Matrix of k transformed snapshots of dimension n.
        """
        self._check_is_trained()
        self._check_shape(states)
        Y = states if inplace else states.copy()

        # Center the snapshots by the mean training snapshot.
        if self.centering:
            Y -= self.mean_.reshape((-1, 1)) if Y.ndim > 1 else self.mean_

        # Scale (non-dimensionalize) the centered snapshot entries.
        if self.scaling is not None:
            _flip = self.byrow and Y.ndim > 1
            Y *= self.scale_.reshape((-1, 1)) if _flip else self.scale_
            Y += self.shift_.reshape((-1, 1)) if _flip else self.shift_

        return Y

    def transform_ddts(self, ddts, inplace=False):
        r"""Apply the learned transformation to snapshot time derivatives.

        Denoting the transformation by
        :math:`\mathcal{T}(\q) = \alpha(\q - \bar{\q}) + \beta`,
        this is the function :math:`\mathcal{T}'(\z) = \alpha\z`.
        Hence, :math:`\mathcal{T}'(\ddt q) = \ddt \mathcal{T}(q)`.

        Parameters
        ----------
        ddts : (n, k) ndarray
            Matrix of k snapshot time derivatives.
        inplace : bool
            If True, overwrite ``ddts`` during the transformation.
            If False, create a copy of the data to transform.

        Returns
        -------
        ddts_transformed : (n, k) ndarray
            Matrix of k transformed snapshot time derivatives.
        """
        self._check_is_trained()
        self._check_shape(ddts)
        Z = ddts if inplace else ddts.copy()

        if self.scaling is not None:
            _flip = self.byrow and Z.ndim > 1
            Z *= self.scale_.reshape((-1, 1)) if _flip else self.scale_

        return Z

    def inverse_transform(self, states_transformed, inplace=False, locs=None):
        """Apply the inverse of the learned transformation.

        Parameters
        ----------
        states_transformed : (n, k) or (p, k) ndarray
            Matrix of k transformed snapshots of dimension n.
        inplace : bool
            If ``True``, overwrite ``states_transformed`` during the inverse
            transformation. If ``False``, create a copy of the data to
            untransform.
        locs : slice or (p,) ndarray of integers or None
            If given, assume ``states_transformed`` contains the transformed
            snapshots at only the p indices given by ``locs``.

        Returns
        -------
        states: (n, k) or (p, k) ndarray
            Matrix of k untransformed snapshots of dimension n, or the p
            entries of such at the indices specified by ``loc``.
        """
        self._check_is_trained()

        if locs is not None:
            if isinstance(locs, slice):
                locs = np.arange(self.state_dimension)[locs]
            if states_transformed.shape[0] != locs.size:
                raise ValueError("states_transformed not aligned with locs")
        else:
            self._check_shape(states_transformed)

        Y = states_transformed if inplace else states_transformed.copy()

        # Unscale (re-dimensionalize) the data.
        if self.scaling:
            Y -= self.shift_
            Y /= self.scale_

        # Uncenter the unscaled snapshots.
        if self.centering:
            mean_ = self.mean_ if locs is None else self.mean_[locs]
            Y += mean_.reshape((-1, 1)) if Y.ndim > 1 else mean_

        return Y

    # Model persistence -------------------------------------------------------
    def save(self, savefile, overwrite=False):
        """Save the current transformer to an HDF5 file.

        Parameters
        ----------
        savefile : str
            Path of the file to save the transformer in.
        overwrite : bool
            If ``True``, overwrite the file if it already exists. If ``False``
            (default), raise a ``FileExistsError`` if the file already exists.
        """
        with utils.hdf5_savehandle(savefile, overwrite) as hf:
            # Store transformation hyperparameter metadata.
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["centering"] = self.centering
            meta.attrs["scaling"] = self.scaling if self.scaling else False
            meta.attrs["byrow"] = self.byrow
            meta.attrs["verbose"] = self.verbose

            # Store learned transformation parameters.
            n = self.state_dimension
            meta.attrs["state_dimension"] = n if n is not None else False
            if self.centering and self.mean_ is not None:
                hf.create_dataset("transformation/mean_", data=self.mean_)
            if self.scaling and self.scale_ is not None:
                hf.create_dataset(
                    "transformation/scale_",
                    data=self.scale_ if self.byrow else [self.scale_],
                )
                hf.create_dataset(
                    "transformation/shift_",
                    data=self.shift_ if self.byrow else [self.shift_],
                )

    @classmethod
    def load(cls, loadfile):
        """Load a previously saved transformer from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            File where the transformer was stored via :meth:`save()`.

        Returns
        -------
        SnapshotTransformer
        """
        with utils.hdf5_loadhandle(loadfile) as hf:
            # Load transformation hyperparameters.
            meta = hf["meta"]
            scl = meta.attrs["scaling"]
            transformer = cls(
                centering=bool(meta.attrs["centering"]),
                scaling=scl if scl else None,
                byrow=meta.attrs["byrow"],
                verbose=meta.attrs["verbose"],
            )

            # Load learned transformation parameters.
            n = meta.attrs["state_dimension"]
            transformer.state_dimension = None if not n else n
            if transformer.centering and "transformation/mean_" in hf:
                transformer.mean_ = hf["transformation/mean_"][:]
            if transformer.scaling and "transformation/scale_" in hf:
                ind = slice(None) if transformer.byrow else 0
                transformer.scale_ = hf["transformation/scale_"][ind]
                transformer.shift_ = hf["transformation/shift_"][ind]

            return transformer


class SnapshotTransformerMulti(TransformerTemplate, _MultivarMixin):
    """Transformer for multivariate snapshots.

    Groups multiple SnapshotTransformers for the centering and/or scaling
    (in that order) of individual variables.

    Parameters
    ----------
    num_variables : int
        Number of variables represented in a single snapshot (number of
        individual transformations to learn). The dimension `n` of the
        snapshots must be evenly divisible by num_variables; for example,
        ``num_variables=3`` means the first `n` entries of a snapshot
        correspond to the first variable, and the next `n` entries correspond
        to the second variable, and the last `n` entries correspond to the
        third variable.
    centering : bool OR list of num_variables bools
        If True, shift the snapshots by the mean training snapshot.
        If a list, centering[i] is the centering directive for the ith
        variable.
    scaling : str, None, OR list of length num_variables
        If given, scale (non-dimensionalize) the centered snapshot entries.
        If a list, scaling[i] is the scaling directive for the ith variable.
        * 'standard': standardize to zero mean and unit standard deviation.
        * 'minmax': minmax scaling to [0, 1].
        * 'minmaxsym': minmax scaling to [-1, 1].
        * 'maxabs': maximum absolute scaling to [-1, 1] (no shift).
        * 'maxabssym': maximum absolute scaling to [-1, 1] (mean shift).
    variable_names : list of num_variables strings
        Names for each of the `num_variables` variables.
        Defaults to 'x1', 'x2', ....
    verbose : bool
        If True, print information upon learning a transformation.

    Attributes
    ----------
    transformers : list of num_variables SnapshotTransformers
        Transformers for each snapshot variable.
    n : int
        Total dimension of the snapshots (all variables).
    ni : int
        Dimension of individual variables, i.e., ni = n / num_variables.

    Notes
    -----
    See SnapshotTransformer for details on available transformations.

    Examples
    --------
    # Center first and third variables and minmax scale the second variable.
    >>> stm = SnapshotTransformerMulti(
    ...     3,
    ...     centering=(True, False, True),
    ...     scaling=(None, "minmax", None),
    ... )
    # Center 6 variables and scale the final variable with a standard scaling.
    >>> stm = SnapshotTransformerMulti(
    ...     6,
    ...     centering=True,
    ...     scaling=(None, None, None, None, None, "standard")
    ... )
    # OR
    >>> stm = SnapshotTransformerMulti(6, centering=True, scaling=None)
    >>> stm[-1].scaling = "standard"
    """

    def __init__(
        self,
        num_variables,
        centering=False,
        scaling=None,
        variable_names=None,
        verbose=False,
    ):
        """Interpret hyperparameters and initialize transformers."""
        _MultivarMixin.__init__(self, num_variables, variable_names)

        def _process_arg(attr, name, dtype):
            """Validation for centering and scaling directives."""
            if isinstance(attr, dtype):
                attr = (attr,) * num_variables
            if len(attr) != num_variables:
                raise ValueError(
                    f"len({name}) = {len(attr)} "
                    f"!= {num_variables} = num_variables"
                )
            return attr

        # Process and store transformation directives.
        centers = _process_arg(centering, "centering", bool)
        scalings = _process_arg(scaling, "scaling", (type(None), str))

        # Initialize transformers.
        self.transformers = [
            SnapshotTransformer(
                centering=ctr, scaling=scl, byrow=False, verbose=False
            )
            for ctr, scl in zip(centers, scalings)
        ]
        self.verbose = verbose

    # Properties --------------------------------------------------------------
    @property
    def centering(self):
        """Snapshot mean-centering directive."""
        return tuple(st.centering for st in self.transformers)

    @property
    def scaling(self):
        """Entrywise scaling (non-dimensionalization) directive.
        * None: no scaling.
        * 'standard': standardize to zero mean and unit standard deviation.
        * 'minmax': minmax scaling to [0, 1].
        * 'minmaxsym': minmax scaling to [-1, 1].
        * 'maxabs': maximum absolute scaling to [-1, 1] (no shift).
        * 'maxabssym': maximum absolute scaling to [-1, 1] (mean shift).
        """
        return tuple(st.scaling for st in self.transformers)

    @property
    def variable_names(self):
        """Names for each of the `num_variables` variables."""
        return self.__variable_names

    @variable_names.setter
    def variable_names(self, names):
        if names is None:
            names = [f"variable {i+1}" for i in range(self.num_variables)]
        if not isinstance(names, list) or len(names) != self.num_variables:
            raise TypeError(
                "variable_names must be list of"
                f" length {self.num_variables}"
            )
        self.__variable_names = names

    @property
    def verbose(self):
        """If True, print information about upon learning a transformation."""
        return self.__verbose

    @verbose.setter
    def verbose(self, vbs):
        """Set verbosity of all transformers (uniformly)."""
        self.__verbose = bool(vbs)
        for st in self.transformers:
            st.verbose = self.__verbose

    @property
    def mean_(self):
        """Mean training snapshot across all transforms ((n,) ndarray)."""
        if not self._is_trained():
            return None
        zeros = np.zeros(self.ni)
        return np.concatenate(
            [(st.mean_ if st.centering else zeros) for st in self.transformers]
        )

    def __getitem__(self, key):
        """Get the transformer for variable i."""
        if key in self.variable_names:
            key = self.variable_names.index(key)
        return self.transformers[key]

    def __setitem__(self, key, obj):
        """Set the transformer for variable i."""
        if not isinstance(obj, SnapshotTransformer):
            raise TypeError("assignment object must be SnapshotTransformer")
        self.transformers[key] = obj

    def __len__(self):
        """Length = number of variables."""
        return len(self.transformers)

    def __eq__(self, other):
        """Test two SnapshotTransformerMulti objects for equality."""
        if not isinstance(other, self.__class__):
            return False
        if self.num_variables != other.num_variables:
            return False
        return all(
            t1 == t2 for t1, t2 in zip(self.transformers, other.transformers)
        )

    # Printing ----------------------------------------------------------------
    def __str__(self):
        """String representation: centering and scaling directives."""
        out = [f"{self.num_variables}-variable snapshot transformer"]
        namelength = max(len(name) for name in self.variable_names)
        for name, st in zip(self.variable_names, self.transformers):
            out.append(f"* {{:>{namelength}}} | {st}".format(name))
        return "\n".join(out)

    def __repr__(self):
        """Unique ID + string representation."""
        uniqueID = f"<{self.__class__.__name__} object at {hex(id(self))}>"
        return f"{uniqueID}\n{str(self)}"

    # Main routines -----------------------------------------------------------
    def _is_trained(self):
        """Return True if transform() and inverse_transform() are ready."""
        return all(st._is_trained() for st in self.transformers)

    def _check_is_trained(self):
        """Raise an exception if the transformer is not trained."""
        if not self._is_trained():
            raise AttributeError(
                "transformer not trained (call fit() or fit_transform())"
            )

    def _apply(self, method, Q, inplace, locs=None):
        """
        Apply a method of each transformer to the corresponding chunk of Q.
        """
        options = dict(inplace=inplace)
        if locs is not None:
            options["locs"] = locs

        Ys = []
        for st, var, name in zip(
            self.transformers,
            np.split(Q, self.num_variables, axis=0),
            self.variable_names,
        ):
            if method is SnapshotTransformer.fit_transform and self.verbose:
                print(f"{name}:")
            Ys.append(method(st, var, **options))

        if inplace and locs is None:
            return Q
        return np.concatenate(Ys, axis=0)

    def fit_transform(self, states, inplace=False):
        """Learn and apply the transformation.

        Parameters
        ----------
        states : (n, k) ndarray
            Matrix of k snapshots. Each column is a snapshot of dimension n;
            this dimension must be evenly divisible by `num_variables`.
        inplace : bool
            If True, overwrite the input data during transformation.
            If False, create a copy of the data to transform.

        Returns
        -------
        states_transformed: (n, k) ndarray
            Matrix of k transformed n-dimensional snapshots.
        """
        if states.ndim != 2:
            raise ValueError("2D array required to fit transformer")
        self.n = states.shape[0]
        return self._apply(SnapshotTransformer.fit_transform, states, inplace)

    def transform(self, states, inplace=False):
        """Apply the learned transformation.

        Parameters
        ----------
        states : (n, k) ndarray
            Matrix of k snapshots. Each column is a snapshot of dimension n;
            this dimension must be evenly divisible by `num_variables`.
        inplace : bool
            If True, overwrite the input data during transformation.
            If False, create a copy of the data to transform.

        Returns
        -------
        states_transformed: (n, k) ndarray
            Matrix of k transformed n-dimensional snapshots.
        """
        self._check_is_trained()
        return self._apply(SnapshotTransformer.transform, states, inplace)

    def inverse_transform(self, states_transformed, inplace=False, locs=None):
        """Apply the inverse of the learned transformation.

        Parameters
        ----------
        states_transformed : (n, k) or (num_variables*p, k) ndarray
            Matrix of k transformed n-dimensional snapshots, or the p entries
            of the snapshots at the indices specified by `locs`.
        inplace : bool
            If True, overwrite the input data during inverse transformation.
            If False, create a copy of the data to untransform.
        locs : slice or (p,) ndarray of integers or None
            If given, assume `states_transformed` contains the transformed
            snapshots at only the indices given by `locs` for each variable.

        Returns
        -------
        states: (n, k) or (num_variables*p, k) ndarray
            Matrix of k untransformed snapshots of dimension n, or the p
            entries of such at the indices specified by `loc`.
        """
        self._check_is_trained()
        if locs is None:
            self._check_shape(states_transformed)
        return self._apply(
            SnapshotTransformer.inverse_transform,
            states_transformed,
            inplace,
            locs,
        )

    # Model persistence -------------------------------------------------------
    def save(self, savefile, overwrite=False):
        """Save the current transformers to an HDF5 file.

        Parameters
        ----------
        savefile : str
            Path of the file to save the transformer in.
        overwrite : bool
            If True, overwrite the file if it already exists. If False
            (default), raise a FileExistsError if the file already exists.
        """
        with utils.hdf5_savehandle(savefile, overwrite) as hf:
            # Metadata
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["num_variables"] = self.num_variables
            meta.attrs["verbose"] = self.verbose
            meta.attrs["variable_names"] = self.variable_names

            for i in range(self.num_variables):
                self.transformers[i].save(hf.create_group(f"variable{i+1}"))

    @classmethod
    def load(cls, loadfile):
        """Load a SnapshotTransformerMulti object from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            Path to the file where the transformer was stored (via save()).

        Returns
        -------
        SnapshotTransformerMulti
        """
        with utils.hdf5_loadhandle(loadfile) as hf:
            # Load transformation hyperparameters.
            if "meta" not in hf:
                raise errors.LoadfileFormatError(
                    "invalid save format (meta/ not found)"
                )
            num_variables = hf["meta"].attrs["num_variables"]
            verbose = hf["meta"].attrs["verbose"]
            names = hf["meta"].attrs["variable_names"].tolist()
            stm = cls(num_variables, variable_names=names, verbose=verbose)

            # Initialize individual transformers.
            for i in range(num_variables):
                group = f"variable{i+1}"
                if group not in hf:
                    raise errors.LoadfileFormatError(
                        f"invalid save format ({group}/ not found)"
                    )
                stm[i] = SnapshotTransformer.load(hf[group])
            stm.n = stm[0].n * num_variables

            return stm
