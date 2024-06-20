# pre/_shiftscale.py
"""Preprocessing transformations based on elementary shifts and scalings."""

__all__ = [
    "shift",
    "scale",
    "ShiftScaleTransformer",
]

import warnings
import numpy as np

from .. import errors, utils
from ._base import TransformerTemplate


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
    r"""Scale the entries of a snapshot matrix to a specified interval.

    The scaling from the interval :math:`[a, b]` to the interval
    :math:`[a', b']` given by

    .. math::
       q' = \frac{q - a}{b - a}(b' - a') + a',

    where :math:`q` is the original variable and :math:`q'` is the transformed
    variable. This follows :class:`sklearn.preprocessing.MinMaxScaler`.

    Parameters
    ----------
    states : (n, k) ndarray
        Matrix of k snapshots to be scaled. Each column is a single snapshot.
    scale_to : (float, float)
        Desired minimum and maximum of the scaled data, i.e., :math:`[a', b']`.
    scale_from : (float, float)
        Minimum and maximum of the snapshot data, i.e., :math:`[a, b]`.
        If ``None`` (default), learn the scaling from the data:
        ``scale_from[0] = min(states)``; ``scale_from[1] = max(states)``.

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
class ShiftScaleTransformer(TransformerTemplate):
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

    Notes
    -----
    A custom shifting vector (i.e., the mean snapshot) can be specified by
    setting the ``mean_`` attribute. Similarly, the scaling
    :math:`q'\mapsto q'' = \alpha q' + \beta` can be adjusted by setting the
    ``scale_`` (:math:`\alpha`) and ``shift_`` (:math:`\beta`) attributes.
    However, calling :meth:`fit()` or :meth:`fit_transform()` will overwrite
    all three attributes.
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

    # TODO: allow scaling to be a tuple [a, b] to scale to (as in scale()).
    def __init__(
        self,
        centering: bool = False,
        scaling: str = None,
        byrow: bool = False,
        name: str = None,
        verbose: bool = False,
    ):
        """Set transformation hyperparameters."""
        # Centering is always a boolean.
        self.__centering = bool(centering)

        # Verify scaling.
        if scaling is not None:
            if not isinstance(scaling, str):
                raise TypeError("'scaling' must be None or of type 'str'")
            if scaling not in self._VALID_SCALINGS:
                opts = ", ".join([f"'{v}'" for v in self._VALID_SCALINGS])
                raise ValueError(
                    f"invalid scaling '{scaling}'; valid options are {opts}"
                )
        self.__scaling = scaling

        # Set byrow, warn if not applied.
        self.__byrow = bool(byrow)
        if self.__byrow and self.scaling is None:
            warnings.warn(
                "scaling=None --> byrow=True will have no effect",
                errors.OpInfWarning,
            )

        # Set other properties.
        self.verbose = verbose
        self.__qbar = None
        self.__alpha = None
        self.__beta = None
        TransformerTemplate.__init__(self, name)

    # Properties: transformation directives -----------------------------------
    @property
    def centering(self) -> bool:
        """If ``True``, center the snapshots by the mean training snapshot."""
        return self.__centering

    @property
    def scaling(self) -> str:
        """Type of scaling (non-dimensionalization)."""
        return self.__scaling

    @property
    def byrow(self) -> bool:
        """If ``True``, scale each row of the snapshot matrix separately."""
        return self.__byrow

    @property
    def verbose(self) -> bool:
        """If ``True``, print information upon learning a transformation."""
        return self.__verbose

    @verbose.setter
    def verbose(self, vbs):
        """Set the verbosity."""
        self.__verbose = bool(vbs)

    # Properties: calibrated quantities ---------------------------------------
    @property
    def mean_(self):
        """Mean training snapshot. ``None`` unless ``centering = True``."""
        return self.__qbar

    @mean_.setter
    def mean_(self, mean):
        """Set the mean vector."""
        if not self.centering:
            raise AttributeError("cannot set mean_ (centering=False)")
        if self.state_dimension is None:
            if np.ndim(mean) != 1:
                raise ValueError("expected one-dimensional mean_")
            self.state_dimension = mean.shape[0]
        if np.shape(mean) != ((n := self.state_dimension),):
            raise ValueError(f"expected mean_ to be ({n:d},) ndarray")
        self.__qbar = mean

    @property
    def scale_(self):
        r"""Multiplicative factor of the scaling, the :math:`\alpha` of
        :math:`q'' = \alpha q' + \beta`.
        """
        return self.__alpha

    @scale_.setter
    def scale_(self, alpha):
        """Set the multiplicative factor of the scaling."""
        if self.scaling is None:
            raise AttributeError("cannot set scale_ (scaling=None)")
        if self.byrow:
            if self.state_dimension is None:
                if np.ndim(alpha) != 1 or np.isscalar(alpha):
                    raise ValueError("expected one-dimensional scale_")
                self.state_dimension = alpha.shape[0]
            if np.shape(alpha) != ((n := self.state_dimension),):
                raise ValueError(f"expected scale_ to be ({n:d},) ndarray")
        self.__alpha = alpha

    @property
    def shift_(self):
        r"""Additive factor of the scaling, the :math:`\beta` of
        :math:`q'' = \alpha q' + \beta`.
        """
        return self.__beta

    @shift_.setter
    def shift_(self, beta):
        """Set the multiplicative factor of the scaling."""
        if self.scaling is None:
            raise AttributeError("cannot set shift_ (scaling=None)")
        if self.byrow:
            if self.state_dimension is None:
                if np.ndim(beta) != 1 or np.isscalar(beta):
                    raise ValueError("expected one-dimensional shift_")
                self.state_dimension = beta.shape[0]
            if np.shape(beta) != ((n := self.state_dimension),):
                raise ValueError(f"expected shift_ to be ({n:d},) ndarray")
        self.__beta = beta

    def __eq__(self, other) -> bool:
        """Test two ShiftScaleTransformers for equality."""
        if not isinstance(other, self.__class__):
            return False
        for attr in ("centering", "scaling", "byrow"):
            if getattr(self, attr) != getattr(other, attr):
                return False
        if self.state_dimension != other.state_dimension:
            return False
        if self.centering and self.mean_ is not None:
            if other.mean_ is None:
                return False
            if not np.all(self.mean_ == other.mean_):
                return False
        if self.scaling and self.scale_ is not None:
            for attr in ("scale_", "shift_"):
                if (oat := getattr(other, attr)) is None:
                    return False
                if not np.all(getattr(self, attr) == oat):
                    return False
        return True

    # Printing ----------------------------------------------------------------
    @staticmethod
    def _statistics_report(Q) -> str:
        """Return a string of basis statistics about a data set."""
        return " | ".join(
            [f"{f(Q):>10.3e}" for f in (np.min, np.mean, np.max, np.std)]
        )

    def __str__(self) -> str:
        """String representation: scaling type + centering bool."""
        out = [self.__class__.__name__]
        if self.state_dimension is not None:
            out.append(f"(state dimension n = {self.state_dimension:d})")
        if self.centering:
            out.append("with mean-snapshot centering")
            if self.scaling:
                out.append(f"and '{self.scaling}' scaling")
        elif self.scaling:
            out.append(f"with '{self.scaling}' scaling")
        if not self._is_trained():
            out.append("(call fit() or fit_transform() to train)")
        return " ".join(out)

    # Main routines -----------------------------------------------------------
    def _is_trained(self) -> bool:
        """Return True if transform() and inverse_transform() are ready."""
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

    def fit_transform(self, states, inplace: bool = False):
        """Learn and apply the transformation.

        Parameters
        ----------
        states : (n, k) ndarray
            Matrix of `k` `n`-dimensional snapshots.
        inplace : bool
            If ``True``, overwrite ``states`` during transformation.
            If ``False``, create a copy of the data to transform.

        Returns
        -------
        states_transformed: (n, k) ndarray
            Matrix of `k` `n`-dimensional transformed snapshots.
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
            if self.name is not None:
                report.insert(0, f"<{self.name}>")
            print("\n".join(report) + "\n")

        return Y

    def transform(self, states, inplace: bool = False):
        """Apply the learned transformation.

        Parameters
        ----------
        states : (n, ...) ndarray
            Matrix of `n`-dimensional snapshots, or a single snapshot.
        inplace : bool
            If ``True``, overwrite ``states`` during transformation.
            If ``False``, create a copy of the data to transform.

        Returns
        -------
        states_transformed: (n, ...) ndarray
            Matrix of `n`-dimensional transformed snapshots, or a single
            transformed snapshot.
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

    def transform_ddts(self, ddts, inplace: bool = False):
        r"""Apply the learned transformation to snapshot time derivatives.

        Denoting the transformation by
        :math:`\mathcal{T}(\q) = \alpha(\q - \bar{\q}) + \beta`,
        this is the function :math:`\mathcal{T}'(\z) = \alpha\z`.
        Hence, :math:`\mathcal{T}'(\ddt q) = \ddt \mathcal{T}(q)`.

        Parameters
        ----------
        ddts : (n, ...) ndarray
            Matrix of `n`-dimensional snapshot time derivatives, or a
            single snapshot time derivative.
        inplace : bool
            If True, overwrite ``ddts`` during the transformation.
            If False, create a copy of the data to transform.

        Returns
        -------
        ddts_transformed : (n, ...) ndarray
            Transformed `n`-dimensional snapshot time derivatives.
        """
        self._check_is_trained()
        self._check_shape(ddts)
        Z = ddts if inplace else ddts.copy()

        if self.scaling is not None:
            _flip = self.byrow and Z.ndim > 1
            Z *= self.scale_.reshape((-1, 1)) if _flip else self.scale_

        return Z

    def inverse_transform(
        self,
        states_transformed,
        inplace: bool = False,
        locs=None,
    ):
        """Apply the inverse of the learned transformation.

        Parameters
        ----------
        states_transformed : (n, ...) or (p, ...)  ndarray
            Matrix of `n`-dimensional transformed snapshots, or a single
            transformed snapshot.
        inplace : bool
            If ``True``, overwrite ``states_transformed`` during the inverse
            transformation. If ``False``, create a copy of the data to
            untransform.
        locs : slice or (p,) ndarray of integers or None
            If given, assume ``states_transformed`` contains the transformed
            snapshots at only the `p` indices described by ``locs``.

        Returns
        -------
        states_untransformed: (n, ...) or (p, ...) ndarray
            Matrix of `n`-dimensional untransformed snapshots, or the `p`
            entries of such at the indices specified by ``locs``.
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
    def save(self, savefile: str, overwrite: bool = False) -> None:
        """Save the transformer to an HDF5 file.

        Parameters
        ----------
        savefile : str
            Path of the file to save the transformer to.
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
            meta.attrs["name"] = str(self.name)

            # Store learned transformation parameters.
            n = self.state_dimension
            meta.attrs["state_dimension"] = n if n is not None else False
            if self.centering and self.mean_ is not None:
                hf.create_dataset(
                    "transformation/mean_",
                    data=self.mean_,
                )
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
    def load(cls, loadfile: str):
        """Load a previously saved transformer from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            File where the transformer was stored via :meth:`save()`.

        Returns
        -------
        ShiftScaleTransformer
        """
        with utils.hdf5_loadhandle(loadfile) as hf:
            # Load transformation hyperparameters.
            meta = hf["meta"]
            scl = meta.attrs["scaling"]
            name = meta.attrs["name"]

            # Instantiate transformer.
            transformer = cls(
                centering=bool(meta.attrs["centering"]),
                scaling=(scl if scl else None),
                byrow=meta.attrs["byrow"],
                name=(None if name == "None" else name),
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
