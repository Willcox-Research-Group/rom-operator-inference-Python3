# pre/_shift_scale.py
"""Tools for preprocessing data."""

__all__ = [
            "shift",
            "scale",
            "SnapshotTransformer",
            "SnapshotTransformerMulti",
          ]

import os
import h5py
import numpy as np


# Shifting and MinMax scaling =================================================
def shift(X, shift_by=None):
    """Shift the columns of X by a vector.

    Parameters
    ----------
    X : (n,k) ndarray
        A matrix of k snapshots. Each column is a single snapshot.
    shift_by : (n,) or (n,1) ndarray
        A vector that is the same size as a single snapshot. If None,
        set to the mean of the columns of X.

    Returns
    -------
    Xshifted : (n,k) ndarray
        The matrix such that Xshifted[:,j] = X[:,j] - shift_by for j=0,...,k-1.
    xbar : (n,) ndarray
        The shift factor. Since this is a one-dimensional array, it must be
        reshaped to be applied to a matrix: Xshifted + xbar.reshape((-1,1)).
        Only returned if shift_by=None.

    Examples
    --------
    # Shift X by its mean, then shift Y by the same mean.
    >>> Xshifted, xbar = pre.shift(X)
    >>> Yshifted = pre.shift(Y, xbar)

    # Shift X by its mean, then undo the transformation by an inverse shift.
    >>> Xshifted, xbar = pre.shift(X)
    >>> X_again = pre.shift(Xshifted, -xbar)
    """
    # Check dimensions.
    if X.ndim != 2:
        raise ValueError("data X must be two-dimensional")

    # If not shift_by factor is provided, compute the mean column.
    learning = (shift_by is None)
    if learning:
        shift_by = np.mean(X, axis=1)
    elif shift_by.ndim != 1:
        raise ValueError("shift_by must be one-dimensional")

    # Shift the columns by the mean.
    Xshifted = X - shift_by.reshape((-1,1))

    return (Xshifted, shift_by) if learning else Xshifted


def scale(X, scale_to, scale_from=None):
    """Scale the entries of the snapshot matrix X from the interval
    [scale_from[0], scale_from[1]] to [scale_to[0], scale_to[1]].
    Scaling algorithm follows sklearn.preprocessing.MinMaxScaler.

    Parameters
    ----------
    X : (n,k) ndarray
        A matrix of k snapshots to be scaled. Each column is a single snapshot.
    scale_to : (2,) tuple
        The desired minimum and maximum of the scaled data.
    scale_from : (2,) tuple
        The minimum and maximum of the snapshot data. If None, learn the
        scaling from X: scale_from[0] = min(X); scale_from[1] = max(X).

    Returns
    -------
    Xscaled : (n,k) ndarray
        The scaled snapshot matrix.
    scaled_to : (2,) tuple
        The bounds that the snapshot matrix was scaled to, i.e.,
        scaled_to[0] = min(Xscaled); scaled_to[1] = max(Xscaled).
        Only returned if scale_from = None.
    scaled_from : (2,) tuple
        The minimum and maximum of the snapshot data, i.e., the bounds that
        the data was scaled from. Only returned if scale_from = None.

    Examples
    --------
    # Scale X to [-1,1] and then scale Y with the same transformation.
    >>> Xscaled, scaled_to, scaled_from = pre.scale(X, (-1,1))
    >>> Yscaled = pre.scale(Y, scaled_to, scaled_from)

    # Scale X to [0,1], then undo the transformation by an inverse scaling.
    >>> Xscaled, scaled_to, scaled_from = pre.scale(X, (0,1))
    >>> X_again = pre.scale(Xscaled, scaled_from, scaled_to)
    """
    # If no scale_from bounds are provided, learn them.
    learning = (scale_from is None)
    if learning:
        scale_from = np.min(X), np.max(X)

    # Check scales.
    if len(scale_to) != 2:
        raise ValueError("scale_to must have exactly 2 elements")
    if len(scale_from) != 2:
        raise ValueError("scale_from must have exactly 2 elements")

    # Do the scaling.
    mini, maxi = scale_to
    xmin, xmax = scale_from
    scl = (maxi - mini)/(xmax - xmin)
    Xscaled = X*scl + (mini - xmin*scl)

    return (Xscaled, scale_to, scale_from) if learning else Xscaled


class SnapshotTransformer:
    """Process snapshots by centering and/or scaling (in that order).

    Parameters
    ----------
    center : bool
        If True, shift the snapshots by the mean training snapshot.
    scaling : str or None
        If given, scale (non-dimensionalize) the centered snapshot entries.
        * 'standard': standardize to zero mean and unit standard deviation.
        * 'minmax': minmax scaling to [0,1].
        * 'minmaxsym': minmax scaling to [-1,1].
        * 'maxabs': maximum absolute scaling to [-1,1] (no shift).
        * 'maxabssym': maximum absolute scaling to [-1,1] (mean shift).
    verbose : bool
        If True, print information upon learning a transformation.

    Attributes
    ----------
    mean_ : (n,) ndarray
        Mean training snapshot. Only recorded if center = True.
    scale_ : float
        Multiplicative factor of scaling (the m of x -> mx + b).
        Only recorded if scaling != None.
    shift_ : float
        Additive factor of scaling (the b of x -> mx + b).
        Only recorded if scaling != None.

    Notes
    -----
    Snapshot centering (center=True):
        X' = X - mean(X, axis=1);
        Guarantees mean(X', axis=1) = [0, ..., 0].
    Standard scaling (scaling='standard'):
        X' = (X - mean(X)) / std(X);
        Guarantees mean(X') = 0, std(X') = 1.
    Min-max scaling (scaling='minmax'):
        X' = (X - min(X))/(max(X) - min(X));
        Guarantees min(X') = 0, max(X') = 1.
    Symmetric min-max scaling (scaling='minmaxsym'):
        X' = (X - min(X))*2/(max(X) - min(X)) - 1
        Guarantees min(X') = -1, max(X') = 1.
    Maximum absolute scaling (scaling='maxabs'):
        X' = X / max(abs(X));
        Guarantees mean(X') = mean(X) / max(abs(X)), max(abs(X')) = 1.
    Min-max absolute scaling (scaling='maxabssym'):
        X' = (X - mean(X)) / max(abs(X - mean(X)));
        Guarantees mean(X') = 0, max(abs(X')) = 1.
    """
    _VALID_SCALINGS = {
        "standard",
        "minmax",
        "minmaxsym",
        "maxabs",
        "maxabssym",
    }

    _table_header = "    |     min    |    mean    |     max    |    std\n"
    _table_header += "----|------------|------------|------------|------------"

    def __init__(self, center=False, scaling=None, verbose=False):
        """Set transformation hyperparameters."""
        self.center = center
        self.scaling = scaling
        self.verbose = verbose

    def _clear(self):
        """Delete all learned attributes."""
        for attr in ("mean_", "scale_", "shift_"):
            if hasattr(self, attr):
                delattr(self, attr)

    # Properties --------------------------------------------------------------
    @property
    def center(self):
        """Snapshot mean-centering directive (bool)."""
        return self.__center

    @center.setter
    def center(self, ctr):
        """Set the centering directive, resetting the transformation."""
        if ctr not in (True, False):
            raise TypeError("'center' must be True or False")
        self._clear()
        self.__center = ctr

    @property
    def scaling(self):
        """Entrywise scaling (non-dimensionalization) directive.
        * None: no scaling.
        * 'standard': standardize to zero mean and unit standard deviation.
        * 'minmax': minmax scaling to [0,1].
        * 'minmaxsym': minmax scaling to [-1,1].
        * 'maxabs': maximum absolute scaling to [-1,1] (no shift).
        * 'maxabssym': maximum absolute scaling to [-1,1] (mean shift).
        """
        return self.__scaling

    @scaling.setter
    def scaling(self, scl):
        """Set the scaling strategy, resetting the transformation."""
        if scl is None:
            self._clear()
            self.__scaling = scl
            return
        if not isinstance(scl, str):
            raise TypeError("'scaling' must be of type 'str'")
        if scl not in self._VALID_SCALINGS:
            opts = ", ".join([f"'{v}'" for v in self._VALID_SCALINGS])
            raise ValueError(f"invalid scaling '{scl}'; "
                             f"valid options are {opts}")
        self._clear()
        self.__scaling = scl

    @property
    def verbose(self):
        """If True, print information about upon learning a transformation."""
        return self.__verbose

    @verbose.setter
    def verbose(self, vbs):
        self.__verbose = bool(vbs)

    def __eq__(self, other):
        """Test two SnapshotTransformers for equality."""
        if not isinstance(other, self.__class__):
            return False
        for attr in ("center", "scaling"):
            if getattr(self, attr) != getattr(other, attr):
                return False
        if self.center and hasattr(self, "mean_"):
            if not hasattr(other, "mean_"):
                return False
            if not np.all(self.mean_ == other.mean_):
                return False
        if self.scaling and hasattr(self, "scale_"):
            for attr in ("scale_", "shift_"):
                if not hasattr(other, attr):
                    return False
                if getattr(self, attr) != getattr(other, attr):
                    return False
        return True

    # Printing ----------------------------------------------------------------
    def __str__(self):
        """String representation: scaling type + centering bool."""
        out = ["Snapshot transformer"]
        if self.center:
            out.append("with mean-snapshot centering")
            if self.scaling:
                out.append(f"and '{self.scaling}' scaling")
        elif self.scaling:
            out.append(f"with '{self.scaling}' scaling")
        if not self._is_trained():
            out.append("(call fit_transform() to train)")
        return ' '.join(out)

    @staticmethod
    def _statistics_report(X):
        """Return a string of basis statistics about a data set."""
        return " | ".join([f"{f(X):>10.3e}"
                           for f in (np.min, np.mean, np.max, np.std)])

    # Persistence -------------------------------------------------------------
    def save(self, savefile, overwrite=False):
        """Save the current transformer to an HDF5 file.

        Parameters
        ----------
        savefile : str
            Path of the file to save the transformer in.
        overwrite : bool
            If True, overwrite the file if it already exists. If False
            (default), raise a FileExistsError if the file already exists.
        """
        # Ensure the file is saved in HDF5 format.
        if not savefile.endswith(".h5"):
            savefile += ".h5"

        # Prevent overwriting and existing file on accident.
        if os.path.isfile(savefile) and not overwrite:
            raise FileExistsError(f"{savefile} (use overwrite=True to ignore)")

        with h5py.File(savefile, 'w') as hf:
            # Store transformation hyperparameter metadata.
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["center"] = self.center
            meta.attrs["scaling"] = self.scaling if self.scaling else False
            meta.attrs["verbose"] = self.verbose

            # Store learned transformation parameters.
            if self.center and hasattr(self, "mean_"):
                hf.create_dataset("transformation/mean_", data=self.mean_)
            if self.scaling and hasattr(self, "scale_"):
                hf.create_dataset("transformation/scale_", data=[self.scale_])
                hf.create_dataset("transformation/shift_", data=[self.shift_])

    @classmethod
    def load(cls, loadfile):
        """Load a SnapshotTransformer from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            Path to the file where the transformer was stored (via save()).

        Returns
        -------
        SnapshotTransformer
        """
        with h5py.File(loadfile, 'r') as hf:
            # Load transformation hyperparameters.
            if "meta" not in hf:
                raise ValueError("invalid save format (meta/ not found)")
            scl = hf["meta"].attrs["scaling"]
            transformer = cls(center=hf["meta"].attrs["center"],
                              scaling=scl if scl else None,
                              verbose=hf["meta"].attrs["verbose"])

            # Load learned transformation parameters.
            if transformer.center and "transformation/mean_" in hf:
                transformer.mean_ = hf["transformation/mean_"][:]
            if transformer.scaling and "transformation/scale_" in hf:
                transformer.scale_ = hf["transformation/scale_"][0]
                transformer.shift_ = hf["transformation/shift_"][0]

            return transformer

    # Main routines -----------------------------------------------------------
    def _is_trained(self):
        """Return True if transform() and inverse_transform() are ready."""
        if self.center and not hasattr(self, "mean_"):
            return False
        if self.scaling and any(not hasattr(self, attr)
                                for attr in ("scale_", "shift_")):
            return False
        return True

    def fit_transform(self, X, inplace=False):
        """Learn and apply the transformation.

        Parameters
        ----------
        X : (n,k) ndarray
            Matrix of k snapshots. Each column is a snapshot of dimension n.
        inplace : bool
            If True, overwrite the input data during transformation.
            If False, create a copy of the data to transform.

        Returns
        -------
        X'': (n,k) ndarray
            Matrix of k transformed n-dimensional snapshots.
        """
        Y = X if inplace else X.copy()

        # Record statistics of the training data.
        if self.verbose:
            report = ["No transformation learned"]
            report.append(self._table_header)
            report.append(f"X   | {self._statistics_report(X)}")

        # Center the snapshots by the mean training snapshot.
        if self.center:
            self.mean_ = np.mean(Y, axis=1)
            Y -= self.mean_.reshape((-1,1))

            if self.verbose:
                report[0] = "Learned mean centering X -> X'"
                report.append(f"X'  | {self._statistics_report(Y)}")

        # Scale (non-dimensionalize) the centered snapshot entries.
        if self.scaling:
            # Standard: X' = (X - µ)/σ
            if self.scaling == "standard":
                µ = np.mean(Y)
                σ = np.std(Y)
                self.scale_ = 1/σ
                self.shift_ = -µ*self.scale_

            # Min-max: X' = (X - min(X))/(max(X) - min(X))
            elif self.scaling == "minmax":
                Ymin = np.min(Y)
                Ymax = np.max(Y)
                self.scale_ = 1/(Ymax - Ymin)
                self.shift_ = -Ymin*self.scale_

            # Symmetric min-max: X' = (X - min(X))*2/(max(X) - min(X)) - 1
            elif self.scaling == "minmaxsym":
                Ymin = np.min(Y)
                Ymax = np.max(Y)
                self.scale_ = 2/(Ymax - Ymin)
                self.shift_ = -Ymin*self.scale_ - 1

            # MaxAbs: X' = X / max(abs(X))
            elif self.scaling == "maxabs":
                self.scale_ = 1/np.max(np.abs(Y))
                self.shift_ = 0

            # maxabssym: X' = (X - mean(X)) / max(abs(X - mean(X)))
            elif self.scaling == "maxabssym":
                µ = np.mean(Y)
                Y -= µ
                self.scale_ = 1/np.max(np.abs(Y))
                self.shift_ = -µ*self.scale_
                Y += µ

            else:                               # pragma nocover
                raise RuntimeError(f"invalid scaling '{self.scaling}'")

            Y *= self.scale_
            Y += self.shift_

            if self.verbose:
                if self.center:
                    report[0] += f" and {self.scaling} scaling X' -> X''"
                else:
                    report[0] = f"Learned {self.scaling} scaling X -> X''"
                report.append(f"X'' | {self._statistics_report(Y)}")

        if self.verbose:
            print('\n'.join(report) + '\n')

        return Y

    def transform(self, X, inplace=False):
        """Apply the learned transformation.

        Parameters
        ----------
        X : (n,k) ndarray
            Matrix of k snapshots. Each column is a snapshot of dimension n.
        inplace : bool
            If True, overwrite the input data during transformation.
            If False, create a copy of the data to transform.

        Returns
        -------
        X'': (n,k) ndarray
            Matrix of k transformed n-dimensional snapshots.
        """
        if not self._is_trained():
            raise AttributeError("transformer not trained "
                                 "(call fit_transform())")

        Y = X if inplace else X.copy()

        # Center the snapshots by the mean training snapshot.
        if self.center is True:
            Y -= self.mean_.reshape((-1,1))

        # Scale (non-dimensionalize) the centered snapshot entries.
        if self.scaling is not None:
            Y *= self.scale_
            Y += self.shift_

        return Y

    def inverse_transform(self, X, inplace=False):
        """Apply the inverse of the learned transformation.

        Parameters
        ----------
        X : (n,k) ndarray
            Matrix of k transformed n-dimensional snapshots.
        inplace : bool
            If True, overwrite the input data during inverse transformation.
            If False, create a copy of the data to untransform.

        Returns
        -------
        X'': (n,k) ndarray
            Matrix of k untransformed n-dimensional snapshots.
        """
        if not self._is_trained():
            raise AttributeError("transformer not trained "
                                 "(call fit_transform())")

        Y = X if inplace else X.copy()

        # Unscale (re-dimensionalize) the data.
        if self.scaling:
            Y -= self.shift_
            Y /= self.scale_

        # Uncenter the unscaled snapshots.
        if self.center:
            Y += self.mean_.reshape((-1,1))

        return Y


class SnapshotTransformerMulti:
    """Transformer for multi-variate snapshots.

    Groups multiple SnapshotTransformers for the centering and/or scaling
    (in that order) of individual variables.

    Parameters
    ----------
    num_variables : int
        Number of variables represented in a single snapshot (number of
        individual transformations to learn). The dimension `n` of the
        snapshots must be evenly divisible by num_variables; for example,
        num_variables=3 means the first n entries of a snapshot correspond to
        the first variable, and the next n entries correspond to the second
        variable, and the last n entries correspond to the third variable.
    center : bool OR list of num_variables bools
        If True, shift the snapshots by the mean training snapshot.
        If a list, center[i] is the centering directive for the ith variable.
    scaling : str, None, OR list of length num_variables
        If given, scale (non-dimensionalize) the centered snapshot entries.
        If a list, scaling[i] is the scaling directive for the ith variable.
        * 'standard': standardize to zero mean and unit standard deviation.
        * 'minmax': minmax scaling to [0,1].
        * 'minmaxsym': minmax scaling to [-1,1].
        * 'maxabs': maximum absolute scaling to [-1,1] (no shift).
        * 'maxabssym': maximum absolute scaling to [-1,1] (mean shift).
    variable_names : list of num_variables strings
        Names for each of the `num_variables` variables.
        Defaults to 'x1', 'x2', ....
    verbose : bool
        If True, print information upon learning a transformation.

    Attributes
    ----------
    transfomers : list of num_variables SnapshotTransformers
        Transformers for each snapshot variable.
    n_ : int
        Dimension of individual variables.

    Notes
    -----
    See SnapshotTransformer for details on available transformations.

    Examples
    --------
    # Center first and third variables and minmax scale the second variable.
    >>> stm = SnapshotTransformerMulti(3, center=(True, False, True),
    ...                                   scaling=(None, "minmax", None))

    # Center 6 variables and scale the final variable with a standard scaling.
    >>> stm = SnapshotTransformerMulti(3, center=True,
    ...                                   scaling=(None, None, None,
    ...                                            None, None, "standard"))
    # OR
    >>> stm = SnapshotTransformerMulti(3, center=True, scaling=None)
    >>> stm[-1].scaling = "standard"
    """
    def __init__(self, num_variables, center=False, scaling=None,
                 variable_names=None, verbose=False):
        """Interpret hyperparameters and initialize transformers."""
        def _process_arg(attr, name, dtype):
            """Validation for centering and scaling directives."""
            if isinstance(attr, dtype):
                attr = (attr,)*num_variables
            if len(attr) != num_variables:
                raise ValueError(f"len({name}) = {len(attr)} "
                                 f"!= {num_variables} = num_variables")
            return attr

        centers = _process_arg(center, "center", bool)
        scalings = _process_arg(scaling, "scaling", (type(None), str))

        # Initialize transformers.
        self.transformers = [SnapshotTransformer(center=ctr, scaling=scl)
                             for ctr, scl in zip(centers, scalings)]
        self.variable_names = variable_names
        self.verbose = verbose

    # Properties --------------------------------------------------------------
    @property
    def num_variables(self):
        """Number of variables represented in a single snapshot."""
        return len(self.transformers)

    @property
    def center(self):
        """Snapshot mean-centering directive."""
        return tuple(st.center for st in self.transformers)

    @property
    def scaling(self):
        """Entrywise scaling (non-dimensionalization) directive.
        * None: no scaling.
        * 'standard': standardize to zero mean and unit standard deviation.
        * 'minmax': minmax scaling to [0,1].
        * 'minmaxsym': minmax scaling to [-1,1].
        * 'maxabs': maximum absolute scaling to [-1,1] (no shift).
        * 'maxabssym': maximum absolute scaling to [-1,1] (mean shift).
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
            raise TypeError("variable_names must be list of"
                            f" length {self.num_variables}")
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
        return np.concatenate([(st.mean_ if st.center else np.zeros(self.n_))
                               for st in self.transformers])

    def __len__(self):
        """Length: number of SnapshotTransformers (number of variables)."""
        return self.num_variables

    def __getitem__(self, key):
        """Get the transformer for variable i."""
        return self.transformers[key]

    def __setitem__(self, key, obj):
        """Set the transformer for variable i."""
        if not isinstance(obj, SnapshotTransformer):
            raise TypeError("assignment object must be SnapshotTransformer")
        self.transformers[key] = obj

    def __eq__(self, other):
        """Test two SnapshotTransformerMulti objects for equality."""
        if not isinstance(other, self.__class__):
            return False
        if self.num_variables != other.num_variables:
            return False
        return all(t1 == t2 for t1,t2 in zip(self.transformers,
                                             other.transformers))

    def __str__(self):
        """String representation: centering and scaling directives."""
        out = ["Multi-variate snapshot transformer"]
        namelength = max(len(name) for name in self.variable_names)
        for name, st in zip(self.variable_names, self.transformers):
            out.append(f"* {{:>{namelength}}} | {st}".format(name))
        return '\n'.join(out)

    # Persistence -------------------------------------------------------------
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
        # Ensure the file is saved in HDF5 format.
        if not savefile.endswith(".h5"):
            savefile += ".h5"

        # Prevent overwriting and existing file on accident.
        if os.path.isfile(savefile) and not overwrite:
            raise FileExistsError(f"{savefile} (use overwrite=True to ignore)")

        with h5py.File(savefile, 'w') as hf:
            # Metadata
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["num_variables"] = self.num_variables
            meta.attrs["verbose"] = self.verbose

            for i in range(self.num_variables):
                group = hf.create_group(f"variable{i+1}")

                # Store transformation hyperparameter metadata.
                meta = group.create_dataset("meta", shape=(0,))
                st = self.transformers[i]
                ctr, scl = st.center, st.scaling
                if scl is None:
                    scl = False
                meta.attrs["center"] = ctr
                meta.attrs["scaling"] = scl

                # Store learned transformation parameters.
                if ctr and hasattr(st, "mean_"):
                    group.create_dataset("transformation/mean_", data=st.mean_)
                if scl and hasattr(st, "scale_"):
                    group.create_dataset("transformation/scale_",
                                         data=[st.scale_])
                    group.create_dataset("transformation/shift_",
                                         data=[st.shift_])

    @classmethod
    def load(cls, loadfile):
        """Load a SnapshotTransformer from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            Path to the file where the transformer was stored (via save()).

        Returns
        -------
        SnapshotTransformerMulti
        """
        with h5py.File(loadfile, 'r') as hf:
            # Load transformation hyperparameters.
            if "meta" not in hf:
                raise ValueError("invalid save format (meta/ not found)")
            num_variables = hf["meta"].attrs["num_variables"]
            verbose = hf["meta"].attrs["verbose"]
            stm = cls(num_variables, verbose=verbose)

            # Modify each component transformer.
            for i in range(num_variables):
                group = hf[f"variable{i+1}"]
                ctr = group["meta"].attrs["center"]
                scl = group["meta"].attrs["scaling"]
                if not scl:
                    scl = None
                stm[i].center = ctr
                stm[i].scaling = scl

                # Load learned transformation parameters.
                if ctr and "transformation/mean_" in group:
                    stm[i].mean_ = group["transformation/mean_"][:]
                if scl and "transformation/scale_" in group:
                    stm[i].scale_ = group["transformation/scale_"][0]
                    stm[i].shift_ = group["transformation/shift_"][0]

            return stm

    # Main routines -----------------------------------------------------------
    def _check_shape(self, X):
        """Verify the shape of the snapshot set X."""
        if X.shape[0] != self.num_variables * self.n_:
            raise ValueError("snapshot set must have num_variables * n "
                             f"= {self.num_variables} * {self.n_} "
                             f"= {self.num_variables * self.n_} rows "
                             f"(got {X.shape[0]})")

    def _is_trained(self):
        """Return True if transform() and inverse_transform() are ready."""
        return all(st._is_trained() for st in self.transformers)

    def _apply(self, method, X, inplace):
        """Apply a method of each transformer to the corresponding chunk of X.
        """
        Ys = []
        for st, var, name in zip(self.transformers,
                                 np.split(X, self.num_variables, axis=0),
                                 self.variable_names):
            if method is SnapshotTransformer.fit_transform and self.verbose:
                print(f"{name}:")
            Ys.append(method(st, var, inplace=inplace))
        return X if inplace else np.row_stack(Ys)

    def fit_transform(self, X, inplace=False):
        """Learn and apply the transformation.

        Parameters
        ----------
        X : (n,k) ndarray
            Matrix of k snapshots. Each column is a snapshot of dimension n;
            this dimension must be evenly divisible by `num_variables`.
        inplace : bool
            If True, overwrite the input data during transformation.
            If False, create a copy of the data to transform.

        Returns
        -------
        X'': (n,k) ndarray
            Matrix of k transformed n-dimensional snapshots.
        """
        Y = self._apply(SnapshotTransformer.fit_transform, X, inplace)
        self.n_ = X.shape[0] // self.num_variables
        return Y

    def transform(self, X, inplace=False):
        """Apply the learned transformation.

        Parameters
        ----------
        X : (n,k) ndarray
            Matrix of k snapshots. Each column is a snapshot of dimension n;
            this dimension must be evenly divisible by `num_variables`.
        inplace : bool
            If True, overwrite the input data during transformation.
            If False, create a copy of the data to transform.

        Returns
        -------
        X'': (n,k) ndarray
            Matrix of k transformed n-dimensional snapshots.
        """
        if not self._is_trained():
            raise AttributeError("transformer not trained "
                                 "(call fit_transform())")
        self._check_shape(X)
        return self._apply(SnapshotTransformer.transform, X, inplace)

    def inverse_transform(self, X, inplace=False):
        """Apply the inverse of the learned transformation.

        Parameters
        ----------
        X : (n,k) ndarray
            Matrix of k transformed n-dimensional snapshots.
        inplace : bool
            If True, overwrite the input data during inverse transformation.
            If False, create a copy of the data to untransform.

        Returns
        -------
        X'': (n,k) ndarray
            Matrix of k untransformed n-dimensional snapshots.
        """
        if not self._is_trained():
            raise AttributeError("transformer not trained "
                                 "(call fit_transform())")
        self._check_shape(X)
        return self._apply(SnapshotTransformer.inverse_transform, X, inplace)
