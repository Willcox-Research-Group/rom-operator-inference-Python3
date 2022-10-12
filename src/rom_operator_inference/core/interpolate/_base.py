# core/interpolate/_base.py
"""Base class for parametric reduced-order models where the parametric
dependencies of operators are handled with elementwise interpolation, i.e,
    A(µ)[i,j] = Interpolator([µ1, µ2, ...], [A1[i,j], A2[i,j], ...])(µ).
where µ1, µ2, ... are parameter values and A1, A2, ... are the corresponding
operator matrices, e.g., A1 = A(µ1).

Relevant operator classes are defined in core.operators._interpolate.
"""

__all__ = []

import numpy as np
import scipy.interpolate

from ... import pre
from ...utils import hdf5_savehandle, hdf5_loadhandle
from .._base import _BaseParametricROM
from ..nonparametric._base import _NonparametricOpInfROM
from .. import operators


class _InterpolatedOpInfROM(_BaseParametricROM):
    """Base class for parametric reduced-order models where the parametric
    dependence of operators are handled with elementwise interpolation, i.e,
        A(µ)[i,j] = Interpolator([µ1, µ2, ...], [A1[i,j], A2[i,j], ...])(µ).
    where µ1, µ2, ... are parameter values and A1, A2, ... are the
    corresponding operator matrices, e.g., A1 = A(µ1). That is, individual
    operators is learned for each training parameter, and those operators are
    interpolated elementwise to construct operators for new parameter values.
    """
    # Must be specified by child classes.
    _ModelFitClass = NotImplemented

    def __init__(self, modelform, InterpolatorClass="auto"):
        """Set the model form (ROM structure) and interpolator type.

        Parameters
        ----------
        modelform : str
            Structure of the reduced-order model. See the class docstring.
        InterpolatorClass : type or str
            Class for elementwise operator interpolation. Must obey the syntax
            >>> interpolator = InterpolatorClass(data_points, data_values)
            >>> interpolator_evaluation = interpolator(new_data_point)
            Convenience options:
            * "cubicspline": scipy.interpolate.CubicSpline (p = 1)
            * "linear": scipy.interpolate.LinearNDInterpolator (p > 1)
            * "auto" (default): choose based on the parameter dimension.
        """
        _BaseParametricROM.__init__(self, modelform)

        # Validate the _ModelFitClass.
        if not issubclass(self._ModelFitClass, _NonparametricOpInfROM):
            raise RuntimeError("invalid _ModelFitClass "
                               f"'{self._ModelFitClass.__name__}'")

        # Save the interpolator class.
        self.__autoIC = (InterpolatorClass == "auto")
        if InterpolatorClass == "cubicspline":
            self.InterpolatorClass = scipy.interpolate.CubicSpline
        elif InterpolatorClass == "linear":
            self.InterpolatorClass = scipy.interpolate.LinearNDInterpolator
        elif not isinstance(InterpolatorClass, str):
            self.InterpolatorClass = InterpolatorClass
        elif not self.__autoIC:
            raise ValueError("invalid InterpolatorClass "
                             f"'{InterpolatorClass}'")

    # Properties --------------------------------------------------------------
    @property
    def s(self):
        """Number of training parameter samples, i.e., the number of data
        points in the interpolation scheme.
        """
        return self.__s

    def __len__(self):
        """Number of training parameter samples, i.e., the number of ROMS to
        interpolate between.
        """
        return self.s

    # Fitting -----------------------------------------------------------------
    def _check_parameters(self, parameters):
        """Extract the parameter dimension and ensure it is consistent
        across parameter samples.
        """
        shape = np.shape(parameters[0])
        if any(np.shape(param) != shape for param in parameters):
            raise ValueError("parameter dimension inconsistent across samples")
        self.__s = len(parameters)
        self._set_parameter_dimension(parameters)

        # If required, select the interpolator based on parameter dimension.
        if self.__autoIC:
            if self.p == 1:
                self.InterpolatorClass = scipy.interpolate.CubicSpline
            else:
                self.InterpolatorClass = scipy.interpolate.LinearNDInterpolator

    def _split_operator_dict(self, known_operators):
        """Unzip the known operators dictionary into separate dictionaries,
        one for each parameter sample. For example:
        {                                   [
            "A": [A1, A2, A3],                  {"A": A1, "H": H1, "B": B},
            "H": [H1, None, H3],    -->         {"A": A2, "B": B},
            "B": B                              {"A": A3, "H": H3, "B": B}
        }                                   ]
        Also check that the right number of operators is specified.

        Parameters
        ----------
        known_operators : dict or None
            Maps modelform keys to list of s operators.

        Returns
        -------
        known_operators_list : list(dict) or None
            List of s dictionarys mapping modelform keys to single operators.
        """
        if known_operators is None:
            return [None] * self.s
        if not isinstance(known_operators, dict):
            raise TypeError("known_operators must be a dictionary")

        # Check that each dictionary value is a list.
        for key in known_operators.keys():
            val = known_operators[key]
            if isinstance(val, np.ndarray) and val.shape[0] != self.s:
                # Special case: single operator matrix given, not a list.
                # TODO: if r == s this could be misinterpreted.
                known_operators[key] = [val] * self.s
            elif not isinstance(val, list):
                raise TypeError("known_operators must be a dictionary mapping "
                                "a string to a list of ndarrays")

        # Check length of each list in the dictionary.
        if not all(len(val) == self.s for val in known_operators.values()):
            raise ValueError("known_operators dictionary must map a modelform "
                             f"key to a list of s = {self.s} ndarrays")

        # "Unzip" the dictionary of lists to a list of dictionaries.
        return [
            {key: known_operators[key][i]
             for key in known_operators.keys()
             if known_operators[key][i] is not None}
            for i in range(self.s)
        ]

    def _check_number_of_training_datasets(self, datasets):
        """Ensure that each data set has the same number of entries as
        the number of parameter samples.

        Parameters
        ----------
        datasets: list of (ndarray, str) tuples
            Datasets paired with labels, e.g., [(Q, "states"), (dQ, "ddts")].
        """
        for data, label in datasets:
            if len(data) != self.s:
                raise ValueError(f"len({label}) = {len(data)} "
                                 f"!= {self.s} = len(parameters)")

    def _process_fit_arguments(self, basis, parameters, states, lhss, inputs,
                               regularizers, known_operators):
        """Do sanity checks, extract dimensions, and check data sizes."""
        # Intialize reset.
        self._clear()           # Clear all data (basis and operators).
        self.basis = basis      # Store basis and (hence) reduced dimension.

        # Validate parameters and set parameter dimension / num training sets.
        self._check_parameters(parameters)

        # Replace any None arguments with [None, None, ..., None] (s times).
        if states is None:
            states = [None] * self.s
        if lhss is None:
            lhss = [None] * self.s
        if inputs is None:
            inputs = [None] * self.s

        # Interpret regularizers argument.
        _reg = regularizers
        if _reg is None or np.isscalar(_reg) or len(_reg) != self.s:
            regularizers = [regularizers] * self.s

        # Separate known operators into one dictionary per parameter sample.
        if isinstance(known_operators, list):
            knownops_list = known_operators
        else:
            knownops_list = self._split_operator_dict(known_operators)

        # Check that the number of training sets is consistent.
        self._check_number_of_training_datasets([
            (parameters, "parameters"),
            (states, "states"),
            (lhss, self._LHS_ARGNAME),
            (inputs, "inputs"),
            (regularizers, "regularizers"),
            (knownops_list, "known_operators"),
        ])

        return states, lhss, inputs, regularizers, knownops_list

    def _interpolate_roms(self, parameters, roms):
        """Interpolate operators from a collection of non-parametric ROMs.

        Parameters
        ----------
        parameters : (s, p) ndarray or (s,) ndarray
            Parameter values corresponding to the training data, either
            s p-dimensional vectors or s scalars (parameter dimension p = 1).
        roms : list of s ROM objects (of a class derived from _BaseROM)
            Trained non-parametric reduced-order models.
        """
        # Ensure that all ROMs are trained.
        for rom in roms:
            if not isinstance(rom, self._ModelFitClass):
                raise TypeError("expected roms of type "
                                f"{self._ModelFitClass.__name__}")
            rom._check_is_trained()

        # Extract dimensions from the ROMs and check for consistency.
        if self.basis is None:
            self.r = roms[0].r
        if 'B' in self.modelform:
            self.m = roms[0].m
        for rom in roms:
            if rom.modelform != self.modelform:
                raise ValueError("ROMs to interpolate must have "
                                 f"modelform='{self.modelform}'")
            if rom.r != self.r:
                raise ValueError("ROMs to interpolate must have equal "
                                 "dimensions (inconsistent r)")
            if rom.m != self.m:
                raise ValueError("ROMs to interpolate must have equal "
                                 "dimensions (inconsistent m)")

        # Extract the operators from the individual ROMs.
        for key in self.modelform:
            attr = f"{key}_"
            ops = [getattr(rom, attr).entries for rom in roms]
            if all(np.all(ops[0] == op) for op in ops):
                # This operator does not depend on the parameters.
                OperatorClass = operators.nonparametric_operators[key]
                setattr(self, attr, OperatorClass(ops[0]))
            else:
                # This operator varies with the parameters (so interpolate).
                OperatorClass = operators.interpolated_operators[key]
                setattr(self, attr, OperatorClass(parameters, ops,
                                                  self.InterpolatorClass))

    def fit(self, basis, parameters, states, lhss, inputs=None,
            regularizers=0, known_operators=None):
        """Learn the reduced-order model operators from data.

        Parameters
        ----------
        basis : (n, r) ndarray or None
            Basis for the linear reduced space (e.g., POD basis matrix).
            If None, statess and lhss are assumed to already be projected.
        parameters : (s, p) ndarray or (s,) ndarray
            Parameter values corresponding to the training data, either
            s p-dimensional vectors or s scalars (parameter dimension p = 1).
        states : list of s (n, k) or (r, k) ndarrays
            State snapshots for each parameter value: `states[i]` corresponds
            to `parameters[i]` and contains column-wise state data, i.e.,
            `states[i][:, j]` is a single snapshot.
            Data may be either full order (n rows) or reduced order (r rows).
        lhss : list of s (n, k) or (r, k) ndarrays
            Left-hand side data for ROM training corresponding to each
            parameter value: `lhss[i]` corresponds to `parameters[i]` and
            contains column-wise left-hand side data, i.e., `lhss[i][:, j]`
            corresponds to the state snapshot `states[i][:, j]`.
            Data may be either full order (n rows) or reduced order (r rows).
            * Steady: forcing function.
            * Discrete: column-wise next iteration
            * Continuous: time derivative of the state
        inputs : list of s (m, k) or (k,) ndarrays or None
            Inputs for ROM training corresponding each parameter value:
            `inputs[i]` corresponds to `parameters[i]` and contains
            column-wise input data, i.e., `inputs[i][:, j]` corresponds to the
            state snapshot `states[i][:, j]`.
            If m = 1 (scalar input), then each `inputs[i]` may be a one-
            dimensional array.
            This argument is required if 'B' is in `modelform` but must be
            None if 'B' is not in `modelform`.
        regularizers : list of s (float >= 0, (d, d) ndarray, or r of these)
            Tikhonov regularization factor(s) for each parameter value:
            `regularizers[i]` is the regularization factor for the regression
            using data corresponding to `parameters[i]`. See lstsq.solve().
            Here, d is the number of unknowns in each decoupled least-squares
            problem, e.g., d = r + m when `modelform`="AB".
        known_operators : dict or None
            Dictionary of known full-order operators at each parameter value.
            Corresponding reduced-order operators are computed directly
            through projection; remaining operators are inferred from data.
            Keys must match the modelform; values are a list of s ndarrays:
            * 'c': (n,) constant term c.
            * 'A': (n, n) linear state matrix A.
            * 'H': (n, n**2) quadratic state matrix H.
            * 'G': (n, n**3) cubic state matrix G.
            * 'B': (n, m) input matrix B.
            If operators are known for some parameter values but not others,
            use None whenever the operator must be inferred, e.g., for
            parameters = [µ1, µ2, µ3, µ4, µ5], if A1, A3, and A4 are known
            linear state operators at µ1, µ3, and µ4, respectively, set
            known_operators = {'A': [A1, None, A3, A4, None]}.
            For known operators (e.g., A) that do not depend on the parameters,
            known_operators = {'A': [A, A, A, A, A]} and
            known_operators = {'A': A} are equivalent.

        Returns
        -------
        self
        """
        args = self._process_fit_arguments(basis, parameters,
                                           states, lhss, inputs,
                                           regularizers, known_operators)
        states, lhss, inputs, regularizers, knownops_list = args

        # Distribute training data to individual OpInf problems.
        nonparametric_roms = [
            self._ModelFitClass(self.modelform).fit(
                self.basis,
                states[i], lhss[i], inputs[i],
                regularizers[i], knownops_list[i]
            ) for i in range(self.s)
        ]
        # TODO: split into _[construct/evaluate]_solver() paradigm?
        # If so, move dimension extraction to construct_solver().

        # Construct interpolated operators.
        self._interpolate_roms(parameters, nonparametric_roms)

        return self

    def set_interpolator(self, InterpolatorClass):
        """Construct the interpolators for the operator entries.
        Use this method to change the interpolator after calling fit().

        Parameters
        ----------
        InterpolatorClass : type
            Class for the elementwise interpolation. Must obey the syntax
            >>> interpolator = InterpolatorClass(data_points, data_values)
            >>> interpolator_evaluation = interpolator(new_data_point)
            This is usually a class from scipy.interpolate.
        """
        for key in self.modelform:
            op = getattr(self, f"{key}_")
            if operators.is_parametric_operator(op):
                op.set_interpolator(InterpolatorClass)

    # Model persistence -------------------------------------------------------
    def save(self, savefile, save_basis=True, overwrite=False):
        """Serialize the ROM, saving it in HDF5 format.
        The model can then be loaded with the load() class method.

        Parameters
        ----------
        savefile : str
            File to save to, with extension '.h5' (HDF5).
        savebasis : bool
            If True, save the basis as well as the reduced operators.
            If False, only save reduced operators.
        overwrite : bool
            If True and the specified file already exists, overwrite the file.
            If False and the specified file already exists, raise an error.
        """
        self._check_is_trained()

        with hdf5_savehandle(savefile, overwrite=overwrite) as hf:
            # Store ROM modelform.
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["modelform"] = self.modelform

            # Store basis (optionally) if it exists.
            if (self.basis is not None) and save_basis:
                meta.attrs["BasisClass"] = self.basis.__class__.__name__
                self.basis.save(hf.create_group("basis"))

            # Store reduced operators.
            for key, op in zip(self.modelform, self):
                if "parameters" not in hf:
                    hf.create_dataset("parameters", data=op.parameters)
                hf.create_dataset(f"operators/{key}_", data=op.matrices)

    @classmethod
    def load(cls, loadfile, InterpolatorClass):
        """Load a serialized ROM from an HDF5 file, created previously from
        a ROM object's save() method.

        Parameters
        ----------
        loadfile : str
            File to load from, which should end in '.h5'.
        InterpolatorClass : type or str
            Class for elementwise operator interpolation. Must obey the syntax
            >>> interpolator = InterpolatorClass(data_points, data_values)
            >>> interpolator_evaluation = interpolator(new_data_point)
            Convenience options:
            * "cubicspline": scipy.interpolate.CubicSpline (p = 1)
            * "linear": scipy.interpolate.LinearNDInterpolator (p > 1)
            * "auto" (default): choose based on the parameter dimension.

        Returns
        -------
        model : _NonparametricOpInfROM
            Trained reduced-order model.
        """
        with hdf5_loadhandle(loadfile) as hf:
            if "meta" not in hf:
                raise ValueError("invalid save format (meta/ not found)")
            if "operators" not in hf:
                raise ValueError("invalid save format (operators/ not found)")
            if "parameters" not in hf:
                raise ValueError("invalid save format (parameters/ not found)")

            # Load metadata.
            modelform = hf["meta"].attrs["modelform"]
            basis = None

            # Load basis if present.
            if "basis" in hf:
                BasisClassName = hf["meta"].attrs["BasisClass"]
                basis = getattr(pre, BasisClassName).load(hf["basis"])

            # Load operators.
            parameters = hf["parameters"][:]
            ops = {}
            for key in modelform:
                attr = f"{key}_"
                op = hf[f"operators/{attr}"][:]
                if op.ndim == (1 if key == "c" else 2):
                    # This is a nonparametric operator.
                    OpClass = operators.nonparametric_operators[key]
                    ops[attr] = OpClass(op)
                else:
                    # This is a parametric operator.
                    OpClass = operators.interpolated_operators[key]
                    ops[attr] = OpClass(parameters, op, InterpolatorClass)

        # Construct the model.
        return cls(modelform, InterpolatorClass)._set_operators(basis, **ops)
