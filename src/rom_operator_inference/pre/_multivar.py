# pre/_multivar.py
"""Private mixin class for transfomers and basis with multivariate states."""

import numpy as np


class _MultivarMixin:
    """Private mixin class for transfomers and basis with multivariate states.

    Parameters
    ----------
    num_variables : int
        Number of variables represented in a single snapshot (number of
        individual transformations to learn). The dimension `n` of the
        snapshots must be evenly divisible by num_variables; for example,
        num_variables=3 means the first n entries of a snapshot correspond to
        the first variable, and the next n entries correspond to the second
        variable, and the last n entries correspond to the third variable.
    variable_names : list of num_variables strings, optional
        Names for each of the `num_variables` variables.
        Defaults to "variable 1", "variable 2", ....

    Attributes
    ----------
    n : int
        Total dimension of the snapshots (all variables).
    ni : int
        Dimension of individual variables, i.e., ni = n / num_variables.

    Notes
    -----
    Child classes must set `n` in their fit() methods.
    """
    def __init__(self, num_variables, variable_names=None):
        """Store variable information."""
        if not np.isscalar(num_variables) or num_variables < 1:
            raise ValueError("num_variables must be a positive integer")
        self.__num_variables = num_variables
        self.variable_names = variable_names
        self.__n = None

    # Properties --------------------------------------------------------------
    @property
    def num_variables(self):
        """Number of variables represented in a single snapshot."""
        return self.__num_variables

    @property
    def variable_names(self):
        """Names for each of the `num_variables` variables."""
        return self.__variable_names

    @variable_names.setter
    def variable_names(self, names):
        if names is None:
            names = [f"variable {i+1}" for i in range(self.num_variables)]
        if not isinstance(names, list) or len(names) != self.num_variables:
            raise TypeError("variable_names must be a list of"
                            f" length {self.num_variables}")
        self.__variable_names = names

    @property
    def n(self):
        """Total dimension of the snapshots (all variables)."""
        return self.__n

    @n.setter
    def n(self, nn):
        """Set the total and individual variable dimensions."""
        if nn % self.num_variables != 0:
            raise ValueError("n must be evenly divisible by num_variables")
        self.__n = nn

    @property
    def ni(self):
        """Dimension of individual variables, i.e., ni = n / num_variables."""
        return None if self.n is None else self.n // self.num_variables

    # Convenience methods -----------------------------------------------------
    def get_varslice(self, var):
        """Get the indices (as a slice) where the specified variable resides.

        Parameters
        ----------
        var : int or str
            Index or name of the variable to extract.

        Returns
        -------
        s : slice
            Slice object for accessing the specified variable, i.e.,
            variable = state[s] for a single snapshot or
            variable = states[:, s] for a collection of snapshots.
        """
        if var in self.variable_names:
            var = self.variable_names.index(var)
        return slice(var*self.ni, (var + 1)*self.ni)

    def get_var(self, var, states):
        """Extract the ith variable from the states.

        Parameters
        ----------
        var : int or str
            Index or name of the variable to extract.
        states : (n, ...) ndarray

        Returns
        -------
        states_var : ndarray, shape (n, num_states)
        """
        self._check_shape(states)
        return states[..., self.get_varslice(var)]

    def _check_shape(self, Q):
        """Verify the shape of the snapshot set Q."""
        if Q.shape[0] != self.n:
            raise ValueError(f"states.shape[0] = {Q.shape[0]:d} "
                             f"!= {self.num_variables} * {self.ni} "
                             "= num_variables * n_i")
