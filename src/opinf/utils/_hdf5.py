# utils/_hdf5.py
"""Utilities for HDF5 file interaction."""

__all__ = [
    "hdf5_savehandle",
    "hdf5_loadhandle",
    "save_sparray",
    "load_sparray",
]

import os
import h5py
import warnings
import scipy.sparse as sparse

from .. import errors


# File handle classes =========================================================
class _hdf5_filehandle:
    """Get a handle to an open HDF5 file to read or write to.

    Parameters
    ----------
    filename : str or h5py File/Group handle
        * str : Name of the file to interact with.
        * h5py File/Group handle : handle to part of an already open HDF5 file.
    mode : str
        Type of interaction for the HDF5 file.
        * "save" : Open the file for writing only.
        * "load" : Open the file for reading only.
    overwrite : bool
        If True, overwrite the file if it already exists. If False,
        raise a FileExistsError if the file already exists.
        Only applies when ``mode = "save"``.
    """

    def __init__(self, filename, mode, overwrite=False):
        """Open the file handle."""
        if isinstance(filename, h5py.HLObject):
            # `filename` is already an open HDF5 file.
            self.file_handle = filename
            self.close_when_done = False

        elif mode == "save":
            # `filename` is the name of a file to create for writing.
            if not filename.endswith(".h5"):
                warnings.warn(
                    "expected file with extension '.h5'",
                    errors.OpInfWarning,
                )
            if os.path.isfile(filename) and not overwrite:
                raise FileExistsError(f"{filename} (overwrite=True to ignore)")
            self.file_handle = h5py.File(filename, "w")
            self.close_when_done = True

        elif mode == "load":
            # `filename` is the name of an existing file to read from.
            if not os.path.isfile(filename):
                raise FileNotFoundError(filename)
            self.file_handle = h5py.File(filename, "r")
            self.close_when_done = True

        else:
            raise ValueError(f"invalid mode '{mode}'")

    def __enter__(self):
        """Return the handle to the open HDF5 file."""
        return self.file_handle

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Close the file if needed."""
        if self.close_when_done:
            self.file_handle.close()
        if exc_type:
            raise


class hdf5_savehandle(_hdf5_filehandle):
    """Get a handle to an open HDF5 file to write to.

    Parameters
    ----------
    savefile : str or h5py File/Group handle
        * str : Name of the file to save to.
        * h5py File/Group handle : handle to part of an already open HDF5 file
          to save data to.
    overwrite : bool
        If ``True``, overwrite the file if it already exists.
        If ``False``, raise a ``FileExistsError`` if the file already exists.

    Examples
    --------
    >>> with hdf5_savehandle("file_to_save_to.h5", False) as hf:
    ...     hf.create_dataset("dataset_label", data=dataset_to_save)
    """

    def __init__(self, savefile, overwrite):
        return _hdf5_filehandle.__init__(self, savefile, "save", overwrite)


class hdf5_loadhandle(_hdf5_filehandle):
    """Get a handle to an open HDF5 file to read from.

    Parameters
    ----------
    loadfile : str or h5py File/Group handle
        * str : Name of the file to read from.
        * h5py File/Group handle : handle to part of an already open HDF5 file
          to read data from.

    Examples
    --------
    >>> with hdf5_loadhandle("file_to_read_from.h5") as hf:
    ...    dataset = hf["dataset_label"][:]
    """

    def __init__(self, loadfile):
        return _hdf5_filehandle.__init__(self, loadfile, "load")

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Close the file if needed. Raise a LoadfileFormatError if needed."""
        try:
            _hdf5_filehandle.__exit__(self, exc_type, exc_value, exc_traceback)
        except errors.LoadfileFormatError:
            raise
        except Exception as ex:
            raise errors.LoadfileFormatError(ex.args[0]) from ex


# Other tools =================================================================
def save_sparray(group: h5py.Group, arr: sparse.sparray) -> None:
    """Save a :mod:`scipy.sparse` matrix efficiently in an HDF5 group.

    This method mimics the behavior of :meth:`scipy.sparse.save_npz()` but
    for an open HDF5 file. See :func:`load_sparray()`.

    Parameters
    ----------
    arr : scipy.sparse.sparray
        Sparse SciPy array, in any sparse format.
    group : h5py.Group
        HDF5 group to save the sparse array to.

    Examples
    --------
    >>> import h5py
    >>> import scipy.sparse as sparse
    >>> from opinf.utils import save_sparray, load_sparray

    # Create a sparse array.
    >>> A = sparse.dok_array((100, 100), dtype=float)
    >>> A[0, 5] = 12
    >>> A[4, 1] = 123.456
    >>> A
    <100x100 sparse array of type '<class 'numpy.float64'>'
        with 2 stored elements in Dictionary Of Keys format>
    >>> print(A)
      (np.int32(0), np.int32(5))    12.0
      (np.int32(4), np.int32(1))    123.456

    # Save the sparse array to an HDF5 file.
    >>> with h5py.File("myfile.h5", "w") as hf:
    ...     save_sparray(hf.create_group("sparsearray"), A)

    # Load the sparse array from the file.
    >>> with h5py.File("myfile.h5", "r") as hf:
    ...     B = load_sparray(hf["sparsearray"])
    >>> B
    <100x100 sparse array of type '<class 'numpy.float64'>'
        with 2 stored elements in Dictionary Of Keys format>
    >>> print(B)
      (np.int32(0), np.int32(5))    12.0
      (np.int32(4), np.int32(1))    123.456
    """
    if not sparse.issparse(arr):
        raise TypeError("second arg must be a scipy.sparse array")

    # Convert to COO format and save data attributes.
    A = arr.tocoo()
    group.create_dataset("data", data=A.data)
    group.create_dataset("row", data=A.row)
    group.create_dataset("col", data=A.col)
    group.attrs["shape"] = A.shape
    group.attrs["arrtype"] = type(arr).__name__[:3]


def load_sparray(group: h5py.Group) -> sparse.sparray:
    """Save a :mod:`scipy.sparse` matrix efficiently in an HDF5 group.

    This method mimics the behavior of :meth:`scipy.sparse.load_npz()` but
    for an open HDF5 file. See :func:`save_sparray()`.

    Parameters
    ----------
    group : h5py.Group
        HDF5 group create and save the sparse array to.

    Returns
    -------
    arr : scipy.sparse.sparray
        Sparse SciPy array, in the sparse format it was in before saving.

    Examples
    --------
    >>> import h5py
    >>> import scipy.sparse as sparse
    >>> from opinf.utils import save_sparray, load_sparray

    # Create a sparse array.
    >>> A = sparse.dok_array((100, 100), dtype=float)
    >>> A[0, 5] = 12
    >>> A[4, 1] = 123.456
    >>> A
    <100x100 sparse array of type '<class 'numpy.float64'>'
        with 2 stored elements in Dictionary Of Keys format>
    >>> print(A)
      (np.int32(0), np.int32(5))    12.0
      (np.int32(4), np.int32(1))    123.456

    # Save the sparse array to an HDF5 file.
    >>> with h5py.File("myfile.h5", "w") as hf:
    ...     save_sparray(hf.create_group("sparsearray"), A)

    # Load the sparse array from the file.
    >>> with h5py.File("myfile.h5", "r") as hf:
    ...     B = load_sparray(hf["sparsearray"])
    >>> B
    <100x100 sparse array of type '<class 'numpy.float64'>'
        with 2 stored elements in Dictionary Of Keys format>
    >>> print(B)
      (np.int32(0), np.int32(5))    12.0
      (np.int32(4), np.int32(1))    123.456
    """
    A = sparse.coo_matrix(
        (group["data"], (group["row"], group["col"])),
        shape=group.attrs["shape"],
    )
    arrtype = str(group.attrs["arrtype"])
    return getattr(A, f"to{arrtype}")()
