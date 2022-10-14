# utils/_hdf5.py
"""Utilities for HDF5 file interaction."""

__all__ = [
    "hdf5_savehandle",
    "hdf5_loadhandle",
]

import os
import h5py


class _hdf5_filehandle:
    """Get a handle to an open HDF5 file to read or write to.

    Parameters
    ----------
    filename : str of h5py File/Group handle
        * str : Name of the file to interact with.
        * h5py File/Group handle : handle to part of an already open HDF5 file.
    mode : str
        Type of interaction for the HDF5 file.
        * "save" : Open the file for writing only.
        * "load" : Open the file for reading only.
    overwrite : bool
        If True, overwrite the file if it already exists. If False,
        raise a FileExistsError if the file already exists.
        Only applies when mode = "save".
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
                filename += ".h5"
            if os.path.isfile(filename) and not overwrite:
                raise FileExistsError(f"{filename} (overwrite=True to ignore)")
            self.file_handle = h5py.File(filename, 'w')
            self.close_when_done = True

        elif mode == "load":
            # `filename` is the name of an existing file to read from.
            if not os.path.isfile(filename):
                raise FileNotFoundError(filename)
            self.file_handle = h5py.File(filename, 'r')
            self.close_when_done = True

        else:
            raise ValueError(f"invalid mode '{mode}'")

    def __enter__(self):
        """Return the handle to the open HDF5 file."""
        return self.file_handle

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """CLose the file if needed."""
        if self.close_when_done:
            self.file_handle.close()
        if exc_type:
            raise


class hdf5_savehandle(_hdf5_filehandle):
    """Get a handle to an open HDF5 file to write to.

    Parameters
    ----------
    savefile : str of h5py File/Group handle
        * str : Name of the file to save to. Extension ".h5" is appended.
        * h5py File/Group handle : handle to part of an already open HDF5 file
        to save data to.
    overwrite : bool
        If True, overwrite the file if it already exists. If False,
        raise a FileExistsError if the file already exists.

    >>> with hdf5_savehandle("file_to_save_to.h5") as hf:
    ...     hf.create_dataset(...)
    """
    def __init__(self, savefile, overwrite):
        return _hdf5_filehandle.__init__(self, savefile, "save", overwrite)


class hdf5_loadhandle(_hdf5_filehandle):
    """Get a handle to an open HDF5 file to read from.

    Parameters
    ----------
    loadfile : str of h5py File/Group handle
        * str : Name of the file to read from. Extension ".h5" is appended.
        * h5py File/Group handle : handle to part of an already open HDF5 file
        to read data from.

    >>> with hdf5_loadhandle("file_to_read_from.h5") as hf:
    ...    data = hf[...]
    """
    def __init__(self, loadfile):
        return _hdf5_filehandle.__init__(self, loadfile, "load")
