# utils/_hdf5.py
"""Utilities for HDF5 file interaction."""

__all__ = [
    "hdf5_savehandle",
    "hdf5_loadhandle",
]

import os
import h5py


def _hdf5_filehandle(filename, mode, overwrite):
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
    if isinstance(filename, h5py.HLObject):
        # `filename` is already an open HDF5 file.
        return filename, False
    elif mode == "save":
        # `filename` is the name of a file to create for writing.
        if not filename.endswith(".h5"):
            filename += ".h5"
        if os.path.isfile(filename) and not overwrite:
            raise FileExistsError(f"{filename} (overwrite=True to ignore)")
        return h5py.File(filename, 'w'), True
    elif mode == "load":
        # `filename` is the name of an existing file to read from.
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        return h5py.File(filename, 'r'), True
    else:
        raise ValueError(f"invalid mode '{mode}'")


def hdf5_savehandle(savefile, overwrite):
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

    Returns
    -------
    hf : h5py File/Group handle
        File writing object.
    close_when_done : bool
        If True, the user should call hf.close() when done.
    """
    return _hdf5_filehandle(savefile, "save", overwrite)


def hdf5_loadhandle(loadfile):
    """Get a handle to an open HDF5 file to read from.

    Parameters
    ----------
    loadfile : str of h5py File/Group handle
        * str : Name of the file to read from. Extension ".h5" is appended.
        * h5py File/Group handle : handle to part of an already open HDF5 file
        to read data from.

    Returns
    -------
    hf : h5py File/Group handle
        File writing object.
    close_when_done : bool
        If True, the user should call hf.close() when done.
    """
    return _hdf5_filehandle(loadfile, "load", False)
