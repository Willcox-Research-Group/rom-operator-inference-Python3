# utils/test_hdf5.py
"""Tests for rom_operator_inference.utils._hdf5."""

import os
import h5py
import pytest

import rom_operator_inference as opinf


def test_hdf5_filehandle():
    """Test utils._hdf5._hdf5_filehandle()."""
    func = opinf.utils._hdf5._hdf5_filehandle

    # Clean up after old tests.
    target = "_hdf5handletest.h5"
    if os.path.isfile(target):              # pragma: no cover
        os.remove(target)

    # Input file is already an open h5py handle.
    h5file = h5py.File(target, 'a')
    hf, toclose = func(h5file, "load", False)
    assert hf is h5file
    assert toclose is False
    hf.close()
    os.remove(target)

    # Save mode with no .h5 extension
    hf, toclose = func(target[:-3], "save", True)
    assert isinstance(hf, h5py.File)
    assert toclose is True
    assert os.path.isfile(target)
    hf.close()

    # Try to overwrite but with overwrite=False.
    with pytest.raises(FileExistsError) as ex:
        func(target, "save", overwrite=False)
    assert ex.value.args[0] == f"{target} (overwrite=True to ignore)"

    # Overwrite.
    hf, toclose = func(target, "save", overwrite=True)
    assert isinstance(hf, h5py.File)
    assert toclose is True
    assert os.path.isfile(target)
    hf.close()

    # Loading.
    hf, toclose = func(target, "load", False)
    assert isinstance(hf, h5py.File)
    assert toclose is True
    hf.close()
    os.remove(target)

    # Try loading a nonexistent file.
    with pytest.raises(FileNotFoundError) as ex:
        func(target, "load", overwrite=False)
    assert ex.value.args[0] == target

    # Invalid mode.
    with pytest.raises(ValueError) as ex:
        func(target, "moose", None)
    assert ex.value.args[0] == "invalid mode 'moose'"


def test_hdf5_savehandle():
    """Test utils._hdf5.hdf5_savehandle()."""
    func = opinf.utils.hdf5_savehandle

    # Clean up after old tests.
    target = "_hdf5savehandletest.h5"
    if os.path.isfile(target):              # pragma: no cover
        os.remove(target)

    # Input file is already an open h5py handle.
    h5file = h5py.File(target, 'a')
    hf, toclose = func(h5file, False)
    assert hf is h5file
    assert toclose is False
    hf.close()
    os.remove(target)

    # Save mode with no .h5 extension
    hf, toclose = func(target[:-3], True)
    assert isinstance(hf, h5py.File)
    assert toclose is True
    assert os.path.isfile(target)
    hf.close()

    # Try to overwrite but with overwrite=False.
    with pytest.raises(FileExistsError) as ex:
        func(target, overwrite=False)
    assert ex.value.args[0] == f"{target} (overwrite=True to ignore)"

    # Overwrite.
    hf, toclose = func(target, overwrite=True)
    assert isinstance(hf, h5py.File)
    assert toclose is True
    assert os.path.isfile(target)
    hf.close()
    os.remove(target)


def test_hdf5_loadhandle():
    """Test utils._hdf5.hdf5_loadhandle()."""
    func = opinf.utils.hdf5_loadhandle

    # Clean up after old tests.
    target = "_hdf5loadhandletest.h5"
    if os.path.isfile(target):              # pragma: no cover
        os.remove(target)

    with h5py.File(target, 'w'):
        pass

    # Loading.
    hf, toclose = func(target)
    assert isinstance(hf, h5py.File)
    assert toclose is True
    hf.close()
    os.remove(target)

    # Try loading a nonexistent file.
    with pytest.raises(FileNotFoundError) as ex:
        func(target)
    assert ex.value.args[0] == target
