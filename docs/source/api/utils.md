# `opinf.utils`

```{eval-rst}
.. automodule:: opinf.utils
```

## Timing Code

Model reduction is all about speeding up computational tasks.
The following class defines a context manager for timing blocks of code and logging errors.

```{eval-rst}
.. currentmodule:: opinf.utils

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    TimedBlock
```

## Load/Save HDF5 Utilities

Many `opinf` classes have `save()` methods that export the object to an HDF5 file and a `load()` class method for importing an object from an HDF5 file.
The following functions facilitate that data transfer.

```{eval-rst}
.. currentmodule:: opinf.utils

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    hdf5_loadhandle
    hdf5_savehandle
```

## Helper Routines

The following functions perform miscellaneous tasks within the rest of the code base.

```{eval-rst}
.. currentmodule:: opinf.utils

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    gridsearch
    requires
    requires2
    str2repr
```

## Documentation

The following function initializes the Matplotlib defaults used in the documentation notebooks.

```{eval-rst}
.. currentmodule:: opinf.utils

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    mpl_config
```
