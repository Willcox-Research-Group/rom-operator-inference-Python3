# `opinf.utils`

```{eval-rst}
.. automodule:: opinf.utils
```

This module contains miscellaneous support functions for the rest of the
package.

## Load/Save HDF5 Utilities

Many `opinf` classes have `save()` methods that export the object to an HDF5 file and a `load()` class method for importing an object from an HDF5
file.
The following functions facilitate that data transfer.

```{eval-rst}
.. currentmodule:: opinf.utils

.. autosummary::
    :toctree: _autosummaries
    :nosignatures:

    hdf5_loadhandle
    hdf5_savehandle
```
