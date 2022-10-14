# pre/transformer/__init__.py
"""Classes for pre-processing state snapshot data.

Please note that this module is private.  All public objects are available in
the pre namespace - use that instead whenever possible.

Public Classes
--------------
* SnapshotTransformer: transformer for single-variable data.
* SnapshotTransformerMulti: transformer for multi-variable data.

Public Functions
----------------
* shift(): TODO
* scale(): TODO

Private Classes
---------------
* _BaseTransformer: Base class for all transformer classes.
"""

from ._shiftscale import *
