# pre/basis/__init__.py
"""Classes for low-dimensional state representations.

Please note that this module is private.  All public objects are available in
the pre namespace - use that instead whenever possible.

Public Classes
--------------
* LinearBasis: q = Vr qhat
* PODBasis: q = Vr qhat

Private Classes
---------------
* _BaseBasis: Base class for all basis classes.
"""

from ._linear import *
from ._pod import *
