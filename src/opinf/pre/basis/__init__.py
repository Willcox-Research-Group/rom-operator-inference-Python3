# pre/basis/__init__.py
"""Classes for low-dimensional state representations.

Please note that this module is private.  All public objects are available in
the pre namespace - use that instead whenever possible.

Public Classes
--------------
* LinearBasis: q = Vr qhat (Vr provided by user)
* PODBasis: q = Vr qhat (Vr chosen via SVD)
* PODBasisMulti : q1 = Vr1 qhat1, q2 = Vr2 qhat2, ... (Vr chosen via SVD)

Private Classes
---------------
* _BaseBasis: Base class for all basis classes.
"""

from ._linear import *
from ._pod import *
