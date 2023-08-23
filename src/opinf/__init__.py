# __init__.py
"""Operator Inference for data-driven model reduction of dynamical systems.

Author: Willcox Research Group
Maintainer: Shane A. McQuarrie
GitHub:
    https://github.com/Willcox-Research-Group/rom-operator-inference-Python3
"""

__version__ = "0.4.5"

from .roms import *
from .operators import *
from . import basis, errors, lstsq, operators, pre, post, roms, utils
