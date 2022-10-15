# test_version.py
"""Ensure __version__ is synchronized in setup.py and the package itself."""

import os
import re


# Helper functions ============================================================
def up_one(f):
    """Get the file name in the folder above the current working directory."""
    return os.path.abspath(os.path.join("..", f))


# Global variables ============================================================
SETUP = "setup.py"
INIT = os.path.join("src", "opinf", "__init__.py")

# Locate files if tests are being run directly from the tests/ directory.
if not os.path.isfile(SETUP) and os.path.basename(os.getcwd()) == "tests":
    SETUP = up_one(SETUP)
    INIT = up_one(INIT)

VNFILES = [SETUP, INIT]
VERSION = re.compile(r'_{0,2}?version_{0,2}?\s*=\s*"([\d\.]+?-?\w+?)"',
                     re.MULTILINE)


# Tests =======================================================================
def test_version_numbers():
    """Check that the version number in setup.py and __init__.py match."""
    file1, file2 = VNFILES

    # Get the version number listed in each file.
    versions = []
    for filename in VNFILES:
        with open(filename, 'r') as infile:
            data = infile.read()
        versions.append(VERSION.findall(data)[0])

    if versions[0] != versions[1]:
        raise ValueError(f"Version numbers in {file1} and {file2} do "
                         f"not match ('{versions[0]}' != '{versions[1]}')")
