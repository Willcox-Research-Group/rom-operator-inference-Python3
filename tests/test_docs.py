# test_docs.py
"""Catch errors in README.md and ensure __version__ is synchronized."""

import os
import re
from collections import defaultdict


# Helper functions ============================================================
def up_one(f):
    """Get the file name in the folder above the current working directory."""
    return os.path.abspath(os.path.join("..", f))


# Global variables ============================================================
README = "README.md"
SETUP = "setup.py"
INIT = os.path.join("src", "rom_operator_inference", "__init__.py")

# Locate files if tests are being run directly from the tests/ directory.
if not os.path.isfile(README) and os.path.basename(os.getcwd()) == "tests":
    README = up_one(README)
    SETUP = up_one(SETUP)
    INIT = up_one(INIT)

VNFILES = [SETUP, INIT]
VERSION = re.compile(r'_{0,2}?version_{0,2}?\s*=\s*"([\d\.]+?-?\w+?)"',
                     re.MULTILINE)
MDLINK = re.compile(r"\[(.+?)\]\(([^\.].+?)\)")
SAFELIST = ["Download"]


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


def test_author_links():
    """Check that Markdown URLs are uniquely mapped."""
    with open(README, 'r') as infile:
        data = infile.read()

    links = defaultdict(set)
    for key, value in MDLINK.findall(data):
        links[key].add(value)

    badrefs = [k for k,v in links.items() if k not in SAFELIST and len(v) > 1]
    if badrefs:
        raise SyntaxError(f"Bad references in {README} (nonunique link)"
                          "\n\t" + '\n\t'.join(badrefs))
