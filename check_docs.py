# check_docs.py
"""Script for catching errors in README.md and synchronizing versions."""

import os
import re
from collections import defaultdict


# Global variables ============================================================
README = "README.md"
SETUP = "setup.py"
INIT = os.path.join("rom_operator_inference", "__init__.py")
VNFILES = [SETUP, INIT]

VERSION = re.compile(r'_{0,2}?version_{0,2}?\s*=\s*"([\d\.]+?)"',
                     re.MULTILINE)
MDLINK = re.compile(r"\[(.+?)\]\(([^\.].+?)\)")


# Tests =======================================================================

def check_version_numbers_match(filelist):
    """Make sure that the version number in setup.py and __init__.py match."""
    if len(filelist) != 2:
        raise ValueError("can only compare 2 files at a time")
    file1, file2 = filelist

    # Get the version number listed in each file.
    versions = []
    for filename in filelist:
        with open(filename, 'r') as infile:
            data = infile.read()
        versions.append(VERSION.findall(data)[0])

    if versions[0] != versions[1]:
        raise ValueError(f"Version numbers in {file1} and {file2} do "
                         f"not match ('{versions[0]}' != '{versions[1]}')")


def check_author_links_consistent(filename, safelist=("Download",)):
    with open(filename, 'r') as infile:
        data = infile.read()

    links = defaultdict(set)
    for key, value in MDLINK.findall(data):
        links[key].add(value)

    badrefs = [k for k,v in links.items()
                 if k not in safelist and len(v) > 1]
    if badrefs:
        raise SyntaxError(f"Bad references in {filename} (nonunique link)"
                          "\n\t" + '\n\t'.join(badrefs))


def main():
    check_version_numbers_match(VNFILES)
    check_author_links_consistent(README)


if __name__ == "__main__":
    main()
