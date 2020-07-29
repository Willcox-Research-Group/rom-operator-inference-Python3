# check_docs.py
"""Script for catching errors in README and DETAILS files."""

import os
import re
import difflib
from collections import defaultdict


# Global variables ============================================================
README = "README.md"
DETAILS = "DETAILS.md"
DOCS = "DOCUMENTATION.md"
SETUP = "setup.py"
INIT = os.path.join("rom_operator_inference", "__init__.py")
MDFILES = [README, DETAILS, DOCS]
DOCFILES = [DETAILS, DOCS]
VNFILES = [SETUP, INIT]

REDSPACE = '\x1b[41m \x1b[49m'
VERSION = re.compile(r'_{0,2}?version_{0,2}?\s*=\s*"([\d\.]+?)"',
                     re.MULTILINE)
MDREF = re.compile(r"\[.+?\]\((\./.+?)\)")
MDIMG = re.compile(r"""<img src=["'](\./.+?)["']>""")
MDLINK = re.compile(r"\[(.+?)\]\(([^\.].+?)\)")


# Tests =======================================================================
def check_references_sections_are_the_same(filelist=MDFILES):
    """Make sure that the "References" sections in each doc file are the same.
    """
    # Read both '## References' sections.
    refsections = []
    for filename in filelist:
        with open(filename, 'r') as infile:
            data = infile.read()
        refsections.append(data[data.index("## References"):].splitlines())

    # Compare sections and report errors.
    errors = False
    for i in range(len(filelist)-1):
        file1, file2 = filelist[i:i+2]
        for line in difflib.unified_diff(refsections[i], refsections[i+1],
                                         fromfile=file1, tofile=file2):
            print(line)
            errors = True
        if errors:
            raise SyntaxError(f"'References' of {file1} and {file2}"
                               " do not match")


def check_notation_sections_are_the_same(filelist=DOCFILES):
    """Make sure that the "Index of Notation" sections in each doc file are
    the same.
    """
    idxsections = []
    for filename in filelist:
        with open(filename, 'r') as infile:
            data = infile.read()
        start = data.index("## Index of Notation")
        end = data.index("## References")
        idxsections.append(data[start:end].splitlines())

    # Compare sections and report errors.
    errors = False
    for i in range(len(filelist)-1):
        file1, file2 = filelist[i:i+2]
        for line in difflib.unified_diff(idxsections[i], idxsections[i+1],
                                         fromfile=file1, tofile=file2):
            print(line)
            errors = True
        if errors:
            raise SyntaxError(f"'Index of Notation' of {file1} and {file2} "
                               "do not match")


def check_version_numbers_match(filelist=VNFILES):
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


def check_file_references_exist(filelist=MDFILES):
    """Make sure that Markdown references of the type [text](./file) and
    <img src="./file"> link to existing files.
    """
    for filename in filelist:
        with open(filename, 'r') as infile:
            data = infile.read()

        badrefs = [m for m in MDREF.findall(data) if not os.path.isfile(m)
                                                  and not os.path.isdir(m)]
        badrefs += [m for m in MDIMG.findall(data) if not os.path.isfile(m)]

        if badrefs:
            raise SyntaxError(f"Bad references in {filename} (missing file)"
                              "\n\t" + '\n\t'.join(badrefs))


def check_author_links_consistent(filelist=MDFILES, safelist=("Download",)):
    for filename in filelist:
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


if __name__ == "__main__":
    check_references_sections_are_the_same()
    check_notation_sections_are_the_same()
    check_author_links_consistent()
    check_file_references_exist()
    check_version_numbers_match()
