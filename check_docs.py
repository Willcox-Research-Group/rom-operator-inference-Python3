# check_docs.py
"""Script for catching errors in README and DETAILS files."""

import os
import re
import difflib


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
VERSION = re.compile(r'_{0,2}?version_{0,2}?\s*=\s*"([\d\.]+?-?\w+?)"',
                     re.MULTILINE)
TEX = re.compile(
                r'''<p\ align="center">                 # begin outer tag
                \s*                                     #  optional space
                <img\ src=                              #  begin inner tag
                "https://latex.codecogs.com/svg.latex\? #   LaTeX web prefix
                (.+)                                    #   LATEX CODE
                "/>                                     #  end inner tag
                \s*                                     #  optional space
                </p>                                    # end outer tag
                ''',
                re.VERBOSE | re.MULTILINE)


# Tests =======================================================================
def check_latex_code_for_spaces(filelist=MDFILES):
    """Make sure there are no spaces in the latex code snippets, since that
    keeps them from displaying correctly on Github.
    """
    # Read files and search for errors.
    errors = []
    for filename in filelist:
        with open(filename, 'r') as infile:
            data = infile.read()
        for index,tex in enumerate(TEX.findall(data)):
            if ' ' in tex:
                errors.append((filename, index, tex))

    # Report errors.
    for name,index,tex in errors:
        print(f"bad space in {name}, LaTeX occurrence #{index+1}:\n\t",
              tex.replace(' ', REDSPACE),
              sep='', end='\n')
    if errors:
        nerrs = len(errors)
        raise SyntaxError(f"{nerrs} LaTeX error{'s' if nerrs > 1 else ''}")


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


def check_index_sections_are_the_same(filelist=DOCFILES):
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
                               " do not match")


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


if __name__ == "__main__":
    check_latex_code_for_spaces()
    check_references_sections_are_the_same()
    check_index_sections_are_the_same()
    check_version_numbers_match()
