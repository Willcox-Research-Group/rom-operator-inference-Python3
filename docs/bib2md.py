# bib2md.py
"""Generate the markdown Literature page from bibtex.

Instructions for updating the Literature page:

1. Add BibTeX entries to `docs/literature.bib`.
    - Keep the entries sorted by year, then by author last name.
    - Authors should be listed with first-then-last names and separated with
    "and": `authors = {{First1 Last1 and First2 Last2}},`.
    Enclose multi-word names in braces, for example,
    `authors = {{Vincent {{van Gogh}} and Balthasar {{van der Pol}}}},`
   - Include a "doi" field if applicable.
   - Add a "category" field to indicate which section the reference should be
     listed under on this page.

2. Add the Google Scholar IDs of each author who has one to the `scholarIDS`
   dictionary in `docs/bib2md.py`. This is the unique part of a Google Scholar
   profile page url:
   `https://scholar.google.com/citations?user=<GoogleScholarID>&hl=en`

Note that `docs/literature.bib` is generated automatically from this file and
is not tracked by git.
"""

import re
import collections

import bibtexparser
import bibtexparser.middlewares as bm


# Configuration ===============================================================

# Categories to group the references by.
# These are the options for the `categories` field in literature.bib entries.
categories = {
    "origin": "## Original Paper",
    "survey": "## Surveys",
    "nonlinear": "## Methodology\n### Lifting and Nonlinearity",
    "reprojection": "### Re-projection",
    "structure": "### Structure Preservation",
    "parametric": "### Parametric Problems",
    "statistical": "### Statistical Methods",
    "domaindecomp": "### Domain Decomposition",
    "manifolds": "### Nonlinear Manifolds",
    "scalable": "### Scalability",
    "application": "## Applications",
    "thesis": "## Dissertations and Theses",
}

# Extra text to place under the heading before continuing.
details = {
    "nonlinear": """Operator Inference learns reduced-order models with
polynomial structure. The methods developed in the following papers focus on
dealing with non-polynomial nonlinearities through variable transformations
(lifting) and/or coupling Operator Inference methods with other approximation
strategies.""",
    "reprojection": """In some cases, if the training data are chosen
judiciously, Operator Inference can recover traditional reduced-order models
defined by intrusive projection. The following papers develop and apply this
idea.""",
    "structure": """The methods developed in these works augment Operator
Inference so that the resulting reduced-order models automatically inherit
certain properties from the full-order system, such as block structure,
symmetries, energy conservation, gradient structure, and more.""",
    "parametric": """Many systems depend on independent parameters that
describe material properties or other physical characteristics of the
phenomenon being modeled. The following papers develop Operator Inference
approaches that are specifically designed for parametric problems.""",
    "statistical": """These papers focus on problems with noisy or missing
data, stochastic systems, and methods for constructing probabilistic
reduced-order models with Operator Inference.""",
    "domaindecomp": """The methods in the following papers focus on scalability
and accuracy improvements by decomposition spatial or latent space domains and
learning a coupled system of reduced-order models.""",
    "manifolds": """Traditional model reduction methods approximate the
high-dimensional system state with a low-dimensional linear (or affine)
representation. The methods in these papers explore using nonlinear
low-dimensional representations in the context of Operator Inference.""",
    "scalable": """These works focus on the computational challenge of applying
Operator Inference to large-scale problems.""",
}

# Author citation IDs (https://scholar.google.com/citation?user=<this ID>)
scholarIDS = {
    "nicole_aretz": "Oje7mbAAAAAJ",
    "anthony_ashley": "9KFAXLYAAAAJ",
    "sean_babiniec": "xcSVh00AAAAJ",
    "laura_balzano": "X6fRNfUAAAAJ",
    "peter_benner": "6zcRrC4AAAAJ",
    "patrick_blonigan": "lOmH5XcAAAAJ",
    "anirban_chaudhuri": "oGL9YJIAAAAJ",
    "pascal_denboef": "vFlzL7kAAAAJ",
    "igor_duff": "OAkPFdkAAAAJ",
    "melina_freitag": "iE4t4WcAAAAJ",
    "ionut-gabriel_farcas": "Cts5ePIAAAAJ",
    "rudy_geelen": "vBzKRMsAAAAJ",
    "yuwei_geng": "lms4MbwAAAAJ",
    "omar_ghattas": "A5vhsIYAAAAJ",
    "leonidas_gkimisis": "0GzUUzMAAAAJ",
    "marcos_gomes": "s6mocWAAAAAJ",
    "pawan_goyal": "9rEfaRwAAAAJ",
    "anthony_gruber": "CJVuqfoAAAAJ",
    "mengwu_guo": "eON6MykAAAAJ",
    "dirk_hartmann": "4XvBneEAAAAJ",
    "jan_heiland": "wkHSeoYAAAAJ",
    "cheng_huang": "lUXijaQAAAAJ",
    "opal_issan": "eEIe19oAAAAJ",
    "lili_ju": "JkKUWoAAAAAJ",
    "bülent_karasözen": "R906kj0AAAAJ",
    "parisa_khodabakhshi": "lYr_g-MAAAAJ",
    "hyeonghun_kim": "sdR-LZ4AAAAJ",
    "tomoki_koike": "HFoIGcMAAAAJ",
    "boris_kramer": "yfmbPNoAAAAJ",
    "diana_manvelyan": "V0k8Xb4AAAAJ",
    "jan_nicolaus": "47DRMUwAAAAJ",
    "joseph_maubach": "nBRKw6cAAAAJ",
    "alexandre_marquez": "p9zb2Y0AAAAJ",
    "shane_mcquarrie": "qQ6JDJ4AAAAJ",
    "jonathan_murray": "NScAg7AAAAAJ",
    "david_najera-flores": "HJ-Dfl8AAAAJ",
    "alberto_nogueira": "66DEy5wAAAAJ",
    "benjamin_peherstorfer": "C81WhlkAAAAJ",
    "arthur_pires": "qIUw-GEAAAAJ",
    "gleb_pogudin": "C5NP1o0AAAAJ",
    "elizabeth_qian": "jnHI7wQAAAAJ",
    "thomas_richter": "C8R6xtMAAAAJ",
    "wil_schilders": "UGKPyqkAAAAJ",
    "harsh_sharma": "Pb-tL5oAAAAJ",
    "jasdeep_singh": "VcmXMxgAAAAJ",
    "renee_swischuk": "L9D0LBsAAAAJ",
    "john_tencer": "M6AwtC4AAAAJ",
    "irina_tezaur": "Q3fx78kAAAAJ",
    "marco_tezzele": "UPcyNXIAAAAJ",
    "michael_todd": "jzY8TSkAAAAJ",
    "michael_tolley": "0kOHVOkAAAAJ",
    "wayne_uy": "hNN_KRQAAAAJ",
    "nathan_vandewouw": "pcQCbN8AAAAJ",
    "arjun_vijaywargiya": "_fcSwDYAAAAJ",
    "zhu_wang": "jkmwEF0AAAAJ",
    "yuxiao_wen": "uXJoQCAAAAAJ",
    "karen_willcox": "axvGyXoAAAAJ",
    "stephen_wright": "VFQRIOwAAAAJ",
    "süleyman_yıldız": "UVPD79MAAAAJ",
    "benjamin_zastrow": "ODLjrBAAAAAJ",
}

# LaTeX special characters to convert for the markdown version.
specialChars = (
    (r"\~{a}", "ã"),
    (r"\'{e}", "é"),
    (r"\i{}", "ı"),
    (r"\"{o}", "ö"),
    (r"\"{u}", "ü"),
)

# Text before the list of references begins.
HEADER = r"""# Literature

This page lists scholarly publications that develop, extend, or apply
Operator Inference, categorized into topics and sorted by publication year,
then by the last name of the first author. Although some could be placed in
multiple categories, each publication is only listed once.

:::{admonition} Share Your Work!
:class: hint

Don't see your publication?
[**Click here**](https://forms.gle/BgZK4b4DfuaPsGFd7)
to submit a request to add entries to this page.
:::
"""

# Text after the list of references.
FOOTER = r"""## BibTex File

:::{{admonition}} Sorted alphabetically by author
:class: dropdown seealso

```bibtex
{}
```
:::

:::{{admonition}} Sorted by year then alphabetically by author
:class: dropdown seealso

```bibtex
{}
```
:::
"""


# Helper functions ============================================================
class TrimMiddleware(bm.BlockMiddleware):
    """Trim out a few fields when writing the bibtex file."""

    def transform_entry(self, entry, *args, **kwargs):
        for field in "category", "url":
            if field in entry:
                entry.pop(field)
        return entry


def clean_name(name):
    r"""Handle special LaTex characters in names:
    \~{a} -> ã
    \'{e} -> é
    \i{} -> ı
    \"{o} -> ö
    \"{u} -> ü
    """
    for before, after in specialChars:
        name = name.replace(before, after)
    return name


def clean_title(title):
    r"""Remove extra braces, etc. from paper titles,
    e.g.,"{B}\'{e}nard" -> Bénard.
    """
    return re.subn(r"\{(\w+?)\}", r"\1", clean_name(title))[0]


def linkedname(author):
    """Get the string of the form "First, Last" with a link to their
    Google Scholar page if possible.
    """
    # Separate names, keeping {} groups together.
    names = [
        match[1:-1] if match.startswith("{") and match.endswith("}") else match
        for match in re.findall(r"\{[^}]*\}|\S+", author)
    ]

    # Extract first and last names and initials.
    firstname = clean_name(names[0]).lower()
    if names[-1] == "Jr":
        lastname = clean_name(names[-2])
        initials = " ".join([name[0] + "." for name in names[:-2]])
        key = f"{firstname}_{lastname.replace(" ", "").lower()}"
        lastname = f"{lastname} Jr."
    else:
        lastname = clean_name(names[-1])
        initials = " ".join([name[0] + "." for name in names[:-1]])
        key = f"{firstname}_{lastname.replace(" ", "").lower()}"

    # Get the Google Scholar link if possible.
    if key in scholarIDS:
        gsID = scholarIDS[key]
        gsURL = f"https://scholar.google.com/citations?user={gsID}"  # &hl=en
        return f"[{initials} {lastname}]({gsURL})"

    return f"{initials} {lastname}"


def entry2txt(bibraw):
    """Convert the bibtexparser entry to a printable string."""
    txt = f"@{bibraw.entry_type}{{{bibraw.key},\n"
    for field in bibraw.fields:
        if (key := field.key) == "category":
            continue
        elif key == "url" and "doi" in bibraw:
            continue
        txt = f"{txt}&nbsp;&nbsp;{key} = {{{bibraw[key]}}},\n"
    txt = f"{txt}  }}".replace("\\", "\\\\")
    return txt.replace('\\\\"', '\\\\\\\\"')


# Main routine ================================================================
def main(bibfile, mdfile):
    """Convert a BibTex file to Markdown."""

    library = bibtexparser.parse_file(bibfile)
    sectiontxt = collections.defaultdict(list)

    for entry in sorted(
        library.entries,
        key=lambda x: x["year"],
        reverse=False,
    ):
        # Parse authors.
        authors = []
        for author in entry["author"].split(" and "):
            author = author.strip()
            if "," in author:
                raise ValueError(
                    f"change {bibfile} to avoid ',' in author '{author}'"
                )
            authors.append(linkedname(author))

        if len(authors) == 0:
            raise ValueError("empty author field")
        if len(authors) == 1:
            authortxt = f"{authors[0]}"
        elif len(authors) == 2:
            authortxt = f"{authors[0]} and {authors[1]}"
        else:
            authortxt = ", ".join(authors[:-1]) + f", and {authors[-1]}"

        # Parse paper title.
        title = clean_title(entry["title"])
        if "url" in entry:
            titletxt = f"* [**{title}**]({entry['url']})"
        elif "doi" in entry:
            titletxt = f"* [**{title}**](https://doi.org/{entry['doi']})"
        else:
            titletxt = f"* **{title}**"

        # Parse journal and year.
        if "journal" in entry:
            publication = entry["journal"]
        elif "booktitle" in entry:
            publication = entry["booktitle"]
        elif entry["category"] == "thesis" and "school" in entry:
            if entry.entry_type == "phdthesis":
                publication = "PhD Thesis, " + entry["school"]
            elif entry.entry_type == "mastersthesis":
                publication = "Master's Thesis, " + entry["school"]
            else:
                raise ValueError("could not identify publication")
        else:
            raise ValueError("could not identify publication")

        citetxt = (
            f"{publication}, {entry['year']} "
            f"<details><summary>BibTeX</summary><pre>{entry2txt(entry)}"
            f"</pre></details>"
        )

        # Combine parsed data.
        cat = "other"
        if "category" in entry:
            cat = entry["category"]
        sectiontxt[cat].append("  \n  ".join([titletxt, authortxt, citetxt]))

    formatter = bibtexparser.BibtexFormat()
    formatter.indent = "    "
    formatter.trailing_comma = True
    formatter.block_separator = "\n"

    with open(mdfile, "w") as outfile:
        outfile.write(HEADER)
        for cat in categories:
            if cat not in sectiontxt:
                continue
            text = f"\n{categories[cat]}\n\n"
            if cat in details:
                text = f"{text}{details[cat]}\n\n"
            outfile.write(text)
            outfile.write("\n  <p></p>\n".join(sectiontxt[cat]) + "\n")

        footer = FOOTER.format(
            bibtexparser.write_string(  # Sorted by 1st author last name.
                bibtexparser.parse_file(
                    bibfile,
                    append_middleware=[bm.SortBlocksByTypeAndKeyMiddleware()],
                ),
                bibtex_format=formatter,
                prepend_middleware=[TrimMiddleware()],
            ),
            bibtexparser.write_string(  # Sorted by year, then author.
                library,
                bibtex_format=formatter,
                prepend_middleware=[TrimMiddleware()],
            ),
        )

        outfile.write(footer)


# =============================================================================
if __name__ == "__main__":
    main("literature.bib", "source/opinf/literature.md")
