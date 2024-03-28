# bib2md.py
"""Generate the markdown literature page from bibtex."""

# import collections

import bibtexparser


HEADER = """# Literature

This page lists scholar publications that develop, extend, or use Operator
Inference.
"""

# For new publications, [click here submit a new entry](TODO).


# Google Scholar IDs (https://scholar.google.com/citation?user=<this ID>)
scholarIDS = {
    "benner": "6zcRrC4AAAAJ",
    "duff": "OAkPFdkAAAAJ",
    "farcas": "Cts5ePIAAAAJ",
    "geelen": "vBzKRMsAAAAJ",
    "ghattas": "A5vhsIYAAAAJ",
    "goyal": "9rEfaRwAAAAJ",
    "guo": "eON6MykAAAAJ",
    "hartmann": "4XvBneEAAAAJ",
    "huang": "lUXijaQAAAAJ",
    "issan": "eEIe19oAAAAJ",
    "khodabakhshi": "lYr_g-MAAAAJ",
    "kramer": "yfmbPNoAAAAJ",
    "mcQuarrie": "qQ6JDJ4AAAAJ",
    "peherstorfer": "C81WhlkAAAAJ",
    "qian": "jnHI7wQAAAAJ",
    "sharma": "Pb-tL5oAAAAJ",
    "tezaur": "Q3fx78kAAAAJ",
    "uy": "hNN_KRQAAAAJ",
    "willcox": "axvGyXoAAAAJ",
    "wright": "VFQRIOwAAAAJ",
}


# Fields in the BibTex file to ignore.
toSkip = {
    "category",
    "url",
}


# Categories to group the references by.
categories = {
    "method": "Operator Inference methodologies",
    "theory": "Operator Inference theory",
    "struct": "Operator Inference with structure",
    "application": "Applications",
    "survey": "Surveys",
    "other": "Other",
}

# define string for each category
categorieswww = {key: "" for key in categories}
# categorieswww = collections.defaultdict()


def linkedname(firstname, lastname):
    """Get the string of the form "First, Last" with a link to their
    Google Scholar page if possible.
    """
    if (lname := lastname.lower()) in scholarIDS:
        gsID = scholarIDS[lname]
        url = f"https://scholar.google.com/citations?user={gsID}"  # &hl=en
        # return "[" + firstname + " " + lastname + "](" + d[lastname] + ")"
        return f"[{firstname} {lastname}]({url})"
    return f"{firstname} {lastname}"


def entry2txt(bibraw):
    """Convert the bibtexparser entry to a printable string."""
    txt = f"@{bibraw.entry_type}{{{bibraw.key},\n"
    for field in bibraw.fields:
        key = field.key
        if key in toSkip:
            continue
        txt = f"{txt}      {key} = {{{bibraw[key]}}},\n"
    return f"{txt}  }}"


def main(bibfile, mdfile):
    """Convert a BibTex file to Markdown."""

    library = bibtexparser.parse_file(bibfile)

    for entry in sorted(
        library.entries,
        key=lambda x: x["year"],
        reverse=False,
    ):
        # Parse the category.
        cat = "other"
        if "category" in entry:
            cat = entry["category"]

        # Parse authors.
        authors = []
        for author in entry["author"].split(" and "):
            author = author.strip()
            if "," in author:
                raise ValueError(f"change {bibfile} to avoid ',' in authors")
            names = author.split(" ")
            initials = " ".join([name[0] + "." for name in names])
            authors.append(linkedname(initials, names[-1]))

        if len(authors) == 0:
            raise ValueError("empty author field")
        if len(authors) == 1:
            line1 = f"- {authors[0]}"
        elif len(authors) == 2:
            line1 = f"- {authors[0]} and {authors[1]}"
        else:
            line1 = "- " + ", ".join(authors[:-1]) + f", and {authors[-1]}"

        # Parse paper title.
        if "url" in entry:
            line2 = f"[**{entry['title']}**]({entry['url']})"
        else:
            line2 = f"**{entry['title']}**"

        # Parse journal and year.
        line3 = (
            f"{entry['journal']}, {entry['year']} "
            "<details><summary>BibTeX</summary><pre>"
        )

        line4 = f"{entry2txt(entry)}</pre></details>"

        # Combine parsed data.
        categorieswww[cat] = (
            categorieswww[cat]
            + "\n"
            + "  \n  ".join([line1, line2, line3, line4])
        )

    with open(mdfile, "w") as outfile:
        outfile.write(HEADER)
        for cat in categorieswww:
            if categorieswww[cat] == "":
                continue
            outfile.write("\n## " + categories[cat] + "\n")
            outfile.write(categorieswww[cat] + "\n")


if __name__ == "__main__":
    main("literature.bib", "source/opinf/literature.md")
