# bib2md.py
"""Generate the markdown literature page from bibtex."""

import re
import collections

import bibtexparser


HEADER = """# Literature

This page lists scholarly publications that develop, extend, or apply
Operator Inference.
"""
# [Click here submit a new entry](TODO).


# Google Scholar IDs (https://scholar.google.com/citation?user=<this ID>)
scholarIDS = {
    "abidnazari": "u8vJ9-oAAAAJ",
    "ashley": "9KFAXLYAAAAJ",
    "benner": "6zcRrC4AAAAJ",
    "chaudhuri": "oGL9YJIAAAAJ",
    "duff": "OAkPFdkAAAAJ",
    "farcas": "Cts5ePIAAAAJ",
    "geelen": "vBzKRMsAAAAJ",
    "ghattas": "A5vhsIYAAAAJ",
    "gomes": "s6mocWAAAAAJ",
    "goyal": "9rEfaRwAAAAJ",
    "gruber": "CJVuqfoAAAAJ",
    "guo": "eON6MykAAAAJ",
    "hartmann": "4XvBneEAAAAJ",
    "heiland": "wkHSeoYAAAAJ",
    "huang": "lUXijaQAAAAJ",
    "issan": "eEIe19oAAAAJ",
    "ju": "JkKUWoAAAAAJ",
    "junior": "66DEy5wAAAAJ",
    "karasözen": "R906kj0AAAAJ",
    "khodabakhshi": "lYr_g-MAAAAJ",
    "koike": "HFoIGcMAAAAJ",
    "kramer": "yfmbPNoAAAAJ",
    "mcquarrie": "qQ6JDJ4AAAAJ",
    "najera-flores": "HJ-Dfl8AAAAJ",
    "peherstorfer": "C81WhlkAAAAJ",
    "qian": "jnHI7wQAAAAJ",
    "sharma": "Pb-tL5oAAAAJ",
    "swischuk": "L9D0LBsAAAAJ",
    "tezaur": "Q3fx78kAAAAJ",
    "todd": "jzY8TSkAAAAJ",
    "tolley": "0kOHVOkAAAAJ",
    "uy": "hNN_KRQAAAAJ",
    "wen": "uXJoQCAAAAAJ",
    "willcox": "axvGyXoAAAAJ",
    "wright": "VFQRIOwAAAAJ",
    "yıldız": "UVPD79MAAAAJ",
}


specialChars = (
    (r"\~{a}", "ã"),
    (r"\'{e}", "é"),
    (r"\i{}", "ı"),
    (r"\"{o}", "ö"),
    (r"\"{u}", "ü"),
)


# Fields in the BibTex file to ignore.
toSkip = {
    "category",
    "url",
}


# Categories to group the references by.
categories = {
    "origin": "Original Paper",
    "survey": "Surveys",
    "method": "Operator Inference Methodologies",
    "theory": "Operator Inference Theory",
    "struct": "Operator Inference with Structure",
    "application": "Applications",
    "other": "Other",
}


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
        if (key := field.key) in toSkip:
            continue
        txt = f"{txt}&nbsp;&nbsp;{key} = {{{bibraw[key]}}},\n"
    txt = f"{txt}  }}".replace("\\", "\\\\")
    return txt.replace('\\\\"', '\\\\\\\\"')


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
                raise ValueError(f"change {bibfile} to avoid ',' in authors")
            names = author.split(" ")
            initials = " ".join([name[0] + "." for name in names[:-1]])
            authors.append(linkedname(initials, clean_name(names[-1])))

        if len(authors) == 0:
            raise ValueError("empty author field")
        if len(authors) == 1:
            authortxt = f"* {authors[0]}"
        elif len(authors) == 2:
            authortxt = f"* {authors[0]} and {authors[1]}"
        else:
            authortxt = "* " + ", ".join(authors[:-1]) + f", and {authors[-1]}"

        # Parse paper title.
        title = clean_title(entry["title"])
        if "url" in entry:
            titletxt = f"[**{title}**]({entry['url']})"
        else:
            titletxt = f"**{title}**"

        # Parse journal and year.
        citetxt = (
            f"{entry['journal']}, {entry['year']} "
            f"<details><summary>BibTeX</summary><pre>{entry2txt(entry)}"
            f"</pre></details>"
        )

        # Combine parsed data.
        cat = "other"
        if "category" in entry:
            cat = entry["category"]
        sectiontxt[cat].append("  \n  ".join([authortxt, titletxt, citetxt]))

    with open(mdfile, "w") as outfile:
        outfile.write(HEADER)
        for cat in categories:
            if cat not in sectiontxt:
                continue
            outfile.write(f"\n## {categories[cat]}\n")
            outfile.write("\n  <p></p>\n".join(sectiontxt[cat]) + "\n")


if __name__ == "__main__":
    main("literature.bib", "source/opinf/literature.md")
