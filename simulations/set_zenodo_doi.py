r"""
Write the real Zenodo DOI where the manuscript and the code repository state where the code
lives, then rebuild both submission packages.

    python simulations/set_zenodo_doi.py 10.5281/zenodo.1234567
    python simulations/set_zenodo_doi.py none        # drop the DOI clause, keep the repository

Four files carry the placeholder `10.5281/zenodo.XXXXXXX`:

    paper/main.tex                    Data and code availability (development source)
    simulations/build_test.py         Declarations block (used by the TEST build)
    code_release/README.md            the repository's own header block
    code_release/CITATION.cff         citation metadata

After a DOI has been written and the package rebuilt, re-generate the release tree and push
the change so the public repository shows its own DOI:

    python simulations/make_code_release.py
    cd code_release
    git add -A && git commit -m "Record the Zenodo DOI" && git push

Run from the repository root.
"""

import io
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLACEHOLDER = "10.5281/zenodo.XXXXXXX"
TARGETS = [os.path.join("paper", "main.tex"),
           os.path.join("simulations", "build_test.py"),
           os.path.join("simulations", "make_code_release.py"),
           os.path.join("code_release", "README.md"),
           os.path.join("code_release", "CITATION.cff")]


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    doi = sys.argv[1].strip()
    if doi.lower() in ("none", "-"):
        replacement = None
    else:
        doi = doi.replace("https://doi.org/", "").replace("doi:", "").strip()
        if not re.match(r"^10\.\d{4,9}/\S+$", doi):
            raise SystemExit(f"[error] '{doi}' does not look like a DOI")
        replacement = doi

    for rel in TARGETS:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            print(f"[skip] {rel} (not present)")
            continue
        t = io.open(p, encoding="utf-8").read()
        before = t.count(PLACEHOLDER)
        if replacement is None:
            t = t.replace(f"https://doi.org/{PLACEHOLDER} (", "(")
            t = t.replace(f"at https://doi.org/{PLACEHOLDER}", "in the repository")
            t = t.replace(f"https://doi.org/{PLACEHOLDER}", "")
            t = t.replace(PLACEHOLDER, "")
        else:
            t = t.replace(PLACEHOLDER, replacement)
        io.open(p, "w", encoding="utf-8").write(t)
        print(f"[written] {rel}  ({before} placeholder(s) -> "
              f"{'removed' if replacement is None else replacement})")

    print("[rebuild] the TEST submission package and the development PDF")
    for cmd, cwd in (("python simulations/build_test.py --si all", ROOT),
                     ("pdflatex -interaction=nonstopmode -halt-on-error main.tex", "paper"),
                     ("bibtex main", "paper"),
                     ("pdflatex -interaction=nonstopmode -halt-on-error main.tex", "paper"),
                     ("pdflatex -interaction=nonstopmode -halt-on-error main.tex", "paper")):
        r = subprocess.run(cmd, cwd=os.path.join(ROOT, cwd) if cwd != ROOT else ROOT,
                           shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = r.stdout.decode("utf-8", "replace")
        if r.returncode != 0:
            print(out[-1500:])
            raise SystemExit(f"[failed] {cmd}")
        tail = [l for l in out.strip().split("\n")
                if "pages" in l or "written" in l or "package" in l or "Output written" in l]
        if tail:
            print("   " + " | ".join(x.strip() for x in tail[-3:]))

    left = []
    for rel in TARGETS:
        p = os.path.join(ROOT, rel)
        if os.path.exists(p) and "XXXXXXX" in io.open(p, encoding="utf-8").read():
            left.append(rel)
    print("[check] placeholder still present in: " + (", ".join(left) if left else "nowhere"))
    print("[next]  python simulations/make_code_release.py   then commit and push the repo")


if __name__ == "__main__":
    main()

