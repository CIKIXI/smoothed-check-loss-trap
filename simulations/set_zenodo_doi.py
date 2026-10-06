r"""
Write the real Zenodo DOI into the two places where the manuscript states where the code
lives, then rebuild both submission packages.

    python simulations/set_zenodo_doi.py 10.5281/zenodo.1234567
    python simulations/set_zenodo_doi.py none        # drop the DOI clause, keep the repository

The two places are the Data and code availability section of `paper/main.tex` (used by the AoS
build) and the Declarations block of `simulations/build_test.py` (used by the TEST build).
Both currently carry the placeholder `10.5281/zenodo.XXXXXXX`.

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
           os.path.join("simulations", "build_test.py")]


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
        t = io.open(p, encoding="utf-8").read()
        before = t.count(PLACEHOLDER)
        if replacement is None:
            t = t.replace(f"https://doi.org/{PLACEHOLDER} (", "(")
            t = t.replace(PLACEHOLDER, "")
        else:
            t = t.replace(PLACEHOLDER, replacement)
        io.open(p, "w", encoding="utf-8").write(t)
        print(f"[written] {rel}  ({before} placeholder(s) -> "
              f"{'removed' if replacement is None else replacement})")

    print("[rebuild] both submission packages")
    for cmd in ("python simulations/build_test.py --si all",
                "python simulations/build_submission.py"):
        r = subprocess.run(cmd, cwd=ROOT, shell=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
        out = r.stdout.decode("utf-8", "replace")
        if r.returncode != 0:
            print(out[-1500:])
            raise SystemExit(f"[failed] {cmd}")
        tail = [l for l in out.strip().split("\n") if "pages" in l or "written" in l]
        print("   " + " | ".join(x.strip() for x in tail[-3:]))

    left = []
    for rel in TARGETS:
        t = io.open(os.path.join(ROOT, rel), encoding="utf-8").read()
        if "XXXXXXX" in t:
            left.append(rel)
    print("[check] placeholder still present in: " + (", ".join(left) if left else "nowhere"))


if __name__ == "__main__":
    main()
