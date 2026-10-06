r"""
Assemble the code release for the paper and set it up as a push-ready Git repository.

The release is written to code_release/ and contains

    README.md            what is here, how to reproduce every table and figure, runtimes
    LICENSE              MIT
    .gitignore           build junk, data downloads, caches
    CITATION.cff         citation metadata with the repository URL
    requirements.txt     Python packages
    simulations/         the 17 scripts the paper's appendix names, plus the 12 JSON results
    review/              the independent checks and the numeric audit
    paper/               manuscript sources, table files, both document classes
    figures/             the three vector figures
    data/nhanes/         where the four NHANES files go (with download URLs)
    docs/PUSH_AND_DOI.md how to publish it and mint the Zenodo DOI
    docs/ZENODO_RELEASE.md  why the DOI is needed and what to check

and is zipped to submission_TEST/code_and_data.zip for the supplementary-material upload.

Run from the repository root:  python simulations/make_code_release.py [--with-data]
"""

import io
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "code_release")
GITHUB_ACCOUNT = "CIKIXI"
REPO_NAME = "smoothed-check-loss-trap"
REPO_URL = f"https://github.com/{GITHUB_ACCOUNT}/{REPO_NAME}"
DOI = "10.5281/zenodo.23175827"          # concept DOI, minted 2026-10-06

SCRIPTS = [
    "routeA_lib.py", "routeA_targets.py", "routeA_experiments.py", "routeA_inference.py",
    "hdc_experiments.py", "hdc_gls_theory.py", "hdc_debias.py", "hdc_debias_final.py",
    "hdc_pgtn.py", "hdc_debias_diag.py", "hdc_debias_fix.py", "nhanes_realdata.py",
    "routeA_tables.py", "routeA_figures.py", "consistency_audit.py",
    "build_test.py", "make_code_release.py", "set_zenodo_doi.py",
]

CHECKS = [
    "verify_theory_full.py", "verify_constants_thm54.py", "verify_knight_identity.py",
    "verify_posterior_adaptive.py", "verify_prop313_edgecases.py", "verify_exact_targets.py",
    "verify_exact_targets2.py", "verify_expansion_vs_exact.py", "verify_fixed_scale_target.py",
    "verify_error_law_M6.py", "verify_fig1_exact_curves.py", "verify_plugin_calibration_bias.py",
    "verify_plugin_bias_scaling.py", "verify_plugin_bias_stage.py", "verify_plugin_score_mean.py",
    "verify_debias_determinism.py", "check_table1_deviation.py", "layout_check.py",
    "indent_paragraph_audit.py", "page_top_indent.py", "table_width_probe.py",
    "ai_style_check.py", "reference_audit_check.py",
]

README = r"""# Code and results for "The Smoothed-Check-Loss Trap"

Manuscript: *The Smoothed-Check-Loss Trap: Why Convolution-Based Corrections Fail in Quantile
Regression with Measurement Error*
Journal: TEST (Springer, for the Spanish Society of Statistics and Operations Research)
Author: Kaixu Cai, School of Mathematics and Statistics, Guangxi Normal University
Manuscript ID: to be added when the submission number is assigned
Archived release: https://doi.org/10.5281/zenodo.23175827
Repository: https://github.com/CIKIXI/smoothed-check-loss-trap

This repository contains the Python code, the saved results and the verification scripts behind
every table, figure and number in the manuscript. Nothing is taken on trust: the numbers typed
into the text are re-derived from the saved results by `simulations/consistency_audit.py`,
which prints a pass/fail table.

---

## 1. Requirements

- Python 3.12 (developed under 3.12.7)
- packages: `pip install -r requirements.txt`
  (numpy, scipy, scikit-learn, matplotlib, pandas, PyMuPDF)
- LaTeX with `pdflatex` and `bibtex`, only if the manuscripts are to be rebuilt

## 2. Repository structure

```
simulations/        the experiments and the table/figure generators, plus their JSON output
review/             independent checks of the theory and of the reported numbers
paper/              manuscript sources (main.tex, tables/, references.bib); the TEST
                    document class lives in test_template/
figures/            the three vector figures, as produced by simulations/routeA_figures.py
data/nhanes/        put the four NHANES 2017-2018 files here (see Section 5)
docs/               how to publish this repository and mint the Zenodo DOI
requirements.txt    Python packages
```

## 3. What produces what

| paper item | script | runs |
| --- | --- | --- |
| Tables 1-2, population targets | `simulations/routeA_targets.py` | ~2 min, exact (no optimiser) |
| Table 3, score check | `simulations/routeA_experiments.py` (`exp_E4`) | ~1 min |
| Tables 4-5, scaling and high dimensions | `simulations/routeA_experiments.py` (`exp_E2`, `exp_E3`) | ~2 min, ~10 min |
| Table 6, coverage | `simulations/routeA_inference.py` | ~5 min |
| Table 7, estimated calibration | `simulations/hdc_experiments.py` | ~10 min |
| Table 8, GLS gains | `simulations/hdc_gls_theory.py` | seconds, closed form |
| Table 9, debiasing (`p<n` and `p>n`) | `simulations/hdc_debias_final.py`, `simulations/hdc_pgtn.py` | ~6 min, ~4 min |
| Table 10, known versus plug-in calibration | `review/verify_plugin_calibration_bias.py`, `review/verify_plugin_bias_scaling.py` | ~4 min, ~6 min |
| Table 11, NHANES application | `simulations/nhanes_realdata.py` | ~1 min, needs Section 5 |
| Figures 1-3 | `simulations/routeA_figures.py` | seconds, vector PDF |
| all LaTeX tables | `simulations/routeA_tables.py` | seconds, reads the JSON results |
| the numeric audit | `simulations/consistency_audit.py` | ~1 min |
| the layout (overfull boxes, page alignment, page-top indents) | `review/layout_check.py`, `review/page_top_indent.py` | seconds |
| the manuscript | `simulations/build_test.py --si all` (TEST layout and supplement) | ~1 min |

Every experiment writes its raw output to `simulations/*.json`; those files are committed, so
the tables and figures can be regenerated without re-running the experiments.

## 4. How to reproduce

Run everything from the repository root, in this order.

```sh
pip install -r requirements.txt

# results (long-running; skip any whose JSON is already present)
python simulations/routeA_targets.py             # ~2 min
python simulations/routeA_experiments.py         # ~13 min
python simulations/routeA_inference.py           # ~5 min
python simulations/hdc_experiments.py            # ~10 min
python simulations/hdc_gls_theory.py             # seconds
python simulations/hdc_debias_final.py           # ~6 min
python simulations/hdc_pgtn.py                   # ~4 min
python review/verify_plugin_calibration_bias.py 200   # ~4 min
python review/verify_plugin_bias_scaling.py 100       # ~6 min
python simulations/nhanes_realdata.py            # ~1 min, after Section 5

# tables, figures and the audit
python simulations/routeA_tables.py              # paper/tables/*.tex
python simulations/routeA_figures.py             # figures/*.pdf
python simulations/consistency_audit.py          # expect: 66/66 claims consistent

# optional: rebuild the manuscript
python simulations/build_test.py --si all        # TEST layout -> submission_TEST/
```

About 45 minutes in total on a laptop. The random seeds are fixed, so repeated runs reproduce
the same numbers; `review/verify_debias_determinism.py` checks that the debiasing study is
bit-identical across processes.

## 5. Data

The application uses four public-use files from NHANES 2017-2018 (US Centers for Disease
Control and Prevention). Download

```
https://wwwn.cdc.gov/nchs/nhanes/2017-2018/DEMO_J.XPT
https://wwwn.cdc.gov/nchs/nhanes/2017-2018/BMX_J.XPT
https://wwwn.cdc.gov/nchs/nhanes/2017-2018/DR1TOT_J.XPT
https://wwwn.cdc.gov/nchs/nhanes/2017-2018/DR2TOT_J.XPT
```

into `data/nhanes/`. The two 24-hour recalls give genuine replicates, so the covariate
covariance and the measurement-error covariance are both estimated from the data and no
synthetic noise is added. The simulation studies need no external data.

## 6. Independent checks

`review/` holds the scripts that verify the theory numerically rather than by re-reading it:
Monte-Carlo and quadrature checks of the population identities, the constant chain in the
high-dimensional oracle inequality, the Knight identity used in the debiased-inference proof,
the exact population-loss curves of Figure 1, the calibration experiments behind Table 10, and
a determinism check. `review/adversarial/` contains an independent re-derivation of the
exact-target proposition, written against the statement alone and without access to the proof.

## 7. Licence and citation

Code released under the MIT licence (`LICENSE`); the NHANES files are public-use data
distributed by the US CDC and are not covered by it.

If you use this code or these results, please cite the archived release
(`CITATION.cff`), https://doi.org/10.5281/zenodo.23175827.
"""

LICENSE = """MIT License

Copyright (c) 2026 Kaixu Cai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

GITIGNORE = """# Python ---------------------------------------------------------------------
__pycache__/
*.pyc
.pytest_cache/
.ipynb_checkpoints/

# Data downloads --------------------------------------------------------------
# The four NHANES files are public and are fetched by the reader (Section 5 of the
# README).  They are 23 MB in total and are deliberately not committed.
data/nhanes/*.xpt

# LaTeX build products --------------------------------------------------------
# Only relevant if a manuscript copy is built inside this repository.
*.aux
*.log
*.blg
*.bbl
*.out
*.toc
*.synctex.gz

# OS / editor -----------------------------------------------------------------
.DS_Store
Thumbs.db
desktop.ini
*~
*.swp
.vscode/
.idea/
"""

CITATION = r"""cff-version: 1.2.0
message: "If you use this code or these results, please cite the archived release."
title: "Code and results for: The Smoothed-Check-Loss Trap: Why Convolution-Based Corrections Fail in Quantile Regression with Measurement Error"
type: software
authors:
  - family-names: Cai
    given-names: Kaixu
    affiliation: "School of Mathematics and Statistics, Guangxi Normal University, Guilin, China"
abstract: >-
  Python code, saved results and verification scripts for a study of covariate measurement
  error in quantile regression. Includes the exact population targets of the convolved,
  naive and calibrated estimators; the low-dimensional scaling and high-dimensional
  comparison studies; the coverage study for the calibrated estimator; the debiasing
  experiments for p smaller than n and p larger than n (including the known-calibration versus
  sample-covariance plug-in comparison); the NHANES 2017-2018 application; and the scripts that
  generate every table and vector figure of the manuscript.
license: MIT
version: "1.0.0"
date-released: "2026-10-06"
repository-code: "https://github.com/CIKIXI/smoothed-check-loss-trap"
keywords:
  - measurement error
  - quantile regression
  - regression calibration
  - high-dimensional inference
  - debiased inference
  - reproducibility
"""

PUSH_AND_DOI = r"""# Publish this repository (2 steps, then fill in the DOI)

This directory is already a **local Git repository with a `v1.0.0` tag**, and the remote is
configured as

```
origin  git@github.com:CIKIXI/smoothed-check-loss-trap.git      (SSH)
```

SSH authentication with your GitHub account already works on this machine
(`ssh -T git@github.com` answers `Hi CIKIXI!`), so pushing needs no password or token.
Everything that can be prepared offline has been prepared; what remains is creating the empty
repository on GitHub and minting the DOI.

---

## Step 1: create the empty GitHub repository and push

1. Open <https://github.com/new>.
2. Owner: **CIKIXI**. Repository name: **smoothed-check-loss-trap**. Visibility: **Public**.
3. Do **not** tick "Add a README file", ".gitignore" or "license": the repository must be
   empty, otherwise the push is rejected.
4. Create the repository, then in PowerShell:

   ```powershell
   cd C:\Users\liangxin2\Desktop\SCI1\code_release
   git push -u origin main
   git push origin v1.0.0
   ```

5. Check on GitHub: 92 files, and the tag `v1.0.0` listed under **Releases / Tags**.

If the repository name or account differs, point the remote at the real one first:

```powershell
git remote set-url origin git@github.com:<account>/<repo>.git
```

> **Commit identity.** The commit is authored as `Kaixu Cai
> <CIKIXI@users.noreply.github.com>`, the GitHub no-reply address of the `CIKIXI` account, so
> GitHub links the commit to your profile.

## Step 2: archive on Zenodo and mint the DOI

1. Sign in at <https://zenodo.org> with the GitHub account (button "Sign in with GitHub").
2. Go to **Settings → GitHub** (or <https://zenodo.org/account/settings/github/>) and press
   **Sync now** if the new repository is not listed yet.
3. Flip the switch for **smoothed-check-loss-trap** to ON. Zenodo now watches it for releases.
4. On GitHub, open the repository and create a release from the tag:
   **Releases → Draft a new release → Choose a tag: v1.0.0 → Release title** (e.g. "Code and
   results for the TEST submission") **→ Publish release**.
5. Refresh the Zenodo GitHub page: a new entry appears with a DOI. Two DOIs are shown:
   - **concept DOI** — "Cite all versions"; always resolves to the newest version.
     **This is the one to put in the paper.**
   - **version DOI** — pins `v1.0.0`.
6. Open the Zenodo record and check the metadata (title, author Cai Kaixu, licence MIT,
   keywords). You can edit it; the files come from the release automatically.

## Step 3: put the real DOI into the manuscripts

```powershell
cd C:\Users\liangxin2\Desktop\SCI1
python simulations/set_zenodo_doi.py 10.5281/zenodo.1234567
```

The script writes the DOI into `paper/main.tex` and `simulations/build_test.py`, rebuilds both
submission packages, and reports whether any placeholder is left. To drop the DOI clause and
keep only the repository link, run it with `none` instead of a DOI.

## Step 4: sanity check before submitting

- [ ] Both URLs resolve in a private/incognito window: the GitHub repository and the Zenodo
      record (they must be genuinely public).
- [ ] The Zenodo record contains the scripts **and** the saved `simulations/*.json` results.
- [ ] The repository shows 92 files and the tag `v1.0.0`.
- [ ] `XXXXXXX` no longer appears in `paper/main.tex` or `simulations/build_test.py`.
- [ ] `submission_TEST/main_test.pdf` was rebuilt after the DOI change and shows the DOI in
      the Declarations.
- [ ] The DOI quoted in the paper is the **concept** DOI, not the version DOI.
"""

ZENODO_RELEASE = r"""# Minting the Zenodo DOI and linking it to the manuscript

> The short operational checklist is `PUSH_AND_DOI.md`; this file explains *why* the DOI is
> wanted and what to check afterwards.

The manuscript's Declarations will state:

> Code availability: The scripts that produce every table and figure, together with their saved
> output and the verification scripts, are archived in a version-controlled repository and on
> Zenodo: https://doi.org/10.5281/zenodo.23175827 (repository:
> https://github.com/CIKIXI/smoothed-check-loss-trap).

`XXXXXXX` is a placeholder and **must be replaced before submission**; a placeholder is no
better than "available on request".

## Why a DOI rather than a bare GitHub link

A GitHub URL alone is a moving target: the default branch can change, files can be renamed, and
the repository can be deleted. A Zenodo DOI is a frozen, citable snapshot of one release, which
is what a reader or a referee needs in order to check the paper years later. Both are given, so
the reader can browse the repository and cite the archive.

## What to check on the Zenodo record

| item | expected |
| --- | --- |
| Title | Code and results for: The Smoothed-Check-Loss Trap ... |
| Author | Cai, Kaixu (affiliation: Guangxi Normal University) |
| Licence | MIT |
| Version | v1.0.0 (from the Git tag) |
| Files | the whole tree: `simulations/`, `review/`, `paper/`, `figures/`, `README.md`, `LICENSE`, `CITATION.cff` |
| Keywords | measurement error, quantile regression, regression calibration, debiased inference |

## If the repository changes after the release

Mint a new version (v1.1.0) from a new tag; Zenodo gives it its own version DOI while the
**concept** DOI keeps resolving to the newest version. The paper cites the concept DOI, so it
stays correct without a correction notice.
"""


def main():
    # Keep the Git history if the release is rebuilt: everything except .git is regenerated.
    if os.path.exists(OUT):
        for name in os.listdir(OUT):
            if name == ".git":
                continue
            p = os.path.join(OUT, name)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    sim_out = os.path.join(OUT, "simulations")
    rev_out = os.path.join(OUT, "review")
    data_out = os.path.join(OUT, "data", "nhanes")
    docs_out = os.path.join(OUT, "docs")
    for d in (sim_out, rev_out, data_out, docs_out):
        os.makedirs(d, exist_ok=True)

    copied = 0
    for name in SCRIPTS:
        p = os.path.join(ROOT, "simulations", name)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(sim_out, name))
            copied += 1
        else:
            print(f"[warn] missing simulations/{name}")
    njson = 0
    for name in os.listdir(os.path.join(ROOT, "simulations")):
        if name.endswith(".json"):
            shutil.copyfile(os.path.join(ROOT, "simulations", name),
                            os.path.join(sim_out, name))
            njson += 1

    for name in CHECKS:
        p = os.path.join(ROOT, "review", name)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(rev_out, name))
        else:
            print(f"[warn] missing review/{name}")
    for name in os.listdir(os.path.join(ROOT, "review")):
        if name.endswith(".json"):
            shutil.copyfile(os.path.join(ROOT, "review", name),
                            os.path.join(rev_out, name))
    for name in ("reference_audit_v2.md", "reference_audit_v2.json"):
        p = os.path.join(ROOT, "review", name)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(rev_out, name))
    adv_src = os.path.join(ROOT, "review", "adversarial")
    if os.path.isdir(adv_src):
        shutil.copytree(adv_src, os.path.join(rev_out, "adversarial"),
                        ignore=shutil.ignore_patterns("results", "__pycache__"))

    # manuscript sources and the classes both builds need, so that the audit and the LaTeX
    # builds run from the release as well
    paper_out = os.path.join(OUT, "paper")
    os.makedirs(os.path.join(paper_out, "tables"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "test_template"), exist_ok=True)
    for name in ("main.tex", "references.bib", "cover_letter_test.tex"):
        p = os.path.join(ROOT, "paper", name)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(paper_out, name))
    for name in os.listdir(os.path.join(ROOT, "paper", "tables")):
        shutil.copyfile(os.path.join(ROOT, "paper", "tables", name),
                        os.path.join(paper_out, "tables", name))
    for name in ("sn-jnl.cls", "sn-basic.bst"):
        p = os.path.join(ROOT, "test_template", name)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(OUT, "test_template", name))
    for name in os.listdir(os.path.join(ROOT, "figures")):
        if name.endswith(".pdf") and name.startswith("fig"):
            shutil.copyfile(os.path.join(ROOT, "figures", name),
                            os.path.join(OUT, "figures", name))

    io.open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(README)
    io.open(os.path.join(OUT, "LICENSE"), "w", encoding="utf-8").write(LICENSE)
    io.open(os.path.join(OUT, ".gitignore"), "w", encoding="utf-8").write(GITIGNORE)
    io.open(os.path.join(OUT, "CITATION.cff"), "w", encoding="utf-8").write(CITATION)
    io.open(os.path.join(OUT, "requirements.txt"), "w", encoding="utf-8").write(
        "numpy>=1.26,<2\nscipy>=1.11\nscikit-learn>=1.3\nmatplotlib>=3.8\npandas>=2.0\n"
        "PyMuPDF>=1.23\n")
    io.open(os.path.join(docs_out, "PUSH_AND_DOI.md"), "w", encoding="utf-8").write(PUSH_AND_DOI)
    io.open(os.path.join(docs_out, "ZENODO_RELEASE.md"), "w",
            encoding="utf-8").write(ZENODO_RELEASE)
    io.open(os.path.join(data_out, "README.md"), "w", encoding="utf-8").write(
        "Put DEMO_J.xpt, BMX_J.xpt, DR1TOT_J.xpt and DR2TOT_J.xpt here; see Section 5 of the\n"
        "repository README for the download URLs.\n")
    if "--with-data" in sys.argv:
        src_data = os.path.join(ROOT, "data", "nhanes")
        for f in os.listdir(src_data):
            if f.lower().endswith(".xpt"):
                shutil.copyfile(os.path.join(src_data, f), os.path.join(data_out, f))
        print("[data] NHANES .xpt files included (--with-data)")

    # The archive is written outside OUT, otherwise the zip ends up inside itself; the Git
    # directory is skipped, since the archive is a snapshot of the working tree only.
    import zipfile
    zip_dir = os.path.join(ROOT, "submission_TEST")
    os.makedirs(zip_dir, exist_ok=True)
    zip_path = os.path.join(zip_dir, "code_and_data.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(OUT):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
            for f in files:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, OUT))

    n_review = len([f for f in os.listdir(rev_out)])
    n_paper = sum(len(files) for _, _, files in os.walk(paper_out))
    size = os.path.getsize(zip_path) / 1024
    print(f"[release] {os.path.relpath(OUT, ROOT)}")
    print(f"  simulations/  {copied} scripts + {njson} JSON results")
    print(f"  review/       {n_review} files")
    print(f"  paper/        {n_paper} files (sources, tables, classes)")
    print(f"  docs/         PUSH_AND_DOI.md, ZENODO_RELEASE.md")
    print(f"  README.md, LICENSE, .gitignore, CITATION.cff, requirements.txt")
    print(f"  zip: {os.path.relpath(zip_path, ROOT)} ({size:.0f} KB)")
    print(f"  push to: git@github.com:{GITHUB_ACCOUNT}/{REPO_NAME}.git  (browse: {REPO_URL})")


if __name__ == "__main__":
    main()
