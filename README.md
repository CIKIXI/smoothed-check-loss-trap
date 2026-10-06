# Code and results for "The Smoothed-Check-Loss Trap"

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
