# Minting the Zenodo DOI and linking it to the manuscript

> The short operational checklist is `PUSH_AND_DOI.md`; this file explains *why* the DOI is
> wanted and what to check afterwards.

The manuscript's Declarations will state:

> Code availability: The scripts that produce every table and figure, together with their saved
> output and the verification scripts, are archived in a version-controlled repository and on
> Zenodo: https://doi.org/10.5281/zenodo.XXXXXXX (repository:
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
