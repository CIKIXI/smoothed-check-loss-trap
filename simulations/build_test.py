r"""
Build the TEST (Springer, SEIO) submission package from paper/main.tex.

TEST is published by Springer for the Spanish Society of Statistics and Operations Research
(SEIO).  Submissions go through Editorial Manager (https://www.editorialmanager.com/seio/);
the journal is not open access, so publishing in the subscription model is free of charge.
The manuscript is prepared with the Springer Nature LaTeX class (sn-jnl) and the author-year
reference style (sn-basic), which is TEST's convention, with the front matter Springer asks
for: title, authors and affiliations, abstract, keywords, then Declarations in the back
matter.

Where the appendices go is a build option:

    python simulations/build_test.py --si all      all three appendices as supplementary
                                                   material (main text 39 pages + 5-page SI)
    python simulations/build_test.py --si proofs   Appendix A stays in the paper, B and C
                                                   move to the supplementary material
    python simulations/build_test.py --si none     everything in the paper (44 pages)

The supplementary document is compiled with the `xr` package, so its cross-references to
equations, tables and propositions of the main text keep the main text's numbers.

Output:

  submission_TEST/            (or --out DIR)
      main_test.pdf           the manuscript
      supplementary.pdf       the supplementary material, when --si is not none
      cover_letter.pdf        the cover letter
      README.md, How_to_submit_to_TEST.md
      source/                 main_test.tex, supplementary.tex, references.bib, sn-jnl.cls,
                              sn-basic.bst, figures/*.pdf, the .bbl

Run from the repository root.
"""

import argparse
import io
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "test_template")

# TEST's own description of the abstract is a short unstructured summary; Springer's template
# asks for it without citations and without displayed equations, so this is a separate text.
ABSTRACT = r"""Quantile regression with covariate measurement error is well understood in
fixed dimension, where corrected scores and conditional estimating equations give consistent
estimators, and penalised high-dimensional versions of those routes have been proposed. What
is missing is a theory for the shortcut that is nonetheless common in practice, and for
calibration-based inference when the covariates outnumber the observations. The first half of
this paper closes the first gap with a negative result: convolving the check loss with the
measurement error distribution is not a correction. At the true coefficient vector the
population score of the convolved loss is a fixed non-zero multiple of the design error, for
every kernel with positive density at zero, and under a Gaussian design the population
minimiser of the smoothed loss is exactly the attenuated value that naive quantile regression
already targets, whatever the bandwidth. The smoothed estimator therefore shares its limit
with the naive one. The second half gives the repair: conditioning on the observed covariate
produces a calibrated regressor under which ordinary quantile regression is correctly
specified and its score is exactly mean-zero, so penalised estimation and debiased inference
become available. We give oracle inequalities in the regime where the covariates outnumber
the observations, first for a known calibration and then for an estimated one, prove that the
debiased estimator is asymptotically normal with the oracle-calibration variance under an
explicit condition on the accuracy of the calibration, and show that the sample-covariance
plug-in does not meet that condition while a known calibration does. The behaviour of the
intervals is measured in simulations, and the correction is applied to NHANES 2017-2018,
where the measurement-error covariance is estimated from two dietary recalls rather than
assumed and the correction changes coefficient magnitudes severalfold."""

PREAMBLE_HEAD = r"""%% Submission to TEST (Springer, for the Spanish Society of Statistics and Operations
%% Research), built from paper/main.tex by simulations/build_test.py -- do not edit by hand.
\documentclass[pdflatex,sn-basic]{sn-jnl}% author-year references, TEST's convention

%%%% Standard packages
\usepackage{graphicx}
\usepackage{multirow}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{amsthm}
\usepackage{mathrsfs}
\usepackage[title]{appendix}
\usepackage{xcolor}
\usepackage{textcomp}
\usepackage{manyfoot}
\usepackage{booktabs}
\usepackage{geometry}
"""

DEFS = r"""
%%%% Theorem environments, as in the manuscript
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{assumption}{Assumption}[section]
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{remark}[theorem]{Remark}

%%%% Notation of the manuscript
\newcommand{\R}{\mathbb{R}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\Pbb}{\mathbb{P}}
\newcommand{\cN}{\mathcal{N}}
\newcommand{\cC}{\mathcal{C}}
\newcommand{\cS}{\mathcal{S}}
\newcommand{\cH}{\mathcal{H}}
\newcommand{\cB}{\mathcal{B}}
\newcommand{\ind}[1]{\mathbf{1}\{#1\}}
\newcommand{\norm}[1]{\|#1\|}
\newcommand{\abs}[1]{|#1|}
\newcommand{\supp}{\mathrm{supp}}
\newcommand{\argmin}{\mathop{\mathrm{argmin}}}
\newcommand{\rht}{\tilde{\rho}_\tau}
\newcommand{\psit}{\psi_\tau}
\newcommand{\betah}{\hat{\beta}}
\newcommand{\betas}{\beta^*}
\newcommand{\sigmah}{\hat{\sigma}}
\newcommand{\bbar}{\bar{\beta}}
\newcommand{\muv}{\mu_W}
\newcommand{\Sxw}{\Sigma_{x|w}}
\newcommand{\Ccal}{C}

%% A slightly looser third pass keeps long inline formulas inside the text block.
\setlength{\emergencystretch}{1em}
"""

FRONTMATTER = r"""
\begin{document}

\title[The smoothed-check-loss trap]{The Smoothed-Check-Loss Trap: Why Convolution-Based
Corrections Fail in Quantile Regression with Measurement Error}

\author*[1]{\fnm{Kaixu} \sur{Cai}}\email{1006282498@qq.com}

\affil*[1]{\orgdiv{School of Mathematics and Statistics}, \orgname{Guangxi Normal
University}, \orgaddress{\street{Guilin 541006}, \state{Guangxi}, \country{China}}}

\abstract{%%ABSTRACT%%}

\keywords{Measurement error, quantile regression, regression calibration, corrected score,
smoothed check loss, errors-in-variables, debiased inference}

\maketitle
"""

SI_FRONTMATTER = r"""%% Supplementary material for the TEST submission, built by simulations/build_test.py.
\documentclass[pdflatex,sn-basic]{sn-jnl}
\usepackage{graphicx}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{amsthm}
\usepackage{mathrsfs}
\usepackage[title]{appendix}
\usepackage{xcolor}
\usepackage{textcomp}
\usepackage{manyfoot}
\usepackage{booktabs}
\usepackage{xr}% reads main_test.aux, so cross-references keep the main text's numbers
\externaldocument{main_test}
""" + DEFS + r"""
\begin{document}

\title[Supplementary material]{Supplementary material for ``The Smoothed-Check-Loss Trap: Why
Convolution-Based Corrections Fail in Quantile Regression with Measurement Error''}

\author*[1]{\fnm{Kaixu} \sur{Cai}}\email{1006282498@qq.com}

\affil*[1]{\orgdiv{School of Mathematics and Statistics}, \orgname{Guangxi Normal
University}, \orgaddress{\street{Guilin 541006}, \state{Guangxi}, \country{China}}}

\maketitle
"""

BACKMATTER = r"""
\backmatter

\bmhead{Acknowledgments}
The author gratefully acknowledges the developers of the NumPy, SciPy and scikit-learn
packages, which were used for all computations reported here, and the US Centers for Disease
Control and Prevention for making the NHANES public-use data available.

\section*{Declarations}

\begin{itemize}
\item Funding: This work was not supported by any funding agency.
\item Competing interests: The author declares no competing interests.
\item Ethics approval: Not applicable; the study uses publicly available, de-identified
secondary data.
\item Consent to participate: Not applicable.
\item Consent for publication: Not applicable.
\item Availability of data and materials: All data are publicly available. The simulation
studies use synthetic Gaussian covariates and known Gaussian measurement error and need no
external data. The application uses the NHANES 2017--2018 public-use demographic,
body-measure and two 24-hour dietary-recall files, archived by the US Centers for Disease
Control and Prevention at \texttt{https://wwwn.cdc.gov/nchs/nhanes/}.
\item Code availability: The scripts that produce every table and figure, together with their
saved output and the verification scripts, are archived in a version-controlled repository and
on Zenodo: \texttt{https://doi.org/10.5281/zenodo.23175827} (repository:
\texttt{https://github.com/CIKIXI/smoothed-check-loss-trap}).
\item Authors' contributions: Kaixu Cai is the sole author and carried out all parts of the
work.
\end{itemize}
"""


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, shell=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    out = r.stdout.decode("utf-8", "replace")
    if r.returncode != 0:
        print(out[-3000:])
        raise SystemExit(f"[failed] {cmd} (exit {r.returncode})")
    return out


def count_pages(pdf):
    import fitz
    with fitz.open(pdf) as d:
        return d.page_count


def split_appendices(body):
    """Return (body, proofs, computation_and_repro) with the appendix environment removed."""
    m = re.search(r"\\begin\{appendices\}(.*?)\\end\{appendices\}", body, re.S)
    if not m:
        return body, "", ""
    inside = m.group(1)
    k = inside.index(r"\section{Computation}")
    return body[:m.start()].rstrip() + "\n", inside[:k].strip(), inside[k:].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--si", choices=("all", "proofs", "none"), default="all",
                    help="which appendices move to the supplementary material (default: all)")
    ap.add_argument("--out", default="submission_TEST", help="output package directory")
    args = ap.parse_args()

    pkg = os.path.join(ROOT, args.out)
    src = os.path.join(pkg, "source")
    os.makedirs(src, exist_ok=True)

    tex_src = io.open(os.path.join(ROOT, "paper", "main.tex"), encoding="utf-8").read()
    body = tex_src[tex_src.index(r"\begin{document}"):]
    body = body[body.index("\\maketitle") + len("\\maketitle"):]
    body = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "", body, flags=re.S)
    body = re.sub(r"\\noindent\\textbf\{Keywords:\}.*?(?=\n\n)", "", body, flags=re.S)
    body = re.sub(r"\\medskip\s*\\noindent\\textbf\{MSC 2020.*?(?=\n\n)", "", body, flags=re.S)
    body = re.sub(r"\\medskip\s*\\noindent\\textbf\{Running title\.\}.*?(?=\n\n)", "",
                  body, flags=re.S)
    body = re.sub(r"\\section\*\{Code and data availability\}.*?(?=\\clearpage)", "", body,
                  flags=re.S)
    body = re.sub(r"\\section\*\{Acknowledgments and funding\}.*?(?=\\clearpage)", "", body,
                  flags=re.S)

    def inline(m):
        return io.open(os.path.join(ROOT, "paper", m.group(1)), encoding="utf-8").read().strip()
    body = re.sub(r"\\input\{(tables/[^}]+)\}", inline, body)
    body = body.replace(r"\appendix", r"\begin{appendices}")
    body = body.replace("\\clearpage\n\\bibliographystyle{plainnat}\n\\bibliography{references}",
                        "\\end{appendices}\n")
    body = body.replace(r"\bibliographystyle{plainnat}", "")
    body = body.replace(r"\bibliography{references}", "")
    body = body.replace(r"\end{document}", "")

    body, proofs, rest = split_appendices(body)

    # Figures: copy next to the source and point at the local copies.
    figdir = os.path.join(src, "figures")
    os.makedirs(figdir, exist_ok=True)
    figs = []
    for name in ("fig1_loss_curves", "fig2_scaling", "fig3_highdim"):
        shutil.copyfile(os.path.join(ROOT, "figures", name + ".pdf"),
                        os.path.join(figdir, name + ".pdf"))
        figs.append(name + ".pdf")
    body = re.sub(r"\.\./figures/([A-Za-z0-9_]+)\.pdf", r"figures/\1.pdf", body)

    # Cross-references to appendices that leave the paper.
    def retarget(text, moved, label, letter):
        if label not in moved:
            return text
        tail = r"Appendix~" + letter + " of the supplementary material"
        text = text.replace(r"\S\ref{" + label + "}", tail)          # "(\S\ref{...}"
        text = text.replace(r"Appendix~\ref{" + label + "}", tail)
        text = text.replace(r"\ref{" + label + "}", letter)          # anything left bare
        return text

    moved = {"app:proofs": args.si == "all", "app:computation": args.si in ("all", "proofs"),
             "app:repro": args.si in ("all", "proofs")}
    for label, letter in (("app:proofs", "A"), ("app:computation", "B"), ("app:repro", "C")):
        body = retarget(body, moved, label, letter)

    keep = []
    if not moved["app:proofs"]:
        keep.append(proofs)
    if not moved["app:computation"]:
        keep.append(rest)
    appendix_text = "\n\n".join(keep)
    backmatter = BACKMATTER
    for label, letter in (("app:proofs", "A"), ("app:computation", "B"), ("app:repro", "C")):
        backmatter = retarget(backmatter, moved, label, letter)
    main_body = (body + ("\n\\begin{appendices}\n" + appendix_text + "\n\\end{appendices}\n"
                         if appendix_text else "")
                 + backmatter + "\n\\bibliography{references}\n\n\\end{document}\n")

    tex = os.path.join(src, "main_test.tex")
    io.open(tex, "w", encoding="utf-8").write(
        PREAMBLE_HEAD + DEFS + FRONTMATTER.replace("%%ABSTRACT%%", ABSTRACT) + main_body)
    print(f"[written] {os.path.relpath(tex, ROOT)}")
    print(f"  --si {args.si}: appendix material in the paper = "
          f"{'A' if not moved['app:proofs'] else '-'}"
          f"{'B,C' if not moved['app:computation'] else ''}")

    # Supplementary material, when something moved.
    si_tex = os.path.join(src, "supplementary.tex")
    if args.si != "none":
        si_body = "\n".join(x for x in ((proofs if moved["app:proofs"] else ""),
                                        (rest if moved["app:computation"] else "")) if x)
        prefix = "\\setcounter{section}{1}%\n" if not moved["app:proofs"] else ""
        io.open(si_tex, "w", encoding="utf-8").write(
            SI_FRONTMATTER + "\n\\begin{appendices}\n" + prefix + si_body
            + "\n\\end{appendices}\n\n\\end{document}\n")
        print(f"[written] {os.path.relpath(si_tex, ROOT)}")

    for name in ("sn-jnl.cls", "sn-basic.bst"):
        shutil.copyfile(os.path.join(TEMPLATE, name), os.path.join(src, name))
    shutil.copyfile(os.path.join(ROOT, "paper", "references.bib"),
                    os.path.join(src, "references.bib"))

    print("[compile] main_test.tex: pdflatex + bibtex + pdflatex x2")
    run("pdflatex -interaction=nonstopmode -halt-on-error main_test.tex", src)
    run("bibtex main_test", src)
    run("pdflatex -interaction=nonstopmode -halt-on-error main_test.tex", src)
    run("pdflatex -interaction=nonstopmode -halt-on-error main_test.tex", src)
    if args.si != "none":
        print("[compile] supplementary.tex (after the main file, for xr): pdflatex x2")
        run("pdflatex -interaction=nonstopmode -halt-on-error supplementary.tex", src)
        run("pdflatex -interaction=nonstopmode -halt-on-error supplementary.tex", src)

    shutil.copyfile(os.path.join(ROOT, "paper", "cover_letter_test.tex"),
                    os.path.join(src, "cover_letter_test.tex"))
    print("[compile] cover_letter_test.tex")
    run("pdflatex -interaction=nonstopmode -halt-on-error cover_letter_test.tex", src)

    for name in ("main_test.pdf", "supplementary.pdf", "cover_letter_test.pdf"):
        p = os.path.join(src, name)
        if os.path.exists(p):
            dest = "cover_letter.pdf" if name == "cover_letter_test.pdf" else name
            shutil.copyfile(p, os.path.join(pkg, dest))
            os.remove(p)
    for f, dest in (("main_test.log", "review/test_build.log"),
                    ("supplementary.log", "review/test_supplementary_build.log"),
                    ("cover_letter_test.log", "review/test_cover_letter_build.log")):
        if os.path.exists(os.path.join(src, f)):
            shutil.move(os.path.join(src, f), os.path.join(ROOT, dest))
    for f in sorted(os.listdir(src)):
        if f.endswith((".aux", ".log", ".blg", ".out", ".toc")):
            os.remove(os.path.join(src, f))

    pages = count_pages(os.path.join(pkg, "main_test.pdf"))
    cover_pages = count_pages(os.path.join(pkg, "cover_letter.pdf"))
    si_pages = (count_pages(os.path.join(pkg, "supplementary.pdf"))
                if os.path.exists(os.path.join(pkg, "supplementary.pdf")) else 0)
    with io.open(os.path.join(ROOT, "paper", "references.bib"), encoding="utf-8") as fh:
        nrefs = len(re.findall(r"^@", fh.read(), re.M))
    print(f"\n[package] {args.out}")
    print(f"  main_test.pdf     {pages} pages")
    if si_pages:
        print(f"  supplementary.pdf {si_pages} pages")
    print(f"  cover_letter.pdf  {cover_pages} page")
    print(f"  source/           main_test.tex, supplementary.tex, references.bib ({nrefs} "
          f"entries), sn-jnl.cls, sn-basic.bst, {len(figs)} figures")
    return dict(pages=pages, si_pages=si_pages)


if __name__ == "__main__":
    main()
