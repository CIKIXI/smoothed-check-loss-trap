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
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "test_template")

# TEST's own description of the abstract is a short unstructured summary; Springer's template
# asks for it without citations and without displayed equations, so this is a separate text.
ABSTRACT = r"""Quantile regression with covariate measurement error is well understood in
fixed dimension, where corrected scores give consistent estimators, and penalised
high-dimensional versions have been proposed. What is missing is a theory for the shortcut
that remains common in practice, and for calibration-based inference when covariates outnumber
observations. The first half closes that gap with a negative result: convolving the check loss
with the measurement error distribution is not a correction. At the true coefficient vector
the population score of the convolved loss is a fixed non-zero multiple of the design error
for every kernel with positive density at zero; under a Gaussian design the smoothed loss is
minimised exactly at the attenuated value that naive quantile regression already targets,
whatever the bandwidth, so the two estimators share their limit. The second half gives the
repair: conditioning on the observed covariate yields a calibrated regressor under which
ordinary quantile regression is correctly specified and its score exactly mean-zero, so
penalised estimation and debiased inference become available. We give oracle inequalities when
the covariates outnumber the observations, first for a known calibration and then for an
estimated one, prove asymptotic normality of the debiased estimator with the
oracle-calibration variance under an explicit condition, and show that the sample-covariance
plug-in does not meet it. Simulations measure the coverage, and an application to NHANES
2017-2018, where the measurement-error covariance is estimated from two dietary recalls rather
than assumed, changes coefficient magnitudes several-fold."""

PREAMBLE_HEAD = r"""%% Submission to TEST (Springer, for the Spanish Society of Statistics and Operations
%% Research), built from paper/main.tex by simulations/build_test.py -- do not edit by hand.
\documentclass[pdflatex,sn-basic]{sn-jnl}% author-year references, TEST's convention

%% Page geometry. The class sets a binding offset that shifts the text block
%%%% sideways on facing pages; for a PDF that is read on screen we centre it, so
%%%% that consecutive pages line up. Text width and height are unchanged.
\geometry{twoside=false,hcentering=true,bindingoffset=0pt}

%%%% The class leaves a full em of vertical space between bibliography entries, which pushes
%%%% the reference list onto an extra page; a tighter list keeps the paper inside TEST's
%%%% 20-page limit without dropping a single reference.
\setlength{\bibsep}{0.35em}

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
University}, \orgaddress{\street{Guilin 541006}, \state{Guangxi}, \country{China}}.
ORCID: 0009-0001-4999-6203}

\abstract{%%ABSTRACT%%}

\keywords{Measurement error, quantile regression, regression calibration, corrected score,
smoothed check loss, debiased inference}

%% TEST asks for an appropriate number of MSC codes. The class's \pacs macro takes the label
%% as its optional argument, so it prints as "MSC codes". The same codes go into Editorial
%% Manager, which is where Mathematical Reviews and Zentralblatt pick them up.
\pacs[MSC codes]{62J05 (primary), 62G08, 62G20, 62J07, 62H12}

\maketitle
"""

SI_FRONTMATTER = r"""%% Online Resource 1 (ESM_1.pdf) for the TEST submission, built by simulations/build_test.py.
\documentclass[pdflatex,sn-basic]{sn-jnl}
\geometry{twoside=false,hcentering=true,bindingoffset=0pt}% same centred block as the paper
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

%% Springer asks every supplementary file to carry the article title, the journal, the
%% author, the affiliation and the e-mail address of the corresponding author.
\title[Online Resource 1]{Online Resource 1: Supplementary material for ``The
Smoothed-Check-Loss Trap: Why Convolution-Based Corrections Fail in Quantile Regression with
Measurement Error''}

\author*[1]{\fnm{Kaixu} \sur{Cai}}\email{1006282498@qq.com}

\affil*[1]{\orgdiv{School of Mathematics and Statistics}, \orgname{Guangxi Normal
University}, \orgaddress{\street{Guilin 541006}, \state{Guangxi}, \country{China}}.
ORCID: 0009-0001-4999-6203}

\abstract{Appendix A gives the proofs of all results stated in the paper. Appendix B collects
the computational details of the population targets, the quadrature used for the loss curves and
the design of the simulation study. Appendix C reports the numerical studies that support the
theory but are not needed for the argument of the paper. Appendix D is the reproducibility
inventory: the scripts, the independent verification scripts and the archived locations of the
code and results.}

\keywords{Supplementary material: proofs, computational details, reproducibility}

\maketitle
"""

# Springer numbers the tables and figures of a supplementary file S1, S2, ..., so that they
# cannot be confused with the tables and figures of the paper.
SI_NUMBERING = r"""
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
"""

BACKMATTER = r"""\backmatter

\bmhead{Acknowledgments}
The author gratefully acknowledges the developers of the NumPy, SciPy and scikit-learn
packages, which were used for all computations reported here, and the US Centers for Disease
Control and Prevention for making the NHANES public-use data available.

\bmhead{Statements and Declarations}

\noindent Funding: this work was not supported by any funding agency. Competing interests:
the author declares no competing interests. Ethics approval: not applicable, as the study
uses publicly available, de-identified secondary data; consent to participate and consent
for publication are likewise not applicable. Data availability: all data are publicly
available, the simulations using synthetic Gaussian covariates and known Gaussian
measurement error and the application using the NHANES 2017--2018 public-use demographic,
body-measure and two 24-hour dietary-recall files archived at
\texttt{https://wwwn.cdc.gov/nchs/nhanes/}. Code availability: the scripts that produce
every table and figure, with their saved output and the verification scripts, are archived
at \texttt{https://doi.org/10.5281/zenodo.23175827} (repository:
\texttt{https://github.com/CIKIXI/smoothed-check-loss-trap}). Authors' contributions: Kaixu
Cai is the sole author and carried out all parts of the work.
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
    # the appendix material travels to the SI with the same figure paths
    proofs = re.sub(r"\.\./figures/([A-Za-z0-9_]+)\.pdf", r"figures/\1.pdf", proofs)
    rest = re.sub(r"\.\./figures/([A-Za-z0-9_]+)\.pdf", r"figures/\1.pdf", rest)

    # Cross-references to appendices that leave the paper. Springer asks for supplementary
    # files to be cited as "Online Resource", with the file itself named ESM_<n>.
    def retarget(text, moved, label, letter):
        if label not in moved:
            return text
        tail = r"Appendix~" + letter + " of Online Resource~1"
        text = text.replace(r"\S\ref{" + label + "}", tail)          # "(\S\ref{...}"
        text = text.replace(r"Appendix~\ref{" + label + "}", tail)
        text = text.replace(r"\ref{" + label + "}", letter)          # anything left bare
        return text

    # Appendix letters in the order they appear in paper/main.tex. Anything that leaves the
    # paper is cited from the main text as "Appendix <letter> of Online Resource 1".
    APPENDICES = [("app:proofs", "A"), ("app:computation", "B"), ("app:numerics", "C"),
                  ("app:repro", "D")]
    moved = {"app:proofs": args.si == "all", "app:computation": args.si in ("all", "proofs"),
             "app:numerics": args.si in ("all", "proofs"),
             "app:repro": args.si in ("all", "proofs")}
    for label, letter in APPENDICES:
        body = retarget(body, moved, label, letter)

    keep = []
    if not moved["app:proofs"]:
        keep.append(proofs)
    if not moved["app:computation"]:
        keep.append(rest)
    appendix_text = "\n\n".join(keep)
    backmatter = BACKMATTER
    for label, letter in APPENDICES:
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
            SI_FRONTMATTER + "\n\\begin{appendices}\n" + prefix + SI_NUMBERING + si_body
            + "\n\\end{appendices}\n\n"
            # the appendices cite the literature too, so the file carries its own list
            # (the class already sets the bibliography style)
            + "\\bibliography{references}\n\n\\end{document}\n")
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
        print("[compile] supplementary.tex (after the main file, for xr): pdflatex + bibtex + x2")
        run("pdflatex -interaction=nonstopmode -halt-on-error supplementary.tex", src)
        run("bibtex supplementary", src)
        run("pdflatex -interaction=nonstopmode -halt-on-error supplementary.tex", src)
        run("pdflatex -interaction=nonstopmode -halt-on-error supplementary.tex", src)

    shutil.copyfile(os.path.join(ROOT, "paper", "cover_letter_test.tex"),
                    os.path.join(src, "cover_letter_test.tex"))
    print("[compile] cover_letter_test.tex")
    run("pdflatex -interaction=nonstopmode -halt-on-error cover_letter_test.tex", src)

    # Springer asks supplementary files to be named ESM_<n>; the SI is Online Resource 1.
    renamed = {"main_test.pdf": "main_test.pdf", "supplementary.pdf": "ESM_1.pdf",
               "cover_letter_test.pdf": "cover_letter.pdf"}
    for name, dest in renamed.items():
        p = os.path.join(src, name)
        if os.path.exists(p):
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
    si_pages = (count_pages(os.path.join(pkg, "ESM_1.pdf"))
                if os.path.exists(os.path.join(pkg, "ESM_1.pdf")) else 0)
    with io.open(os.path.join(ROOT, "paper", "references.bib"), encoding="utf-8") as fh:
        nrefs = len(re.findall(r"^@", fh.read(), re.M))

    # Springer requires the editable sources at every submission ("failing to submit a
    # complete set of editable source files will result in your article not being considered
    # for review"), so they travel as one archive next to the PDFs.
    zip_path = os.path.join(pkg, "latex_source.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _dirs, files in os.walk(src):
            for f in sorted(files):
                full = os.path.join(root, f)
                z.write(full, os.path.relpath(full, src))

    print(f"\n[package] {args.out}")
    print(f"  main_test.pdf     {pages} pages")
    if si_pages:
        print(f"  ESM_1.pdf         {si_pages} pages (Online Resource 1)")
    print(f"  cover_letter.pdf  {cover_pages} page")
    print(f"  latex_source.zip  {round(os.path.getsize(zip_path) / 1024)} KB (editable sources)")
    print(f"  source/           main_test.tex, supplementary.tex, references.bib ({nrefs} "
          f"entries), sn-jnl.cls, sn-basic.bst, {len(figs)} figures")
    return dict(pages=pages, si_pages=si_pages)


if __name__ == "__main__":
    main()
