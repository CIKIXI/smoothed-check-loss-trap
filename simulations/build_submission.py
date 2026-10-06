r"""
Build the IMS-formatted submission file from paper/main.tex.

main.tex is the development source: it uses the `article` class and \input's the generated
tables.  The IMS (Annals of Statistics) class requires a single .tex file, so this script

  * swaps the preamble for the imsart AoS preamble (class, packages, theorem environments),
  * converts the title/author/abstract/keywords block into \begin{frontmatter} form,
  * inlines every \input{tables/*.tex},
  * converts \appendix into \begin{appendix}...\end{appendix},
  * switches the bibliography style to spr-ims-nameyear (author-year, matching \citet),
  * copies the IMS class support files from paper/texclass/ next to the output,
  * writes submission_AoS/source/main_aos.tex (the dedicated submission package folder;
   use simulations/make_submission_folder.py to build and compile the whole package).

Run from the repository root:  python simulations/build_submission.py
"""

import io
import os
import re
import shutil

SRC = "paper/main.tex"
OUTDIR = "submission_AoS/source"
OUT = os.path.join(OUTDIR, "main_aos.tex")
# class/package/bst files the AoS build needs; shipped with the submission so that it
# compiles without a full TeX Live installation (grfext.sty and textcase.sty are small
# stand-ins for packages that a minimal installation may lack -- see paper/texclass).
SUPPORT = ["imsart.cls", "imsart.sty", "spr-ims-nameyear.bst", "spr-ims-number.bst",
           "grfext.sty", "textcase.sty", "aos-template.tex"]

PREAMBLE = r"""%% Submission to The Annals of Statistics (IMS), built from paper/main.tex by
%% simulations/build_submission.py -- do not edit by hand.
\documentclass[aos]{imsart}

\RequirePackage{amsthm,amsmath,amsfonts,amssymb}
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{bm}
\RequirePackage{graphicx}
\RequirePackage{booktabs}
\RequirePackage{multirow}
\RequirePackage[round,authoryear]{natbib}
%% The manuscript indents the first paragraph after every heading; the IMS class leaves it
%% flush (LaTeX's default).  This makes the two versions agree.
\RequirePackage{indentfirst}

%% A slightly looser third pass keeps long unbreakable items (URLs, inline formulas)
%% inside the text block of the narrower IMS layout.
\setlength{\emergencystretch}{1em}

%% The manuscript uses \paragraph as an unnumbered run-in heading, as in the development
%% file.  The IMS AoS layout numbers it (labels such as "6.0.0.1."), sets it as a displayed
%% heading and appends a further period, so define the run-in form directly.
\setcounter{secnumdepth}{2}
\makeatletter
\renewcommand\paragraph[1]{\par\addvspace{1.2ex plus .2ex}\noindent
    \textbf{#1}\hspace{0.6em}\ignorespaces}
\makeatother

\startlocaldefs
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{assumption}{Assumption}[section]
%% imsart makes \theoremstyle{remark} an error: its AoS layout wants
%% remark-like environments declared in the `definition' style (see aos-template.tex).
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{remark}[theorem]{Remark}

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
\endlocaldefs

\begin{document}

\begin{frontmatter}

\title{The Smoothed-Check-Loss Trap: Why Convolution-Based Corrections Fail in Quantile
Regression with Measurement Error}
\runtitle{The smoothed-check-loss trap}

\begin{aug}
\author[A]{\fnms{Kaixu}~\snm{Cai}\ead[label=e1]{1006282498@qq.com}}
\address[A]{School of Mathematics and Statistics, Guangxi Normal University,
Guilin 541006, China\printead[presep={,\ }]{e1}}
\end{aug}

\begin{abstract}
%%ABSTRACT%%
\end{abstract}

\begin{keyword}[class=MSC]
\kwdgroup[type=primary]{\kwd{62J05}}
\kwdgroup[type=secondary]{\kwd{62G08}\kwd{62G20}\kwd{62J07}\kwd{62H12}}
\end{keyword}

\begin{keyword}
\kwd{Measurement error}\kwd{quantile regression}\kwd{regression calibration}
\kwd{corrected score}\kwd{smoothed check loss}\kwd{errors-in-variables}
\kwd{debiased inference}
\end{keyword}

\end{frontmatter}
"""

# Back matter that the IMS template places after the appendices and before the
# bibliography.  There is no funding to report.
BACKMATTER = r"""
\begin{acks}[Acknowledgments]
The author gratefully acknowledges the developers of the NumPy, SciPy and
scikit-learn packages, which were used for all computations reported here, and the
US Centers for Disease Control and Prevention for making the NHANES public-use data
available. The author declares no conflicts of interest.
\end{acks}

\begin{funding}
This work was not supported by any funding agency.
\end{funding}

"""


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    src = io.open(SRC, encoding="utf-8").read()

    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S).group(1).strip()
    body = src[src.index(r"\begin{document}"):]
    body = body[body.index("\\maketitle") + len("\\maketitle"):]

    # drop the old abstract/keywords/MSC/running-title block
    body = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "", body, flags=re.S)
    body = re.sub(r"\\noindent\\textbf\{Keywords:\}.*?(?=\n\n)", "", body, flags=re.S)
    body = re.sub(r"\\medskip\s*\\noindent\\textbf\{MSC 2020.*?(?=\n\n)", "", body, flags=re.S)
    body = re.sub(r"\\medskip\s*\\noindent\\textbf\{Running title\.\}.*?(?=\n\n)", "", body, flags=re.S)

    # inline the generated tables (the IMS class wants a single file)
    def inline(m):
        path = os.path.join("paper", m.group(1))
        return io.open(path, encoding="utf-8").read().strip()
    body = re.sub(r"\\input\{(tables/[^}]+)\}", inline, body)

    # figures: the development source points at ../figures (relative to paper/); ship
    # copies next to the submission and prefer the vector (PDF) versions
    figdir = os.path.join(OUTDIR, "figures")
    os.makedirs(figdir, exist_ok=True)
    copied = []

    def swap_fig(m):
        name = m.group(1)
        pdf = os.path.join("figures", name + ".pdf")
        png = os.path.join("figures", name + ".png")
        if os.path.exists(pdf):
            shutil.copyfile(pdf, os.path.join(figdir, name + ".pdf"))
            copied.append(name + ".pdf")
            return "figures/" + name + ".pdf"
        shutil.copyfile(png, os.path.join(figdir, name + ".png"))
        copied.append(name + ".png")
        return "figures/" + name + ".png"

    body = re.sub(r"\.\./figures/([A-Za-z0-9_]+)\.(?:png|pdf)", swap_fig, body)
    # drop figure files left over from earlier builds so the folder ships only what is used
    for old in os.listdir(figdir):
        if old not in copied:
            os.remove(os.path.join(figdir, old))

    # appendix environment
    body = body.replace(r"\appendix", r"\begin{appendix}")
    # the development source carries its own acknowledgments/funding section; the IMS
    # back matter below replaces it, so drop it here to avoid printing it twice
    body = re.sub(r"\\section\*\{Acknowledgments and funding\}.*?(?=\\end\{document\})",
                  "", body, flags=re.S)
    # the development source ends with \clearpage + a plainnat bibliography; the IMS
    # layout wants a single bibliography after \end{appendix}, in the IMS style, so the
    # original commands are removed here and re-issued once at the very end
    body = body.replace("\\clearpage\n\\bibliographystyle{plainnat}\n"
                        "\\bibliography{references}", "")
    body = body.replace(r"\bibliographystyle{plainnat}", "")
    body = body.replace(r"\bibliography{references}", "")
    body = body.replace(r"\end{document}",
                        "\\end{appendix}\n\n" + BACKMATTER +
                        "\\bibliographystyle{spr-ims-nameyear}\n"
                        "\\bibliography{references}\n\n\\end{document}")

    out = PREAMBLE.replace("%%ABSTRACT%%", abstract) + "\n" + body
    io.open(OUT, "w", encoding="utf-8").write(out)
    for name in SUPPORT:
        s = os.path.join("paper", "texclass", name)
        if os.path.exists(s):
            shutil.copyfile(s, os.path.join(OUTDIR, name))
    if os.path.exists("paper/references.bib"):
        shutil.copyfile("paper/references.bib", os.path.join(OUTDIR, "references.bib"))
    print(f"[written] {OUT}  ({len(out)} chars)")
    print(f"  inlined {len(re.findall(r'begin.table', out))} tables")
    print(f"  appendix environment: {'begin{appendix}' in out}")
    print(f"  figures: {', '.join(copied) if copied else 'none'}")
    print(f"  support files copied: {', '.join(n for n in SUPPORT if os.path.exists(os.path.join('paper', 'texclass', n)))}")


if __name__ == "__main__":
    main()
