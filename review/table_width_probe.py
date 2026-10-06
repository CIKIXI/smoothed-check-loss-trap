"""
Measure the natural width of every generated table.

Each paper/tables/tab_*.tex is a float whose body is a tabular.  This script extracts the
tabular part, boxes it in a probe document with a wide text block, and prints
\\wd of the box, so that the widths can be compared with the real text widths:

    development (article, 12pt, geometry)  451pt
    AoS (imsart aosa)                      404pt

Run:  python review/table_width_probe.py
"""

import glob
import io
import os
import re
import subprocess

OUT = "review/_table_probe"
os.makedirs(OUT, exist_ok=True)

LIMITS = {"AoS": 404.0, "development": 451.0}


def extract_tabular(path):
    txt = io.open(path, encoding="utf-8").read()
    m = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", txt, re.S)
    if not m:
        return None
    pre = txt[:m.start()]
    setup = ""
    for pat in (r"\\footnotesize", r"\\small", r"\\scriptsize", r"\\tiny",
                r"\\setlength\{\\tabcolsep\}\{[^}]*\}"):
        for mm in re.finditer(pat, pre):
            setup += mm.group(0) + "\n"
    return setup, m.group(0)


probe = [r"\documentclass[12pt]{article}",
         r"\usepackage[margin=1in,paperwidth=9in,paperheight=11in]{geometry}",
         r"\usepackage{amsmath,amssymb,booktabs,multirow}",
         r"\usepackage[T1]{fontenc}",
         r"\newsavebox{\probeBox}",
         r"\begin{document}", r"probe", r"\newpage"]
widths = {}
for path in sorted(glob.glob("paper/tables/*.tex")):
    got = extract_tabular(path)
    name = os.path.basename(path)[:-4]
    if not got:
        widths[name] = None
        continue
    setup, tabular = got
    probe.append(r"\sbox{\probeBox}{%" + "\n" + setup + tabular + "\n}")
    probe.append(r"\typeout{PROBEWIDTH " + name + r" \the\wd\probeBox\space\the\ht\probeBox}")
    probe.append(r"\newpage")
probe.append(r"\end{document}")

tex = os.path.join(OUT, "probe.tex")
io.open(tex, "w", encoding="utf-8").write("\n".join(probe))
r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "probe.tex"], cwd=OUT,
                   capture_output=True, text=True, errors="replace")
log = io.open(os.path.join(OUT, "probe.log"), encoding="utf-8", errors="replace").read()
found = re.findall(r"PROBEWIDTH (\S+) ([\d.]+)pt ([\d.]+)pt", log)
print(f"{'table':<24} {'width(pt)':>10} {'height':>8}   fits AoS(404)  fits dev(451)")
worst = []
for name, w, h in found:
    wf, hf = float(w), float(h)
    ok_aos = "yes" if wf <= LIMITS["AoS"] else f"NO (+{wf - LIMITS['AoS']:.1f})"
    ok_dev = "yes" if wf <= LIMITS["development"] else f"NO (+{wf - LIMITS['development']:.1f})"
    print(f"{name:<24} {wf:10.1f} {hf:8.1f}   {ok_aos:<14} {ok_dev}")
    if wf > LIMITS["AoS"]:
        worst.append((name, wf))
print()
print("tables too wide for the AoS text block:", worst if worst else "none")
print("(probe log:", os.path.join(OUT, "probe.log"), ")")
