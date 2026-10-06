"""
Measure the natural width of every generated table in the two layouts that matter.

Each paper/tables/tab_*.tex is a float whose body is a tabular.  This script extracts the
tabular part, boxes it in probe documents with the real class and font size of each layout,
and prints \\wd of the box, so the widths can be compared with the real text widths:

    development (article, 12pt, geometry as in paper/main.tex)   451pt
    TEST        (Springer sn-jnl, the submitted layout)          372pt

Run:  python review/table_width_probe.py
"""

import glob
import io
import os
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = "review/_table_probe"
os.makedirs(OUT, exist_ok=True)

# text block widths, measured with \the\textwidth in each layout
LIMITS = {"TEST": 372.0, "development": 451.0}
TEST_PREAMBLE = [r"\documentclass[pdflatex,sn-basic]{sn-jnl}",
                 r"\usepackage{graphicx,amsmath,amssymb,booktabs,multirow}",
                 r"\usepackage[T1]{fontenc}",
                 r"\newsavebox{\probeBox}"]
DEV_PREAMBLE = [r"\documentclass[12pt]{article}",
                r"\usepackage[margin=1in,paperwidth=9in,paperheight=11in]{geometry}",
                r"\usepackage{amsmath,amssymb,booktabs,multirow}",
                r"\usepackage[T1]{fontenc}",
                r"\newsavebox{\probeBox}"]


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


def probe(preamble, tag):
    body = list(preamble) + [r"\begin{document}", r"probe", r"\newpage"]
    names = []
    for path in sorted(glob.glob("paper/tables/*.tex")):
        got = extract_tabular(path)
        name = os.path.basename(path)[:-4]
        if not got:
            continue
        setup, tabular = got
        names.append(name)
        body.append(r"\sbox{\probeBox}{%" + "\n" + setup + tabular + "\n}")
        body.append(r"\typeout{PROBEWIDTH " + name + r" \the\wd\probeBox\space"
                    r"\the\ht\probeBox}")
        body.append(r"\newpage")
    body.append(r"\end{document}")
    tex = os.path.join(OUT, f"probe_{tag}.tex")
    io.open(tex, "w", encoding="utf-8").write("\n".join(body))
    # the sn-jnl class is kept in test_template/ and is needed for the TEST probe
    candidates = [os.path.join("test_template", "sn-jnl.cls")]
    for c in candidates:
        if os.path.exists(c):
            import shutil
            shutil.copyfile(c, os.path.join(OUT, os.path.basename(c)))
    subprocess.run(["pdflatex", "-interaction=nonstopmode", f"probe_{tag}.tex"], cwd=OUT,
                   capture_output=True, text=True, errors="replace")
    log = io.open(os.path.join(OUT, f"probe_{tag}.log"), encoding="utf-8",
                  errors="replace").read()
    return re.findall(r"PROBEWIDTH (\S+) ([\d.]+)pt ([\d.]+)pt", log)


test = {n: (float(w), float(h)) for n, w, h in probe(TEST_PREAMBLE, "test")}
dev = {n: (float(w), float(h)) for n, w, h in probe(DEV_PREAMBLE, "dev")}

print(f"{'table':<24} {'width(pt)':>10} {'height':>8}   fits TEST(372)  fits dev(451)")
too_wide = []
for name in sorted(set(test) | set(dev)):
    w, h = test.get(name, dev.get(name, (0, 0)))
    ok_test = "yes" if w <= LIMITS["TEST"] else f"NO (+{w - LIMITS['TEST']:.1f})"
    ok_dev = "yes" if w <= LIMITS["development"] else f"NO (+{w - LIMITS['development']:.1f})"
    print(f"{name:<24} {w:10.1f} {h:8.1f}   {ok_test:<15} {ok_dev}")
    if w > LIMITS["TEST"]:
        too_wide.append((name, w))
print()
print("tables too wide for the TEST text block:", too_wide if too_wide else "none")
print("(probe logs:", os.path.join(OUT, "probe_test.log"),
      os.path.join(OUT, "probe_dev.log"), ")")
