"""
Direct measurement: take the first sentence of every paragraph in the source, locate it in
the compiled PDF, and report whether the paragraph starts indented or flush.

Run:  python review/indent_paragraph_audit.py
"""

import io
import re

import fitz

DOCS = [("development", "paper/main.tex", "paper/main.pdf"),
        ("TEST", "submission_TEST/source/main_test.tex", "submission_TEST/main_test.pdf")]


def norm(s):
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = re.sub(r"[{}$\\~^_]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower().strip()


def source_paragraphs(path):
    txt = io.open(path, encoding="utf-8").read()
    # drop the preamble and the float bodies, keep the running text
    txt = txt[txt.index("\\begin{document}"):]
    txt = re.sub(r"\\begin\{(table|figure|tabular)\}.*?\\end\{\1\}", " ", txt, flags=re.S)
    out = []
    blocks = re.split(r"\n\s*\n", txt)
    for b in blocks:
        b = b.strip()
        if not b or b.startswith(("%", "\\begin{", "\\end{", "\\input", "\\section",
                                  "\\subsection", "\\paragraph", "\\item", "\\[")):
            continue
        if len(norm(b)) < 60:
            continue
        out.append(norm(b)[:60])
    return out


def pdf_lines(pdf):
    doc = fitz.open(pdf)
    lines = []
    for pno, page in enumerate(doc, start=1):
        for blk in page.get_text("dict")["blocks"]:
            if blk.get("type") != 0:
                continue
            for ln in blk["lines"]:
                t = norm("".join(s["text"] for s in ln["spans"]))
                if t:
                    lines.append((pno, ln["bbox"][0], t))
    return lines


def main():
    for name, tex, pdf in DOCS:
        starts = source_paragraphs(tex)
        lines = pdf_lines(pdf)
        margins = [x for _, x, _ in lines]
        margin = sorted(margins)[len(margins) // 20]        # 5th percentile as body margin
        found = 0
        indented = 0
        rows = []
        for s in starts:
            hit = None
            for pno, x0, t in lines:
                if t.startswith(s[:40]):
                    hit = (pno, x0, t)
                    break
            if not hit:
                continue
            found += 1
            kind = "indented" if hit[1] > margin + 4 else "FLUSH   "
            if kind == "indented":
                indented += 1
            rows.append((hit[0], hit[1], kind, hit[2][:66]))
        print("=" * 100)
        print(f"### {name}: located {found}/{len(starts)} paragraph starts in the PDF; "
              f"{indented} indented, {found - indented} flush (body margin {margin:.1f}pt)")
        for pno, x0, kind, t in rows:
            if kind == "FLUSH   ":
                print(f"   p{pno:>3} x0={x0:6.1f} {kind} {t}")
        print("   (flush ones are expected only after headings, lists and in floats)")


if __name__ == "__main__":
    main()

