"""
Layout check: overfull boxes in the LaTeX logs, text that runs past the body-text block in
the compiled PDFs, and the length of every table/figure caption and file name.

Run:  python review/layout_check.py
"""

import io
import os
import re
from collections import Counter

import fitz

PAPERS = [("development", "paper/main.pdf", "paper/main.log", "paper/main.tex"),
          ("TEST", "submission_TEST/main_test.pdf", "review/test_build.log",
           "submission_TEST/source/main_test.tex"),
          ("TEST supplement", "submission_TEST/supplementary.pdf",
           "review/test_supplementary_build.log", "submission_TEST/source/supplementary.tex")]


# ------------------------------------------------------------------ LaTeX log
def log_report(log):
    if not os.path.exists(log):
        return None
    txt = io.open(log, encoding="utf-8", errors="replace").read()
    over_h = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)[^\n]*\n?([^\n]*)", txt)
    over_v = re.findall(r"Overfull \\vbox \(([\d.]+)pt too high\)", txt)
    under_h = re.findall(r"Underfull \\hbox", txt)
    bad = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\) detected at line (\d+)", txt)
    return dict(n_hbox=len(over_h), n_vbox=len(over_v), n_under=len(under_h),
                max_hbox=max([float(a) for a, _ in over_h], default=0.0),
                worst=sorted(over_h, key=lambda t: -float(t[0]))[:8],
                float_overfull=bad[:8])


# ------------------------------------------------------------------ PDF margins
def body_block(doc):
    """Dominant left/right text edges across the document."""
    lefts, rights = Counter(), Counter()
    for page in doc:
        for blk in page.get_text("dict")["blocks"]:
            if blk.get("type") != 0:
                continue
            for line in blk["lines"]:
                for span in line["spans"]:
                    if not span["text"].strip():
                        continue
                    lefts[round(span["bbox"][0])] += 1
                    rights[round(span["bbox"][2])] += 1
    left = min(lefts, key=lambda k: k) if lefts else 0
    right = max(rights, key=lambda k: k) if rights else 0
    return left, right, lefts, rights


def alignment_report(doc):
    """The body-text block must sit at the same place on every page.

    A binding offset in the document class moves the block sideways on facing pages,
    which is invisible in print but shows up as a horizontal jump when the PDF is read
    one page at a time. The modal left edge is computed over each parity separately
    (so lists, tables and the bibliography, which are indented or hanging, cannot mask
    a shift), and any page whose content starts left of the block is reported.
    """
    per_parity = {0: Counter(), 1: Counter()}
    mins = []
    for pno, page in enumerate(doc, start=1):
        rows = []
        for blk in page.get_text("dict")["blocks"]:
            if blk.get("type") != 0:
                continue
            for line in blk["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        rows.append((span["bbox"][0], span["bbox"][2]))
        if not rows:
            continue
        for x0, _x1 in rows:
            per_parity[pno % 2][round(x0, 1)] += 1
        mins.append((pno, round(min(r[0] for r in rows), 1), round(max(r[1] for r in rows), 1)))

    def modal(counter):
        return max(counter.items(), key=lambda kv: kv[1])[0] if counter else None

    odd, even = modal(per_parity[1]), modal(per_parity[0])
    body = odd if odd is not None else even
    out = [p for p in mins if body is not None and p[1] < body - 1.0]
    return dict(odd=odd, even=even, body_left=body,
                spread=round(abs((odd or 0) - (even or 0)), 1),
                left_of_block=out,
                right_max=max([m[2] for m in mins], default=None))


def pdf_report(pdf):
    doc = fitz.open(pdf)
    left, right, lefts, rights = body_block(doc)
    # body text edge = the most common right edge among frequently used positions
    common_right = max((r for r, c in rights.items() if c >= 5), default=right)
    offenders = []
    for pno, page in enumerate(doc, start=1):
        for blk in page.get_text("dict")["blocks"]:
            if blk.get("type") != 0:
                continue
            for line in blk["lines"]:
                for span in line["spans"]:
                    t = span["text"].strip()
                    if not t:
                        continue
                    x1 = span["bbox"][2]
                    if x1 > common_right + 2.0:
                        offenders.append((pno, round(x1 - common_right, 2), t[:70]))
        for img in page.get_images(full=True):
            pass
    # wide images
    wide = []
    for pno, page in enumerate(doc, start=1):
        for info in page.get_image_info():
            bb = info["bbox"]
            if bb[2] > common_right + 2.0 or bb[0] < left - 2.0:
                wide.append((pno, round(bb[0], 1), round(bb[2], 1),
                             round(bb[2] - bb[0], 1)))
    return dict(pages=doc.page_count, body_left=left, body_right=common_right,
                page_width=round(doc[0].rect.width, 1),
                overflow=sorted(offenders, key=lambda t: -t[1])[:15],
                n_overflow=len(offenders), wide_images=wide,
                align=alignment_report(doc))


# ------------------------------------------------------------------ captions
def captions(tex):
    txt = io.open(tex, encoding="utf-8").read()
    out = []
    for m in re.finditer(r"\\caption\{", txt):
        i = m.end()
        depth, start = 1, i
        while i < len(txt) and depth:
            if txt[i] == "{":
                depth += 1
            elif txt[i] == "}":
                depth -= 1
            i += 1
        body = txt[start:i - 1]
        plain = re.sub(r"\\[a-zA-Z]+\*?", " ", body)
        plain = re.sub(r"[{}$\\]", " ", plain)
        words = len([w for w in re.split(r"\s+", plain) if w.strip()])
        # which float does it belong to?
        before = txt[max(0, m.start() - 4000):m.start()]
        kind = "table" if before.rfind("\\begin{table") > before.rfind("\\begin{figure") \
            else "figure"
        lab = re.search(r"\\label\{([^}]+)\}", txt[i:i + 400])
        figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
                          before[before.rfind("\\begin{" + kind):] if kind else before)
        out.append(dict(kind=kind, words=words, chars=len(plain.strip()),
                        label=lab.group(1) if lab else "?",
                        file=figs[0] if figs else None,
                        head=re.sub(r"\s+", " ", plain.strip())[:80]))
    return out


def main():
    for name, pdf, log, tex in PAPERS:
        print("=" * 100)
        print(f"### {name}: {pdf}")
        lr = log_report(log)
        if lr:
            print(f"  log: overfull hbox {lr['n_hbox']} (max {lr['max_hbox']:.1f}pt), "
                  f"overfull vbox {lr['n_vbox']}, underfull hbox {lr['n_under']}")
            for size, snippet in lr["worst"]:
                print(f"     {float(size):6.2f}pt  {snippet.strip()[:90]}")
        pr = pdf_report(pdf)
        print(f"  pdf: {pr['pages']} pages, page width {pr['page_width']}pt, "
              f"body text block x = [{pr['body_left']}, {pr['body_right']}]")
        al = pr["align"]
        print(f"  alignment: modal left edge odd pages {al['odd']}, even pages {al['even']} "
              f"(spread {al['spread']}pt); body block {al['body_left']}")
        if al["left_of_block"]:
            print(f"     content left of the block on pages {al['left_of_block']}")
        else:
            print("     no page starts left of the block; facing pages line up")
        print(f"  spans past the body block: {pr['n_overflow']}")
        for pno, over, t in pr["overflow"]:
            print(f"     p{pno:>3}  +{over:5.2f}pt  {t}")
        if pr["wide_images"]:
            for pno, x0, x1, w in pr["wide_images"]:
                print(f"     image p{pno}: x=[{x0}, {x1}] width {w}pt")
    print("=" * 100)
    caps = captions("paper/main.tex")
    import glob
    for tf in sorted(glob.glob("paper/tables/*.tex")):
        caps += captions(tf)
    print(f"### captions (paper/main.tex + paper/tables/*.tex): {len(caps)}")
    for c in sorted(caps, key=lambda c: -c["words"]):
        print(f"  {c['kind']:6s} {c['words']:4d} words  label={c['label']:<28} "
              f"file={str(c['file']):<32} {c['head'][:52]}")
    print("=" * 100)
    files = []
    for root in ("submission_TEST/source/figures",):
        if not os.path.isdir(root):
            continue
        for f in sorted(os.listdir(root)):
            files.append((os.path.join(root, f), len(f)))
    print("### figure file names")
    for path, n in files:
        print(f"  {n:3d} chars  {path}")


if __name__ == "__main__":
    main()
