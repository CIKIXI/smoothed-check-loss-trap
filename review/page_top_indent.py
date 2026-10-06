"""Whole-paragraph indentation check.

A paragraph whose every line is indented with respect to the body margin reads as a
layout slip when it starts a page: the reader sees a paragraph that is shifted right for
no visible reason. This happens when a list item (or a proof step, or an itemised block)
continues over a page break. Report every page whose first paragraph behaves that way.

Run:  python review/page_top_indent.py
"""

import fitz

DOCS = [("TEST paper", "submission_TEST/main_test.pdf"),
        ("TEST supplement", "submission_TEST/ESM_1.pdf"),
        ("development", "paper/main.pdf")]

BODY_LEFT = {"TEST paper": 112.3, "TEST supplement": 112.3, "development": 72.0}


def paragraphs(page):
    out = []
    for blk in page.get_text("dict")["blocks"]:
        if blk.get("type") != 0:
            continue
        lines = [l for l in blk["lines"] if any(s["text"].strip() for s in l["spans"])]
        if lines:
            out.append(lines)
    return out


def main():
    for name, path in DOCS:
        doc = fitz.open(path)
        margin = BODY_LEFT[name]
        flagged = []
        for pno, page in enumerate(doc, start=1):
            blocks = paragraphs(page)
            if not blocks:
                continue
            first = min(blocks, key=lambda ls: min(l["bbox"][1] for l in ls))
            xs = [round(l["bbox"][0], 1) for l in first]
            # A normal paragraph that begins a page carries the usual first-line
            # indent, so its later lines return to the margin. Only a paragraph whose
            # every line is shifted right looks like a layout slip.
            if len(first) > 1 and min(xs) > margin + 3.0:
                text = "".join(s["text"] for s in first[0]["spans"])[:60]
                flagged.append((pno, min(xs), round(min(xs) - margin, 1), len(first), text))
        print("=" * 96)
        print(f"### {name} ({path}): body margin {margin}pt, {doc.page_count} pages")
        if flagged:
            for pno, top, off, n, text in flagged:
                print(f"   p{pno:>3} first paragraph starts at {top} (+{off}pt), "
                      f"{n} lines: {text}")
        else:
            print("   every page starts at the body margin")
    print("=" * 96)


if __name__ == "__main__":
    main()
