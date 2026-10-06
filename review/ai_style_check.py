"""
Scan paper/main.tex (and the generated tables) for stylistic fingerprints that read as
machine-written academic prose, and report them with line numbers.

Categories
  * stock LLM vocabulary
  * meta-commentary about the writing itself ("we report this plainly", "we emphasise")
  * rhetorical antithesis ("not X but Y", "is not merely", "exactly ... not")
  * em-dash density (LaTeX "---") per 1000 words
  * sentence-opening discourse markers ("Notably,", "Crucially,", ...)
  * enumeration tics ("Three facts stand out", "Two cases are worth naming")
  * sentence-length uniformity (low variance is a machine tell)

Run:  python review/ai_style_check.py
"""

import glob
import io
import re
from collections import Counter

FILES = ["paper/main.tex"] + sorted(glob.glob("paper/tables/*.tex"))

VOCAB = [
    "delve", "leverage", "leveraging", "crucial", "crucially", "pivotal", "underscore",
    "underscores", "underscoring", "showcase", "comprehensive", "nuanced", "nuance",
    "seamless", "seamlessly", "testament", "realm", "landscape", "tapestry", "navigate",
    "it is worth noting", "it is important to note", "it should be noted", "in conclusion",
    "plays a key role", "plays a vital role", "shed light", "pave the way",
    "at the heart of", "the key insight", "rich", "vibrant", "intricate",
]
META = [
    "we emphasise", "we emphasize", "we stress", "we report this plainly", "report plainly",
    "we are explicit", "to be explicit", "we are careful", "we have been careful",
    "honest", "honestly", "transparent", "we do not hide", "we acknowledge",
    "what is and is not", "we state plainly", "this is the sharpest way",
]
ANTITHESIS = [
    "not merely", "not only", "more than just", "rather than merely",
    "is not a", "it is not a", "not a bug", "is exactly", "exactly the",
    "precisely the", "is precisely", "and not", "but not",
]
ENUM = [
    "facts stand out", "things stand out", "cases are worth naming", "worth naming",
    "worth recording", "two features", "three facts", "two things", "three reasons",
    "we make three", "we report both", "first, ", "second, ", "third, ",
]
OPENERS = [
    "Notably", "Importantly", "Crucially", "Moreover", "Furthermore", "Consequently",
    "Therefore", "Specifically", "In particular", "By contrast", "Overall", "Hence",
    "Indeed", "Finally", "First", "Second", "Third", "Thus", "So ",
]


def strip_tex(s):
    s = re.sub(r"%.*", "", s)
    s = re.sub(r"\\begin\{(table|figure|tabular|equation|align)\}.*?\\end\{\1\}", " ",
               s, flags=re.S)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", s)
    s = re.sub(r"[{}$\\&_^~]", " ", s)
    return re.sub(r"\s+", " ", s)


def main():
    total_words = 0
    text_all = []
    for path in FILES:
        raw = io.open(path, encoding="utf-8").read()
        plain = strip_tex(raw)
        total_words += len(plain.split())
        text_all.append((path, raw, plain))

    print(f"corpus: {len(FILES)} files, ~{total_words} words of prose\n")

    print("### stock LLM vocabulary")
    for pat in VOCAB:
        hits = []
        for path, raw, _ in text_all:
            for i, line in enumerate(raw.split("\n"), 1):
                if re.search(pat, line, re.I):
                    hits.append((path, i, line.strip()[:90]))
        if hits:
            print(f"  {pat!r}: {len(hits)}")
            for path, i, line in hits[:4]:
                print(f"      {path}:{i}  {line}")

    print("\n### meta-commentary about the writing")
    for pat in META:
        hits = []
        for path, raw, _ in text_all:
            for i, line in enumerate(raw.split("\n"), 1):
                if re.search(pat, line, re.I):
                    hits.append((path, i, line.strip()[:90]))
        if hits:
            print(f"  {pat!r}: {len(hits)}")
            for path, i, line in hits[:4]:
                print(f"      {path}:{i}  {line}")

    print("\n### rhetorical antithesis / 'exactly' tics")
    for pat in ANTITHESIS:
        n = 0
        for path, raw, _ in text_all:
            n += len(re.findall(pat, raw, re.I))
        if n >= 3:
            print(f"  {pat!r}: {n}")

    print("\n### enumeration tics")
    for pat in ENUM:
        n = 0
        for path, raw, _ in text_all:
            n += len(re.findall(pat, raw, re.I))
        if n:
            print(f"  {pat!r}: {n}")

    print("\n### sentence-initial discourse markers (main text only)")
    body = io.open("paper/main.tex", encoding="utf-8").read()
    body = re.sub(r"\\begin\{(table|figure|tabular)\}.*?\\end\{\1\}", " ", body, flags=re.S)
    sents = re.split(r"(?<=[.!?])\s+", strip_tex(body))
    cnt = Counter()
    for s in sents:
        for op in OPENERS:
            if s.startswith(op):
                cnt[op.strip()] += 1
    for op, n in cnt.most_common():
        print(f"  {op!r}: {n}")
    print(f"  total sentences: {len(sents)}")
    lens = [len(s.split()) for s in sents if len(s.split()) > 3]
    mean = sum(lens) / max(len(lens), 1)
    var = sum((x - mean) ** 2 for x in lens) / max(len(lens), 1)
    print(f"  sentence length: mean {mean:.1f}, sd {var ** 0.5:.1f} "
          f"(uniform-length prose is a machine tell)")

    print("\n### em-dash density (LaTeX '---')")
    for path, raw, plain in text_all:
        words = len(plain.split())
        dashes = len(re.findall(r"---", raw))
        if words:
            print(f"  {path:<28} {dashes:3d} dashes / {words:5d} words "
                  f"= {1000 * dashes / words:.1f} per 1000 words")

    print("\n### bullet / list density (main text)")
    items = len(re.findall(r"\\item", body))
    envs = len(re.findall(r"\\begin\{itemize\}|\\begin\{enumerate\}", body))
    print(f"  {items} \\item in {envs} lists; words per list item "
          f"{total_words / max(items, 1):.0f}")


if __name__ == "__main__":
    main()
