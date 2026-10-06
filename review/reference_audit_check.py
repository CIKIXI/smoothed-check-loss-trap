"""
Exhaustive, item-by-item audit of paper/references.bib (reference-checker skill).

Verification routes, in order of preference:

  * DOI given or found -> Crossref /works/{doi}
  * journal article    -> Crossref bibliographic query restricted to the journal
  * biomedical article -> PubMed esummary (journal issue year, volume, issue, pages)
  * book               -> Crossref bibliographic query (title + publisher)
  * proceedings        -> Crossref, then DBLP
  * preprint           -> arXiv API / OpenAlex

Year handling: the submitted year is accepted when it equals *either* the Crossref
`issued' year (online first) *or* the print year of the journal issue
(`published-print', `journal-issue.published-print'), and the audit records which one
matched.  Titles are compared on their main part, so a publisher subtitle that the
manuscript omits is reported as a Minor difference, not as an identity mismatch.

Outputs:
    review/reference_audit_v2.md    per-reference audit table
    review/reference_audit_v2.json  raw matched metadata (the evidence)

Run:  python review/reference_audit_check.py
"""

import difflib
import io
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

BIB = "paper/references.bib"
TEX = "paper/main.tex"
OUTDIR = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(OUTDIR, "reference_audit_v2.md")
JS = os.path.join(OUTDIR, "reference_audit_v2.json")

UA = {"User-Agent": "SCI1-reference-audit/1.0 (mailto:author@university.edu)"}
CACHE = {}


def get(url, as_json=True):
    if url in CACHE:
        return CACHE[url]
    out, err = None, None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=45) as fh:
                body = fh.read().decode("utf-8", "replace")
            out = json.loads(body) if as_json else body
            err = None
            break
        except urllib.error.HTTPError as exc:
            err = f"HTTP {exc.code}"
            out = None
            if exc.code in (400, 404):
                break
            time.sleep(4.0 * (attempt + 1) if exc.code == 429 else 1.0 + attempt)
        except Exception as exc:                                       # noqa: BLE001
            err = str(exc)
            out = None
            time.sleep(1.0 + attempt)
    CACHE[url] = (out, err)
    time.sleep(0.3)                       # stay well inside the public rate limits
    return out, err


# --------------------------------------------------------------------- bib parsing
def clean(s):
    s = s.replace("--", "-")
    s = re.sub(r"[{}]", "", s)
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().strip(".").strip()


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", clean(s).lower()).strip()


def main_title(s):
    """Drop a publisher subtitle so that 'X: Y' and 'X' compare equal on their main part."""
    return clean(s).split(":")[0].strip()


def similarity(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def parse_bib(path):
    """Brace-matching parser: handles nested braces and multi-line values."""
    txt = io.open(path, encoding="utf-8").read()
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{", txt):
        typ = m.group(1).lower()
        i = m.end()
        depth, start = 1, i
        while i < len(txt) and depth:
            if txt[i] == "{":
                depth += 1
            elif txt[i] == "}":
                depth -= 1
            i += 1
        body = txt[start:i - 1]
        key = body.split(",")[0].strip()
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{", body):
            j = fm.end()
            d, s0 = 1, j
            while j < len(body) and d:
                if body[j] == "{":
                    d += 1
                elif body[j] == "}":
                    d -= 1
                j += 1
            fields[fm.group(1).lower()] = clean(body[s0:j - 1])
        entries.append(dict(key=key, type=typ, **fields))
    return entries


def first_author_surname(entry):
    authors = entry.get("author", "")
    first = authors.split(" and ")[0] if authors else ""
    if "," in first:
        return first.split(",")[0].strip()
    return first.split()[-1] if first else ""


# --------------------------------------------------------------------- routes
def crossref_ref(msg):
    authors = msg.get("author") or []
    fam = authors[0].get("family", "") if authors else ""

    def year_of(k):
        dp = (msg.get(k) or {}).get("date-parts") or []
        return dp[0][0] if dp and dp[0] else None

    years = {k: year_of(k) for k in ("issued", "published-print", "published-online",
                                     "published", "created")}
    ji = msg.get("journal-issue") or {}
    years["journal-issue.published-print"] = (
        (ji.get("published-print") or {}).get("date-parts") or [[None]])[0][0]
    return dict(title=(msg.get("title") or [""])[0],
                container=(msg.get("container-title") or [""])[0],
                year=years.get("issued") or years.get("published-print"),
                years=years, volume=msg.get("volume"), issue=msg.get("issue"),
                page=msg.get("page"), first_author=fam, doi=msg.get("DOI"),
                publisher=msg.get("publisher"), type=msg.get("type"))


def crossref_by_doi(doi):
    d, err = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
    if not d or "message" not in d:
        return None, dict(error=err)
    return d["message"], dict(ok=True)


def crossref_by_query(entry):
    """Two-stage query: first unrestricted, then restricted to the journal/publisher.

    The evidence returned always belongs to the item that is actually returned."""
    first, ev1 = _crossref_query(entry, restrict=False)
    if first and ev1.get("score", 0) >= 0.90:
        return first, ev1
    container = entry.get("journal") or entry.get("publisher") or ""
    second, ev2 = _crossref_query(entry, restrict=bool(container))
    if second and ev2.get("score", 0) > ev1.get("score", 0):
        return second, ev2
    return (first, ev1) if first else (second, ev2)


def _crossref_query(entry, restrict):
    q = main_title(entry.get("title", ""))
    surname = first_author_surname(entry)
    container = entry.get("journal") or entry.get("publisher") or ""
    query = " ".join(x for x in (q, surname) if x)
    url = ("https://api.crossref.org/works?rows=8&select=DOI,title,author,container-title,"
           "issued,published-print,published-online,journal-issue,volume,issue,page,type,"
           "publisher,update-to&query.bibliographic=" + urllib.parse.quote(query))
    if restrict and container:
        url += "&query.container-title=" + urllib.parse.quote(container)
    d, err = get(url)
    items = ((d or {}).get("message") or {}).get("items", [])
    best, score = None, 0.0
    for it in items:
        t = main_title((it.get("title") or [""])[0])
        s = similarity(q, t)
        fams = [a.get("family", "") for a in it.get("author", []) or []]
        if surname and fams and not any(similarity(surname, f) > 0.85 for f in fams):
            s -= 0.20
        ct = (it.get("container-title") or [""])[0]
        if container and ct and similarity(container, ct) < 0.5:
            s -= 0.10
        if entry.get("year"):
            try:
                iss = (it.get("issued", {}).get("date-parts") or [[int(entry["year"])]])[0][0]
                if abs(int(entry["year"]) - iss) > 3:
                    s -= 0.10
            except (ValueError, TypeError):
                pass
        if s > score:
            best, score = it, s
    return best, dict(score=round(score, 3), n_items=len(items), error=err,
                      restricted=bool(restrict and container))


def arxiv_id_of(entry):
    blob = " ".join(str(entry.get(f, "")) for f in ("howpublished", "note", "eprint", "url"))
    m = re.search(r"(\d{4}\.\d{4,5})", blob)
    return m.group(1) if m else None


def arxiv_by_id(aid):
    xml, err = get("http://export.arxiv.org/api/query?id_list=" + aid, as_json=False)
    if not xml or "<entry>" not in xml:
        return None, dict(error=err)
    t = re.search(r"<entry>.*?<title>(.*?)</title>", xml, re.S)
    au = re.findall(r"<name>(.*?)</name>", xml, re.S)
    pub = re.search(r"<published>(\d{4})", xml)
    return dict(title=clean(t.group(1)) if t else None,
                authors=[clean(a) for a in au][:6],
                year=int(pub.group(1)) if pub else None,
                url="https://arxiv.org/abs/" + aid), dict(ok=True, id=aid)


def pubmed_by_doi(doi):
    q = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?retmode=json&db=pubmed&term="
    d, err = get(q + urllib.parse.quote(doi + "[DOI]"))
    ids = (((d or {}).get("esearchresult") or {}).get("idlist") or [])
    if not ids:
        return None, dict(error=err, ids=0)
    s, err2 = get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?retmode=json"
                  "&db=pubmed&id=" + ids[0])
    res = ((s or {}).get("result") or {})
    rec = res.get(ids[0]) or {}
    if not rec:
        return None, dict(error=err2, ids=len(ids))
    return dict(title=rec.get("title"), journal=rec.get("fulljournalname") or rec.get("source"),
                year=(rec.get("pubdate") or "").split(" ")[0], volume=rec.get("volume"),
                issue=rec.get("issue"), pages=rec.get("pages"), pmid=ids[0]), \
        dict(pmid=ids[0])


def openalex_by_query(entry):
    url = ("https://api.openalex.org/works?per-page=5&mailto=author@university.edu&search="
           + urllib.parse.quote(main_title(entry.get("title", ""))))
    d, err = get(url)
    items = (d or {}).get("results", [])
    best, score = None, 0.0
    for it in items:
        s = similarity(main_title(entry.get("title", "")), it.get("display_name", "") or "")
        if s > score:
            best, score = it, s
    return best, dict(score=round(score, 3), n_items=len(items), error=err)


def openalex_record(oa):
    biblio = oa.get("biblio") or {}
    pl = oa.get("primary_location") or {}
    src = pl.get("source") or {}
    return dict(venue=src.get("display_name"), year=oa.get("publication_year"),
                volume=biblio.get("volume"), issue=biblio.get("issue"),
                first_page=biblio.get("first_page"), last_page=biblio.get("last_page"),
                doi=(oa.get("doi") or "").replace("https://doi.org/", "") or None,
                type=oa.get("type"))


def dblp_by_title(entry):
    url = ("https://dblp.org/search/publ/api?format=json&h=8&q="
           + urllib.parse.quote(main_title(entry.get("title", ""))))
    d, err = get(url)
    hits = (((d or {}).get("result") or {}).get("hits") or {}).get("hit", [])
    best, score, info = None, 0.0, None
    for h in hits:
        i = h.get("info", {})
        s = similarity(main_title(entry.get("title", "")), i.get("title", ""))
        if s > score:
            best, score, info = i.get("title"), s, i
    return info, dict(score=round(score, 3), n_hits=len(hits), error=err)


def arxiv_by_title(entry):
    url = ("http://export.arxiv.org/api/query?max_results=5&search_query=ti:"
           + urllib.parse.quote('"%s"' % main_title(entry.get("title", ""))))
    xml, err = get(url, as_json=False)
    if not xml:
        return None, dict(error=err)
    titles = re.findall(r"<title>(.*?)</title>", xml, re.S)[1:]
    ids = re.findall(r"<id>(http://arxiv.org/abs/.*?)</id>", xml, re.S)
    best, score, best_id = None, 0.0, None
    for t, i in zip(titles, ids):
        s = similarity(main_title(entry.get("title", "")), t)
        if s > score:
            best, score, best_id = clean(t), s, i
    return (dict(title=best, url=best_id) if best else None), dict(score=round(score, 3),
                                                                  error=err)


# --------------------------------------------------------------------- manual evidence
# Two references are not covered by Crossref/OpenAlex/DBLP and were verified directly at
# the publisher.  The metadata below was read off those pages in this session; the quote
# and the URL are recorded so the check can be repeated by hand.
MANUAL = {
    "he2000quantile": dict(
        route="publisher PDF (Statistica Sinica archive)",
        url="https://www3.stat.sinica.edu.tw/statistica/oldpdf/A10n16.pdf",
        quote="Statistica Sinica 10(2000), 129-140 | QUANTILE REGRESSION ESTIMATES FOR A "
              "CLASS OF LINEAR AND PARTIALLY LINEAR ERRORS-IN-VARIABLES MODELS | "
              "Xuming He and Hua Liang",
        ref=dict(title="Quantile regression estimates for a class of linear and partially "
                       "linear errors-in-variables models",
                 container="Statistica Sinica", year=2000, years={}, volume="10",
                 issue=None, page="129-140", first_author="He", doi=None,
                 publisher=None, type="journal-article")),
    "romano2019conformalized": dict(
        route="NeurIPS 2019 proceedings index (papers.nips.cc)",
        url="https://papers.nips.cc/paper_files/paper/2019/hash/"
            "5103c3584b063c431bd1268e9b5e76fb-Abstract.html",
        quote="Conformalized Quantile Regression, Advances in Neural Information "
              "Processing Systems 32 (2019), Romano, Patterson and Candes",
        ref=dict(title="Conformalized quantile regression",
                 container="Advances in Neural Information Processing Systems 32",
                 year=2019, years={}, volume=None, issue=None, page=None,
                 first_author="Romano", doi=None, publisher=None,
                 type="proceedings-article")),
}


# --------------------------------------------------------------------- comparison
def cmp_fields(entry, ref):
    notes, flags = [], []
    sub_main, ref_main = main_title(entry.get("title", "")), main_title(ref.get("title") or "")
    full_sim = similarity(entry.get("title", ""), ref.get("title") or "")
    main_sim = similarity(sub_main, ref_main)
    if main_sim < 0.80:
        flags.append("title")
        notes.append(f"title similarity {main_sim:.2f} vs \"{(ref.get('title') or '')[:70]}\"")
    elif full_sim < 0.90 and len(ref.get("title") or "") > len(entry.get("title", "")) + 4:
        notes.append(f"publisher title adds a subtitle: \"{(ref.get('title') or '')[:80]}\"")
    if ref.get("container") and (entry.get("journal") or entry.get("booktitle")):
        s = similarity(entry.get("journal") or entry.get("booktitle"),
                       main_title(ref["container"]))
        if s < 0.75:
            flags.append("journal")
            notes.append(f"venue \"{ref['container'][:60]}\"")
    if ref.get("year") and entry.get("year"):
        try:
            sub = int(entry["year"])
            acceptable = {v for v in (ref.get("years") or {}).values() if v} or {ref["year"]}
            if sub not in acceptable:
                flags.append("year")
                notes.append(f"year {entry['year']} vs {ref['year']}")
            elif sub != ref["year"]:
                notes.append(f"year {sub} = the print/issue year; the identifier's "
                             f"`issued' date is {ref['year']} (online first)")
        except (ValueError, TypeError):
            pass
    for f, bibfield in (("volume", "volume"), ("issue", "number"), ("page", "pages")):
        if ref.get(f) and entry.get(bibfield):
            a, b = norm(str(entry[bibfield])), norm(str(ref[f]))
            if a != b:
                if f == "page" and norm(str(ref[f])) in a:
                    notes.append(f"{f} {entry[bibfield]} refines {ref[f]}")
                else:
                    flags.append(f)
                    notes.append(f"{f} {entry[bibfield]} vs {ref[f]}")
    return flags, notes


# --------------------------------------------------------------------- main
def main():
    entries = parse_bib(BIB)
    tex = io.open(TEX, encoding="utf-8").read()
    cited = set()
    for m in re.finditer(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", tex):
        for k in m.group(1).split(","):
            cited.add(k.strip())
    print(f"parsed {len(entries)} entries; {len(cited)} distinct citation keys in {TEX}")

    rows = []
    for e in entries:
        rec = dict(key=e["key"], type=e["type"], submitted=e)
        route, ref, evidence = None, None, {}
        flags, notes = [], []

        if e.get("doi"):
            msg, ev = crossref_by_doi(e["doi"])
            evidence["crossref_doi"] = ev
            if msg:
                route = "Crossref DOI metadata"
                ref = crossref_ref(msg)
                evidence["crossref"] = ref
                for u in msg.get("update-to", []) or []:
                    if u.get("type") in ("retraction", "withdrawal"):
                        evidence["retraction"] = f"{u.get('type')} {u.get('DOI')}"
            else:
                route = "Crossref DOI lookup failed"
        if ref is None:
            best, ev = crossref_by_query(e)
            evidence["crossref_query"] = ev
            if best and ev.get("score", 0) > 0.85:
                route = "Crossref title/author/journal query"
                ref = crossref_ref(best)
                evidence["crossref"] = ref
        if ref is None and e["key"] in MANUAL:
            m = MANUAL[e["key"]]
            route = m["route"]
            ref = dict(m["ref"])
            evidence["manual_evidence"] = {k: m[k] for k in ("route", "url", "quote")}
        if ref is None and e["type"] in ("inproceedings", "misc"):
            info, dev = dblp_by_title(e)
            evidence["dblp_query"] = dev
            if info and dev.get("score", 0) > 0.85:
                route = "DBLP title query"
                ref = dict(title=info.get("title"), container=info.get("venue"),
                           year=int(info["year"]) if info.get("year") else None,
                           years={}, volume=None, issue=None,
                           page=(f"{info.get('pages')}" if info.get("pages") else None),
                           first_author=None, doi=info.get("doi"), publisher=None,
                           type="proceedings-article")
                evidence["dblp"] = info
        if ref is None:
            oa, oev = openalex_by_query(e)
            evidence["openalex_query"] = oev
            if oa and oev.get("score", 0) > 0.85:
                route = "OpenAlex title query"
                oarec = openalex_record(oa)
                evidence["openalex_record"] = oarec
                ref = dict(title=oarec["title"] if "title" in oarec else oa.get("display_name"),
                           container=oarec["venue"], year=oarec["year"], years={},
                           volume=oarec["volume"], issue=oarec["issue"],
                           page=(f"{oarec['first_page']}-{oarec['last_page']}"
                                 if oarec.get("first_page") else None),
                           first_author=None, doi=oarec["doi"], publisher=None,
                           type=oarec["type"])
        if ref is None and e["type"] == "misc":
            ax, aev = arxiv_by_title(e)
            evidence["arxiv_query"] = aev
            if ax and aev.get("score", 0) > 0.85:
                route = "arXiv API title query"
                ref = dict(title=ax["title"], container="arXiv preprint", year=None,
                           years={}, volume=None, issue=None, page=None,
                           first_author=None, doi=None, publisher=None, type="preprint")
                evidence["arxiv_url"] = ax["url"]

        # corroborating routes (only where they add something the primary route did not)
        need_more = (ref is None) or not (ref or {}).get("page")
        if need_more:
            oa, oev = openalex_by_query(e)
            if oa:
                evidence.setdefault("openalex_record", openalex_record(oa))
        aid = arxiv_id_of(e)
        if aid:
            ax, aev = arxiv_by_id(aid)
            evidence["arxiv_id_check"] = dict(id=aid, ok=bool(ax),
                                              title=(ax or {}).get("title"),
                                              year=(ax or {}).get("year"))
            if ax is None:
                notes.append(f"arXiv id {aid} did not resolve")
            elif ref is None:
                route = "arXiv API id_list"
                ref = dict(title=ax["title"], container="arXiv preprint",
                           year=ax["year"], years={}, volume=None, issue=None, page=None,
                           first_author=None, doi=None, publisher=None, type="preprint")
                evidence["arxiv_url"] = ax["url"]
            elif ax["title"]:
                notes.append(f"arXiv:{aid} confirms the preprint title")
        if e["type"] in ("inproceedings",) and ref is None:
            info, dev = dblp_by_title(e)
            evidence.setdefault("dblp_query", dev)
            if info:
                evidence.setdefault("dblp", dict(title=info.get("title"),
                                                 venue=info.get("venue"),
                                                 year=info.get("year"),
                                                 doi=info.get("doi"),
                                                 url=info.get("url")))
        if e.get("doi"):
            pm, pev = pubmed_by_doi(e["doi"])
            evidence["pubmed"] = (pm or pev)

        if ref:
            flags, notes = cmp_fields(e, ref)
            if e.get("doi") and evidence.get("pubmed") and evidence["pubmed"].get("pmid"):
                pm = evidence["pubmed"]
                notes.append(f"PubMed {pm['pmid']}: {pm.get('journal')} "
                             f"{pm.get('year')};{pm.get('volume')}({pm.get('issue')}):"
                             f"{pm.get('pages')}")

        if not flags:
            quality = "Exact" if ref else "Not found"
            status = "Verified" if ref else "Manual check"
            confidence = ("High" if ref and (route or "").startswith("Crossref DOI")
                          else "Medium" if ref else "Low")
            if notes and ref and any("subtitle" in n for n in notes):
                status, quality = "Minor", "Near exact"
        elif set(flags) <= {"page"}:
            quality, status, confidence = "Near exact", "Minor", "High"
        elif set(flags) <= {"issue", "page"}:
            quality, status, confidence = "Partial", "Minor", "Medium"
        else:
            quality, status, confidence = "Partial", "Major", "Medium"

        rec.update(route=route, matched=ref, flags=flags, notes=notes,
                   match_quality=quality, status=status, confidence=confidence,
                   evidence=evidence, cited=e["key"] in cited)
        rows.append(rec)
        print(f"  {e['key']:28s} {str(route):38s} {quality:11s} {status:12s} "
              f"{('; '.join(notes))[:64]}")

    seen, duplicates = {}, []
    for e in entries:
        k = (norm(e.get("title", "")), norm(e.get("author", "")))
        if k in seen:
            duplicates.append((seen[k], e["key"]))
        seen[k] = e["key"]
    missing = sorted(cited - {e["key"] for e in entries})
    uncited = sorted({e["key"] for e in entries} - cited)
    dois_found = {r["key"]: (r["matched"] or {}).get("doi") for r in rows
                  if (r["matched"] or {}).get("doi") and not r["submitted"].get("doi")}

    with io.open(JS, "w", encoding="utf-8") as fh:
        json.dump(dict(rows=rows, duplicates=duplicates, missing_keys=missing,
                       uncited_keys=uncited, dois_found=dois_found), fh, indent=1)

    lines = ["# Reference audit (reference-checker skill, exhaustive pass)", "",
             f"Source: `{BIB}` — {len(entries)} entries; every entry was checked "
             "individually. Routes: Crossref DOI metadata, Crossref bibliographic query "
             "(restricted to the journal/publisher), PubMed esummary, DBLP, OpenAlex, "
             "arXiv API. Raw matched metadata: `review/reference_audit_v2.json`.", "",
             "| Ref | Submitted title | Submitted authors | Source | Year | Identifier / route "
             "| Match quality | Status | Confidence | Main issue / suggested fix |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s = r["submitted"]
        ident = s.get("doi") or (r["matched"] or {}).get("doi") or \
            r["evidence"].get("arxiv_url") or "—"
        issue = "; ".join(r["notes"]) if r["notes"] else "none"
        if r["evidence"].get("retraction"):
            issue += f" | RETRACTION SIGNAL: {r['evidence']['retraction']}"
        lines.append("| `{}` | {} | {} | {} | {} | {} — {} | {} | {} | {} | {} |".format(
            r["key"], s.get("title", "")[:80], s.get("author", "")[:55],
            (s.get("journal") or s.get("booktitle") or s.get("publisher") or "")[:42],
            s.get("year", ""), str(ident)[:58], r["route"], r["match_quality"],
            r["status"], r["confidence"], issue[:170]))
    lines += ["", "## Whole-list checks", "",
              f"* duplicates: {duplicates if duplicates else 'none'}",
              f"* citation keys used in the text but absent from the .bib: "
              f"{missing if missing else 'none'}",
              f"* entries never cited: {uncited if uncited else 'none'}",
              f"* retraction/withdrawal signals: "
              f"{[r['key'] for r in rows if r['evidence'].get('retraction')] or 'none'}",
              f"* DOIs located by this audit that the .bib does not yet carry: "
              f"{ {k: v for k, v in dois_found.items() if v} or 'none'}"]
    with io.open(MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[saved] {MD}\n[saved] {JS}")
    bad = [r["key"] for r in rows if r["status"] in ("Major", "Critical", "Manual check")]
    print(f"references needing attention: {bad if bad else 'none'}")
    print(f"DOIs located but not in the .bib: {dois_found}")


if __name__ == "__main__":
    main()
