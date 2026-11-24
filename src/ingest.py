"""Step 1 - Parse RBI/SEBI PDFs and split them into clause-level chunks.

Usage:  python -m src.ingest
Reads   data/raw/rbi/*.pdf and data/raw/sebi/*.pdf
Writes  data/processed/chunks.jsonl  (one JSON object per chunk)
"""
import json
import re
from collections import Counter

import pymupdf

from src import config

# A new clause starts at a line that begins with a paragraph number such as
# "5.", "5.2", "5.2.1", "(5)", "Chapter III" or "Annex I".
CLAUSE_START = re.compile(
    r"^\s*(?:"
    r"(?P<num>\d{1,3}(?:\.\d{1,3}){0,4})[.)]\s+(?=[A-Z(\"'‘“])"      # "5. Text", "5.2.1. Text", "5)"
    r"|(?P<dnum>\d{1,3}(?:\.\d{1,3}){1,4})\s+(?=[A-Z(\"'‘“])"        # "5.2 Text" (dotted, no trailing dot)
    r"|(?P<chap>(?:CHAPTER|Chapter)\s+[IVXLC\d]+)\b"                 # Chapter III
    r"|(?P<annex>(?:ANNEX|Annex|Annexure|ANNEXURE)[\s-]+[IVXLC\dA-Z]+)\b"
    r")"
)
# Sub-points such as "(2)" or "(a)" stay inside their parent clause, so a chunk always
# carries its paragraph number. A bare "169 Inserted ..." (footnote) is not a clause start.
TOC_LINE = re.compile(r"(\.{3,}|…)\s*\d{1,4}\s*$|\s\d{1,4}\s*$")


def clean_lines(pages):
    """Drop lines that repeat on most pages (headers, footers, page numbers)."""
    counts = Counter()
    for lines in pages:
        counts.update(set(l.strip() for l in lines if l.strip()))
    n = len(pages)
    repeated = {l for l, c in counts.items() if n >= 4 and c >= 0.6 * n and len(l) < 120}
    out, in_toc = [], False
    for page_no, lines in enumerate(pages, start=1):
        text_lines = [l.strip() for l in lines if l.strip()]
        # table-of-contents pages: most lines end in a page number ("... 12"). The first one
        # names itself ("Contents", "Index"); the pages right after it continue the list.
        ratio = sum(bool(TOC_LINE.search(l)) for l in text_lines) / max(len(text_lines), 1)
        toc_like = len(text_lines) >= 8 and ratio > 0.4
        if toc_like and (in_toc or re.search(r"contents|index", " ".join(text_lines[:40]), re.I)):
            in_toc = True
            continue
        in_toc = False
        for l in lines:
            s = l.strip()
            if not s or s in repeated or re.fullmatch(r"(Page\s*)?\d+(\s*of\s*\d+)?", s, re.I):
                continue
            out.append((page_no, s))
    return out


NUM_ONLY = re.compile(r"^\(?\d{1,3}(?:\.\d{1,3}){0,4}[.)]?$")


def join_number_lines(lines):
    """PDFs often print a paragraph number on its own line ("6." then the text): join them."""
    out, i = [], 0
    while i < len(lines):
        page_no, s = lines[i]
        if NUM_ONLY.match(s) and i + 1 < len(lines):
            nxt = lines[i + 1][1]
            num = s if s.endswith((".", ")")) or "." in s else s + "."
            out.append((page_no, f"{num} {nxt}"))
            i += 2
            continue
        out.append((page_no, s))
        i += 1
    return out


def split_clauses(lines):
    """Group lines into clauses using the paragraph-number pattern."""
    lines = join_number_lines(lines)
    clauses, current = [], None
    chapter = ""
    for page_no, line in lines:
        m = CLAUSE_START.match(line)
        if m:
            if m.group("chap") or m.group("annex"):
                chapter = (m.group("chap") or m.group("annex")).strip()
                label = chapter
            else:
                label = m.group("num") or m.group("dnum")
            if current:
                clauses.append(current)
            current = {"clause": label, "chapter": chapter, "page": page_no, "text": line}
        else:
            if current is None:
                current = {"clause": "preamble", "chapter": chapter, "page": page_no, "text": line}
            else:
                current["text"] += " " + line
    if current:
        clauses.append(current)
    return clauses


def size_chunks(clauses):
    """Merge tiny fragments into the previous clause and split very long ones."""
    merged, heading = [], ""
    for c in clauses:
        text = c["text"].strip()
        short = len(text) < config.MIN_CHUNK_CHARS
        if short and not text.endswith((".", ";", ":")):
            # a heading such as "4. Periodic Updation" or "CHAPTER II GENERAL": attach it to the next clause
            heading = (heading + " " + text).strip()
            continue
        c = dict(c)
        if heading:
            c["text"] = heading + " " + text
            heading = ""
        if short and merged:
            merged[-1]["text"] += " " + c["text"]      # a stray sentence fragment: keep with the previous clause
        else:
            merged.append(c)
    if heading and merged:
        merged[-1]["text"] += " " + heading
    out = []
    for c in merged:
        text = re.sub(r"\s+", " ", c["text"]).strip()
        if len(text) <= config.MAX_CHUNK_CHARS:
            out.append({**c, "text": text, "part": 1})
            continue
        sentences = re.split(r"(?<=[.;:])\s+", text)
        buf, part = "", 1
        for s in sentences:
            if buf and len(buf) + len(s) > config.MAX_CHUNK_CHARS:
                out.append({**c, "text": buf.strip(), "part": part})
                buf, part = "", part + 1
            buf += " " + s
        if buf.strip():
            out.append({**c, "text": buf.strip(), "part": part})
    return out


def ingest_pdf(path, issuer):
    doc = pymupdf.open(path)
    pages = [page.get_text("text").splitlines() for page in doc]
    title = (doc.metadata or {}).get("title") or path.stem
    doc.close()
    chunks = size_chunks(split_clauses(clean_lines(pages)))
    doc_id = path.stem
    for i, c in enumerate(chunks):
        c.update({
            "chunk_id": f"{doc_id}::{c['clause']}::{c['part']}::{i}",
            "doc_id": doc_id,
            "issuer": issuer,
            "title": title,
            "source_file": path.name,
        })
    return chunks


def load_registry():
    """data/documents.csv: title, issue date and current/superseded status for each doc_id (= PDF file name)."""
    path = config.ROOT / "data" / "documents.csv"
    if not path.exists():
        return {}
    import csv
    with open(path, encoding="utf-8") as f:
        return {r["doc_id"]: r for r in csv.DictReader(f)}


def main():
    registry = load_registry()
    all_chunks = []
    for issuer in ("rbi", "sebi"):
        folder = config.RAW_DIR / issuer
        folder.mkdir(parents=True, exist_ok=True)   # a fresh clone has no data folders (they are git-ignored)
        pdfs = sorted(folder.glob("*.pdf"))
        for pdf in pdfs:
            try:
                chunks = ingest_pdf(pdf, issuer.upper())
            except Exception as e:  # a broken PDF should not stop the run
                print(f"  ! skipped {pdf.name}: {e}")
                continue
            meta = registry.get(pdf.stem)
            if meta:
                for c in chunks:
                    c.update({"title": meta["title"], "issued": meta["issued"],
                              "status": meta["status"], "superseded_by": meta["superseded_by"]})
            else:
                print(f"  ! {pdf.name} is not listed in data/documents.csv (title/status unknown)")
            print(f"  {issuer.upper():4} {pdf.name}: {len(chunks)} chunks")
            all_chunks.extend(chunks)
    if not all_chunks:
        print("No PDFs found. Put files in data/raw/rbi/ and data/raw/sebi/ first.")
        return
    config.CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(all_chunks)} chunks from "
          f"{len({c['doc_id'] for c in all_chunks})} documents -> {config.CHUNKS_FILE}")


if __name__ == "__main__":
    main()
