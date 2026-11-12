"""Turn 3GPP .docx specifications into clause-tagged chunks.

3GPP Word files mark clause headings with the styles "Heading 1" to "Heading 9" and start the
heading text with the clause number and a tab, e.g. "5.2.1\tSystem information". Every chunk keeps
the spec number and the clause number, so an answer can point to where its context came from.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from . import DATA

HEADING_RE = re.compile(r"^\s*((?:\d+|[A-Z])(?:\.\d+)*)\s+(.+)$")
TOC_STYLES = ("toc", "TOC")


@dataclass
class Clause:
    spec: str
    clause: str
    title: str
    text: str


@dataclass
class Chunk:
    id: str
    spec: str
    clause: str
    title: str
    text: str


def _heading(paragraph) -> tuple[str, str] | None:
    style = (paragraph.style.name if paragraph.style is not None else "") or ""
    if not style.startswith("Heading"):
        return None
    m = HEADING_RE.match(paragraph.text.replace("\t", " ").strip())
    if not m:
        return None
    return m.group(1), m.group(2).strip()


def clauses_from_paragraphs(spec: str, paragraphs) -> list[Clause]:
    """Group paragraphs under the most recent numbered heading. Text before clause 1 (cover,
    foreword, table of contents) is dropped."""
    out: list[Clause] = []
    current: Clause | None = None
    for p in paragraphs:
        style = (p.style.name if p.style is not None else "") or ""
        if style.startswith(TOC_STYLES):
            continue
        h = _heading(p)
        if h is not None:
            if current is not None and current.text.strip():
                out.append(current)
            current = Clause(spec, h[0], h[1], "")
            continue
        if current is not None and p.text.strip():
            current.text += p.text.strip() + "\n"
    if current is not None and current.text.strip():
        out.append(current)
    return out


def read_docx(spec: str, path) -> list[Clause]:
    import docx  # python-docx, imported here so tests of the chunker do not need it

    return clauses_from_paragraphs(spec, docx.Document(str(path)).paragraphs)


def chunk_clause(c: Clause, words: int, overlap: int) -> list[Chunk]:
    toks = c.text.split()
    if not toks:
        return []
    step = max(1, words - overlap)
    out = []
    for i, start in enumerate(range(0, max(1, len(toks) - overlap), step)):
        piece = " ".join(toks[start : start + words])
        out.append(Chunk(f"TS {c.spec} §{c.clause} #{i}", c.spec, c.clause, c.title, piece))
        if start + words >= len(toks):
            break
    return out


def build_chunks(cfg) -> list[Chunk]:
    words = cfg["corpus"]["chunk_words"]
    overlap = cfg["corpus"]["overlap_words"]
    chunks: list[Chunk] = []
    for spec in cfg["corpus"]["specs"]:
        for c in read_docx(spec, DATA / "specs" / f"{spec}.docx"):
            chunks.extend(chunk_clause(c, words, overlap))
    return chunks


def save_chunks(chunks: list[Chunk], path=None) -> None:
    path = path or DATA / "chunks.jsonl"
    with open(path, "w") as f:
        for ch in chunks:
            f.write(json.dumps(asdict(ch)) + "\n")


def load_chunks(path=None) -> list[Chunk]:
    path = path or DATA / "chunks.jsonl"
    with open(path) as f:
        return [Chunk(**json.loads(line)) for line in f]
