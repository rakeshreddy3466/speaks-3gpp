"""Acronym glossary from the 3GPP vocabulary (TR 21.905) and each spec's abbreviation clause.

Abbreviation clauses list one term per paragraph as "ABBR<tab>Expansion". The same acronym can
have several expansions in 3GPP (for example "MAC"), so every expansion is kept, spec-local ones
first, and the prompt shows at most two.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

from . import DATA
from .specs import Clause

ENTRY_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9\-/_+]{1,19})\t+(\S.{2,200})$")
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/]*[A-Za-z0-9]")
# short all-caps words that are also English or too generic to expand
STOP = set("A AN AND OR IN IS IT OF ON TO THE FOR NOT ALL ID".split())


def entries_from_clause_lines(lines: list[str]) -> list[tuple[str, str]]:
    out = []
    for line in lines:
        m = ENTRY_RE.match(line.strip("\n"))
        if m:
            out.append((m.group(1).strip(), m.group(2).strip()))
    return out


def entries_from_clauses(clauses: list[Clause]) -> list[tuple[str, str]]:
    out = []
    for c in clauses:
        if "abbreviation" in c.title.lower():
            out.extend(entries_from_clause_lines(c.text.split("\n")))
    return out


def _raw_lines_by_abbrev_clause(spec: str, path) -> list[str]:
    """Abbreviation entries keep their tab only in the raw paragraph text, so read raw lines."""
    import docx

    lines, inside = [], False
    for p in docx.Document(str(path)).paragraphs:
        style = (p.style.name if p.style is not None else "") or ""
        if style.startswith("Heading"):
            inside = "abbreviation" in p.text.lower()
            continue
        if inside and p.text.strip():
            lines.append(p.text)
    return lines


def build(cfg) -> dict[str, list[str]]:
    glossary: dict[str, list[str]] = defaultdict(list)
    sources = [*cfg["corpus"]["specs"], cfg["corpus"]["vocabulary"]]  # spec-local first
    for spec in sources:
        path = DATA / "specs" / f"{spec}.docx"
        for abbr, exp in entries_from_clause_lines(_raw_lines_by_abbrev_clause(spec, path)):
            if exp not in glossary[abbr]:
                glossary[abbr].append(exp)
    return dict(glossary)


def save(glossary: dict[str, list[str]], path=None) -> None:
    (path or DATA / "glossary.json").write_text(json.dumps(glossary, indent=1, sort_keys=True))


def load(path=None) -> dict[str, list[str]]:
    return json.loads((path or DATA / "glossary.json").read_text())


def find_terms(text: str, glossary: dict[str, list[str]], limit: int = 8) -> list[str]:
    """Acronyms from the glossary that occur in the text, in order of first appearance.

    Only tokens with at least two capitals (or a capital and a digit) count, so ordinary words
    such as "Slot" or "Data" are not expanded even if they appear in the glossary."""
    seen: list[str] = []
    for tok in TOKEN_RE.findall(text):
        caps = sum(ch.isupper() for ch in tok)
        digits = sum(ch.isdigit() for ch in tok)
        if caps < 2 and not (caps >= 1 and digits >= 1):
            continue
        if tok in STOP or tok not in glossary or tok in seen:
            continue
        seen.append(tok)
        if len(seen) >= limit:
            break
    return seen


def describe(terms: list[str], glossary: dict[str, list[str]], per_term: int = 2) -> list[str]:
    return [f"{t}: {' / '.join(glossary[t][:per_term])}" for t in terms]
