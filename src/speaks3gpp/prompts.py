"""Prompts for the three setups. Only the context block changes between them."""

from __future__ import annotations

import re

from .questions import Question
from .specs import Chunk

SETUPS = ("closed_book", "retrieval", "retrieval_glossary")

SYSTEM = (
    "You answer multiple-choice questions about 3GPP telecom standards. "
    "Reply with the number of the correct option only, for example: 3"
)

DIGIT_RE = re.compile(r"[1-5]")


def options_block(q: Question) -> str:
    return "\n".join(f"{k}. {v}" for k, v in q.options.items())


def context_block(chunks: list[Chunk]) -> str:
    parts = [f"[TS {c.spec} clause {c.clause} {c.title}]\n{c.text}" for c in chunks]
    return "\n\n".join(parts)


def user_prompt(q: Question, chunks: list[Chunk] | None = None, glossary_lines=None) -> str:
    parts = []
    if glossary_lines:
        parts.append("Acronyms (from the 3GPP vocabulary):\n" + "\n".join(glossary_lines))
    if chunks:
        parts.append(
            "Excerpts from the 3GPP specifications (they may or may not contain the answer):\n"
            + context_block(chunks)
        )
    parts.append(
        f"Question: {q.text}\nOptions:\n{options_block(q)}\nAnswer with the option number."
    )
    return "\n\n".join(parts)


def parse_answer(text: str, n_options: int) -> int | None:
    """First digit 1..n in the reply, or None (counted as wrong and reported separately)."""
    for m in DIGIT_RE.finditer(text):
        v = int(m.group(0))
        if 1 <= v <= n_options:
            return v
    return None
