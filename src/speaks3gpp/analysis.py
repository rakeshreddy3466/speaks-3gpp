"""Aggregate runs/answers.jsonl into results/ (no question text, ids and numbers only).

* accuracy per setup with a 95% bootstrap CI over questions
* paired exact McNemar tests between setups
* error groups for the retrieval setups (missing context vs context present)
* flips against closed book (retrieval helped / retrieval hurt)
* acronym questions vs the rest
"""

from __future__ import annotations

import argparse
import json
import math
import re

import numpy as np
import pandas as pd

from . import RESULTS, RUNS, config
from . import glossary as gl
from .prompts import SETUPS
from .questions import Question, load_selected
from .specs import load_chunks

WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")
STOPWORDS = set(
    "a an the of to in on for and or is are be by with as at from that this it its which when "
    "can may shall should will not all only each any both one two none above following".split()
)


def bootstrap_ci(values: np.ndarray, resamples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b (only first right) and c
    (only second right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def content_words(text: str) -> set[str]:
    return {w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 1}


def context_hit(answer_text: str, chunk_texts: list[str], threshold: float) -> bool:
    """True if at least `threshold` of the answer's content words occur in one retrieved chunk.
    A heuristic for 'the needed text was in front of the model', reported as such."""
    words = content_words(answer_text)
    if not words:
        return False
    for t in chunk_texts:
        have = content_words(t)
        if len(words & have) / len(words) >= threshold:
            return True
    return False


def load_answers(path=None) -> pd.DataFrame:
    path = path or RUNS / "answers.jsonl"
    with open(path) as f:
        return pd.DataFrame([json.loads(line) for line in f])


def analyse(
    df: pd.DataFrame, questions: list[Question], chunk_text: dict[str, str], glossary, cfg
) -> dict:
    seed = cfg["seed"]
    n_boot = cfg["analysis"]["bootstrap_resamples"]
    thr = cfg["analysis"]["context_hit_threshold"]
    qmap = {q.id: q for q in questions}
    ids = [q.id for q in questions]
    wide = df.pivot_table(index="qid", columns="setup", values="correct", aggfunc="first")
    wide = wide.reindex(ids)
    setups = [s for s in SETUPS if s in wide.columns]
    if wide[setups].isna().any().any():
        raise SystemExit("some (setup, question) pairs are missing: finish `make run` first")
    wide = wide[setups].astype(bool)
    out: dict = {"n_questions": len(ids), "model": df["model"].iloc[0], "setups": {}}

    acronym_q = {q.id: bool(gl.find_terms(q.text, glossary)) for q in questions}
    chance = float(np.mean([1 / len(q.options) for q in questions]))
    out["chance_accuracy"] = round(chance, 4)

    for s in setups:
        v = wide[s].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(v, n_boot, seed)
        sub = df[df.setup == s]
        acr = np.array([acronym_q[i] for i in ids])
        out["setups"][s] = {
            "accuracy": round(float(v.mean()), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "correct": int(v.sum()),
            "unparsed": int((~sub["parsed"]).sum()),
            "acronym_questions": {"n": int(acr.sum()), "accuracy": round(float(v[acr].mean()), 4)},
            "other_questions": {
                "n": int((~acr).sum()),
                "accuracy": round(float(v[~acr].mean()), 4),
            },
            "median_prompt_tokens": int(sub["prompt_tokens"].median()),
            "median_seconds": round(float(sub["wall_s"].median()), 2),
        }

    pairs = [
        ("retrieval", "closed_book"),
        ("retrieval_glossary", "retrieval"),
        ("retrieval_glossary", "closed_book"),
    ]
    out["paired"] = []
    for a, b in pairs:
        if a in setups and b in setups:
            only_a = int((wide[a] & ~wide[b]).sum())
            only_b = int((~wide[a] & wide[b]).sum())
            out["paired"].append(
                {
                    "comparison": f"{a} vs {b}",
                    "difference_points": round(100 * (wide[a].mean() - wide[b].mean()), 1),
                    "only_first_right": only_a,
                    "only_second_right": only_b,
                    "mcnemar_p": round(mcnemar_exact(only_a, only_b), 4),
                }
            )

    out["error_groups"] = {}
    rows = df.set_index(["setup", "qid"])
    for s in ("retrieval", "retrieval_glossary"):
        if s not in setups:
            continue
        groups = {"missing_context": 0, "context_present": 0, "unparsed": 0}
        hits = 0
        for qid in ids:
            r = rows.loc[(s, qid)]
            q = qmap[qid]
            texts = [chunk_text[c] for c in r["chunk_ids"]]
            hit = context_hit(q.options[q.answer], texts, thr)
            hits += hit
            if r["correct"]:
                continue
            if not r["parsed"]:
                groups["unparsed"] += 1
            elif hit:
                groups["context_present"] += 1
            else:
                groups["missing_context"] += 1
        out["error_groups"][s] = {**groups, "context_hit_rate": round(hits / len(ids), 4)}

    out["flips_vs_closed_book"] = {}
    for s in ("retrieval", "retrieval_glossary"):
        if s in setups:
            out["flips_vs_closed_book"][s] = {
                "helped": int((~wide["closed_book"] & wide[s]).sum()),
                "hurt": int((wide["closed_book"] & ~wide[s]).sum()),
                "both_right": int((wide["closed_book"] & wide[s]).sum()),
                "both_wrong": int((~wide["closed_book"] & ~wide[s]).sum()),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = config.load(args.config) if args.config else config.load()
    chunk_text = {c.id: c.text for c in load_chunks()}
    res = analyse(load_answers(), load_selected(), chunk_text, gl.load(), cfg)
    (RESULTS / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
