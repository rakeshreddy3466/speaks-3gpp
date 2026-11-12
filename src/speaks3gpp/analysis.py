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
from .specs import Chunk, load_chunks

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
    df: pd.DataFrame, questions: list[Question], chunks: dict[str, Chunk], glossary, cfg
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
                    "mcnemar_p": float(f"{mcnemar_exact(only_a, only_b):.2g}"),
                }
            )

    out["error_groups"] = {}
    hit_of: dict[str, dict[str, bool]] = {}
    rows = df.set_index(["setup", "qid"])
    for s in ("retrieval", "retrieval_glossary"):
        if s not in setups:
            continue
        groups = {"missing_context": 0, "context_present": 0, "unparsed": 0}
        hits = 0
        hit_of[s] = {}
        for qid in ids:
            r = rows.loc[(s, qid)]
            q = qmap[qid]
            texts = [chunks[c].text for c in r["chunk_ids"]]
            hit = context_hit(q.options[q.answer], texts, thr)
            hit_of[s][qid] = hit
            hits += hit
            if r["correct"]:
                continue
            if not r["parsed"]:
                groups["unparsed"] += 1
            elif hit:
                groups["context_present"] += 1
            else:
                groups["missing_context"] += 1
        right_hit = [bool(wide.loc[q, s]) for q in ids if hit_of[s][q]]
        right_miss = [bool(wide.loc[q, s]) for q in ids if not hit_of[s][q]]
        out["error_groups"][s] = {
            **groups,
            "context_hit_rate": round(hits / len(ids), 4),
            "accuracy_when_hit": {"n": len(right_hit), "accuracy": round(np.mean(right_hit), 4)},
            "accuracy_when_missed": {
                "n": len(right_miss),
                "accuracy": round(np.mean(right_miss), 4),
            },
        }

    out["flips_vs_closed_book"] = {}
    for s in ("retrieval", "retrieval_glossary"):
        if s in setups:
            out["flips_vs_closed_book"][s] = {
                "helped": int((~wide["closed_book"] & wide[s]).sum()),
                "hurt": int((wide["closed_book"] & ~wide[s]).sum()),
                "both_right": int((wide["closed_book"] & wide[s]).sum()),
                "both_wrong": int((~wide["closed_book"] & ~wide[s]).sum()),
                "hurt_with_answer_in_context": int(
                    sum(
                        hit_of[s][q]
                        for q in ids
                        if wide.loc[q, "closed_book"] and not wide.loc[q, s]
                    )
                ),
            }

    out["diagnostics"] = diagnostics(df, ids, chunks)
    return out


def diagnostics(df: pd.DataFrame, ids: list[str], chunks: dict[str, Chunk]) -> dict:
    """Why did the glossary change so little? Compare what B and C retrieved."""
    by = {s: df[df.setup == s].set_index("qid") for s in ("retrieval", "retrieval_glossary")}
    if any(len(v) == 0 for v in by.values()):
        return {}
    b, c = by["retrieval"], by["retrieval_glossary"]
    overlap = [len(set(b.loc[q, "chunk_ids"]) & set(c.loc[q, "chunk_ids"])) for q in ids]
    k = len(b.loc[ids[0], "chunk_ids"])

    def def_share(frame) -> float:
        all_ids = [i for q in ids for i in frame.loc[q, "chunk_ids"]]
        bad = sum(
            any(w in chunks[i].title.lower() for w in ("abbreviation", "definition"))
            for i in all_ids
        )
        return round(bad / len(all_ids), 4)

    terms = c.loc[ids, "glossary_terms"]
    return {
        "same_chunks_b_and_c": int(sum(o == k for o in overlap)),
        "mean_chunk_overlap_b_c": round(float(np.mean(overlap)) / k, 4),
        "questions_without_glossary_terms": int((terms == 0).sum()),
        "mean_glossary_terms": round(float(terms.mean()), 2),
        "definition_or_abbreviation_chunk_share": {
            "retrieval": def_share(b),
            "retrieval_glossary": def_share(c),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = config.load(args.config) if args.config else config.load()
    chunks = {c.id: c for c in load_chunks()}
    res = analyse(load_answers(), load_selected(), chunks, gl.load(), cfg)
    (RESULTS / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
