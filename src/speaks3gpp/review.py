"""Manual error review: read wrong answers by hand and label why they went wrong.

`make review` writes data/review_sheet.csv (git ignored, it contains TeleQnA text): a seeded
sample of wrong answers from setup B (retrieval) with the question, the correct and the chosen
option, and the start of each retrieved excerpt. Fill the `label` column with one of LABELS.
`make review-summary` counts the labels into results/manual_review.json (counts only, no text).
"""

from __future__ import annotations

import argparse
import csv
import json
import random

from . import DATA, RESULTS, config
from .analysis import context_hit, load_answers
from .questions import load_selected
from .specs import load_chunks

LABELS = {
    "missing_context": "no excerpt is about the topic of the question",
    "wrong_clause": "excerpts are on the right topic but from a clause without the answer",
    "misread_acronym": "an acronym was misread or matched with the wrong meaning",
    "context_ignored": "the answer is in an excerpt but the model chose another option",
    "question_issue": "the question or its official answer is ambiguous or wrong",
    "other": "anything else (explain in the note column)",
}
SHEET = DATA / "review_sheet.csv"
FIELDS = [
    "qid",
    "auto_group",
    "question",
    "correct_option",
    "model_option",
    "excerpts",
    "label",
    "note",
]


def build_sheet(n: int, setup: str, cfg) -> int:
    df = load_answers()
    qs = {q.id: q for q in load_selected()}
    chunks = {c.id: c for c in load_chunks()}
    wrong = df[(df.setup == setup) & (~df.correct)]
    rng = random.Random(cfg["seed"])
    picked = sorted(
        rng.sample(list(wrong.qid), min(n, len(wrong))), key=lambda s: int(s.split()[-1])
    )
    rows = wrong.set_index("qid")
    thr = cfg["analysis"]["context_hit_threshold"]
    with open(SHEET, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for qid in picked:
            r, q = rows.loc[qid], qs[qid]
            texts = [chunks[c].text for c in r["chunk_ids"]]
            hit = context_hit(q.options[q.answer], texts, thr)
            excerpts = "\n\n".join(
                f"[{c}] {chunks[c].title}: {chunks[c].text[:300]}..." for c in r["chunk_ids"]
            )
            chosen = q.options.get(r["pred"], "(unparsed)") if r["pred"] else "(unparsed)"
            w.writerow(
                {
                    "qid": qid,
                    "auto_group": "context_present" if hit else "missing_context",
                    "question": q.text,
                    "correct_option": f"{q.answer}: {q.options[q.answer]}",
                    "model_option": f"{r['pred']}: {chosen}",
                    "excerpts": excerpts,
                    "label": "",
                    "note": "",
                }
            )
    return len(picked)


def summarise() -> dict:
    with open(SHEET, newline="") as f:
        rows = list(csv.DictReader(f))
    labelled = [r for r in rows if r["label"].strip()]
    bad = {r["label"] for r in labelled} - set(LABELS)
    if bad:
        raise SystemExit(f"unknown labels {bad}; use one of {sorted(LABELS)}")
    counts = {k: sum(r["label"].strip() == k for r in labelled) for k in LABELS}
    agree = sum(
        (r["auto_group"] == "missing_context")
        == (r["label"] in ("missing_context", "wrong_clause"))
        for r in labelled
    )
    out = {
        "setup": "retrieval",
        "sampled": len(rows),
        "labelled": len(labelled),
        "counts": counts,
        "label_definitions": LABELS,
        "auto_vs_manual_agreement": round(agree / len(labelled), 3) if labelled else None,
    }
    (RESULTS / "manual_review.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sheet", "summary"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--setup", default="retrieval")
    args = ap.parse_args()
    cfg = config.load()
    if args.cmd == "sheet":
        n = build_sheet(args.n, args.setup, cfg)
        print(f"{n} wrong answers written to {SHEET}; fill the label column with one of:")
        for k, v in LABELS.items():
            print(f"  {k:16s} {v}")
    else:
        print(json.dumps(summarise(), indent=1))


if __name__ == "__main__":
    main()
