"""Select the TeleQnA questions once, with a fixed rule and seed, and freeze their ids.

Only the ids go to git (results/question_ids.json). The question text stays in data/, which is
git ignored, because the TeleQnA authors ask that the dataset is not published in plain text.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass

from . import DATA, RESULTS, config

OPTION_RE = re.compile(r"^option (\d+)$")


@dataclass
class Question:
    id: str
    text: str
    options: dict[int, str]
    answer: int
    category: str


def parse(qid: str, raw: dict) -> Question:
    options = {}
    for k, v in raw.items():
        m = OPTION_RE.match(k)
        if m and v is not None:
            options[int(m.group(1))] = str(v)
    answer = int(re.match(r"option (\d+)", raw["answer"]).group(1))
    return Question(qid, raw["question"], dict(sorted(options.items())), answer, raw["category"])


def ran_matcher(acronyms: list[str], phrases: list[str]):
    acr = re.compile(
        r"(?<![A-Za-z0-9\-])(" + "|".join(map(re.escape, acronyms)) + r")(?![A-Za-z0-9])"
    )
    phr = re.compile(r"\b(" + "|".join(map(re.escape, phrases)) + r")", re.IGNORECASE)
    return lambda text: bool(acr.search(text) or phr.search(text))


def select(raw: dict[str, dict], cfg) -> tuple[list[str], int]:
    q = cfg["questions"]
    is_ran = ran_matcher(q["acronyms"], q["phrases"])
    pool = [
        qid
        for qid, v in raw.items()
        if v["category"] in q["categories"]
        and q["release_tag"] in v["question"]
        and is_ran(v["question"].replace(q["release_tag"], ""))
    ]
    pool.sort(key=lambda s: int(s.split()[-1]))
    rng = random.Random(cfg["seed"])
    chosen = rng.sample(pool, min(q["max_questions"], len(pool)))
    return sorted(chosen, key=lambda s: int(s.split()[-1])), len(pool)


def load_raw() -> dict[str, dict]:
    return json.loads((DATA / "TeleQnA.json").read_text())


def load_selected() -> list[Question]:
    ids = json.loads((RESULTS / "question_ids.json").read_text())["ids"]
    raw = load_raw()
    return [parse(i, raw[i]) for i in ids]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = config.load(args.config) if args.config else config.load()
    raw = load_raw()
    ids, pool = select(raw, cfg)
    cats: dict[str, int] = {}
    for i in ids:
        cats[raw[i]["category"]] = cats.get(raw[i]["category"], 0) + 1
    out = {
        "rule": "see configs/experiment.yaml: questions",
        "seed": cfg["seed"],
        "pool_size": pool,
        "selected": len(ids),
        "by_category": cats,
        "ids": ids,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "question_ids.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"pool {pool}, selected {len(ids)}, {cats}")


if __name__ == "__main__":
    main()
