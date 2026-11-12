"""Answer every selected question in the three setups. Resumable: one JSON line per
(setup, question id) in runs/answers.jsonl, already answered pairs are skipped."""

from __future__ import annotations

import argparse
import json
import time

from . import RUNS, config
from . import glossary as gl
from .llm import LLM, OllamaLLM
from .prompts import SETUPS, SYSTEM, parse_answer, user_prompt
from .questions import Question, load_selected
from .retrieve import Index, expanded_query, load_index

ANSWERS = "answers.jsonl"


def done_keys(path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    with open(path) as f:
        return {(r["setup"], r["qid"]) for r in map(json.loads, f)}


def answer_one(llm: LLM, q: Question, setup: str, index: Index | None, glossary, cfg) -> dict:
    k = cfg["retrieval"]["top_k"]
    limit = cfg["retrieval"]["max_glossary_terms"]
    chunks, scores, gloss_lines, query = [], [], None, None
    if setup == "retrieval":
        query = q.text
    elif setup == "retrieval_glossary":
        query = expanded_query(q.text, glossary, limit)
        terms = gl.find_terms(q.text, glossary, limit)
        gloss_lines = gl.describe(terms, glossary) if terms else None
    if query is not None:
        hits = index.search(query, k)
        chunks = [c for c, _ in hits]
        scores = [s for _, s in hits]
    reply = llm.chat(SYSTEM, user_prompt(q, chunks, gloss_lines))
    pred = parse_answer(reply.text, max(q.options))
    return {
        "setup": setup,
        "qid": q.id,
        "pred": pred,
        "gold": q.answer,
        "correct": pred == q.answer,
        "parsed": pred is not None,
        "reply": reply.text,
        "chunk_ids": [c.id for c in chunks],
        "scores": [round(s, 4) for s in scores],
        "glossary_terms": len(gloss_lines or []),
        "prompt_tokens": reply.prompt_tokens,
        "wall_s": round(reply.wall_s, 3),
        "model": llm.model,
    }


def run(llm: LLM, questions: list[Question], index, glossary, cfg, out_path, setups=SETUPS):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(out_path)
    todo = [(s, q) for s in setups for q in questions if (s, q.id) not in done]
    print(f"{len(done)} done, {len(todo)} to go")
    t0 = time.time()
    with open(out_path, "a") as f:
        for i, (s, q) in enumerate(todo, 1):
            rec = answer_one(llm, q, s, index, glossary, cfg)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                rate = (time.time() - t0) / i
                eta = rate * (len(todo) - i) / 60
                print(f"{i}/{len(todo)}  {rate:.1f} s/question  eta {eta:.0f} min")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (pilot)")
    args = ap.parse_args()
    cfg = config.load(args.config) if args.config else config.load()
    m = cfg["model"]
    llm = OllamaLLM(
        m["name"], m["url"], m["num_ctx"], m["num_predict"], m["temperature"], cfg["seed"]
    )
    questions = load_selected()
    if args.limit:
        questions = questions[: args.limit]
    run(llm, questions, load_index(cfg), gl.load(), cfg, RUNS / ANSWERS)


if __name__ == "__main__":
    main()
