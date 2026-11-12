from types import SimpleNamespace

import numpy as np
import pytest

from speaks3gpp import analysis, glossary, prompts, questions, specs
from speaks3gpp.download import latest_version
from speaks3gpp.llm import FakeLLM, OllamaLLM
from speaks3gpp.retrieve import Index, expanded_query
from speaks3gpp.run import answer_one

CFG = {
    "seed": 42,
    "questions": {
        "categories": ["Standards specifications"],
        "release_tag": "[3GPP Release 17]",
        "max_questions": 2,
        "acronyms": ["NR", "RRC", "UE"],
        "phrases": ["handover"],
    },
    "retrieval": {"top_k": 2, "max_glossary_terms": 8},
    "analysis": {"bootstrap_resamples": 200, "context_hit_threshold": 0.6},
}


def para(text, style="Normal"):
    return SimpleNamespace(text=text, style=SimpleNamespace(name=style))


# ---------- specs ----------


def test_clauses_follow_numbered_headings_and_skip_front_matter():
    ps = [
        para("Foreword text"),
        para("1\tScope", "Heading 1"),
        para("This document describes NR."),
        para("5.2.1\tSystem information", "Heading 3"),
        para("SIB1 is broadcast."),
        para("TOC line", "TOC 1"),
        para("Unnumbered heading", "Heading 2"),
        para("still under 5.2.1"),
    ]
    cl = specs.clauses_from_paragraphs("38.300", ps)
    assert [(c.clause, c.title) for c in cl] == [("1", "Scope"), ("5.2.1", "System information")]
    assert "Foreword" not in "".join(c.text for c in cl)
    assert "still under 5.2.1" in cl[1].text


def test_annex_headings_are_clauses():
    cl = specs.clauses_from_paragraphs("38.331", [para("A.1\tExample", "Heading 2"), para("x y")])
    assert cl[0].clause == "A.1"


def test_chunks_keep_clause_id_and_overlap():
    c = specs.Clause("38.300", "4.1", "Overall", " ".join(f"w{i}" for i in range(500)))
    ch = specs.chunk_clause(c, words=220, overlap=40)
    assert ch[0].id == "TS 38.300 §4.1 #0"
    assert ch[0].text.split()[-40:] == ch[1].text.split()[:40]
    assert ch[-1].text.split()[-1] == "w499"
    assert all(len(x.text.split()) <= 220 for x in ch)


def test_short_clause_is_one_chunk():
    c = specs.Clause("38.300", "4", "t", "a b c")
    assert len(specs.chunk_clause(c, 220, 40)) == 1


# ---------- glossary ----------


def test_glossary_entries_and_lookup():
    lines = [
        "RRC\tRadio Resource Control",
        "MAC\tMedium Access Control",
        "not an entry",
        "MAC\tMessage Authentication Code",
    ]
    e = glossary.entries_from_clause_lines(lines)
    assert ("RRC", "Radio Resource Control") in e and len(e) == 3
    g = {
        "RRC": ["Radio Resource Control"],
        "SSB": ["SS/PBCH block"],
        "THE": ["x"],
        "Slot": ["y"],
        "N3": ["interface"],
    }
    terms = glossary.find_terms("How does RRC use the SSB and the Slot over N3", g)
    assert terms == ["RRC", "SSB", "N3"]
    assert glossary.describe(["RRC"], g) == ["RRC: Radio Resource Control"]


def test_expanded_query_adds_definitions_only_when_terms_exist():
    g = {"BWP": ["Bandwidth Part"]}
    assert expanded_query("What is a BWP", g, 8) == "What is a BWP (BWP: Bandwidth Part)"
    assert expanded_query("plain text", g, 8) == "plain text"


# ---------- questions ----------

RAW = {
    "question 1": {
        "question": "What does RRC do? [3GPP Release 17]",
        "option 1": "a",
        "option 2": "b",
        "answer": "option 2: b",
        "category": "Standards specifications",
    },
    "question 2": {
        "question": "Is this about Wi-Fi? [IEEE 802.11]",
        "option 1": "a",
        "option 2": "b",
        "answer": "option 1: a",
        "category": "Standards specifications",
    },
    "question 3": {
        "question": "When is handover done? [3GPP Release 17]",
        "option 1": "a",
        "option 2": "b",
        "option 3": "c",
        "answer": "option 3: c",
        "category": "Standards specifications",
    },
    "question 4": {
        "question": "What is NRF? [3GPP Release 17]",
        "option 1": "a",
        "option 2": "b",
        "answer": "option 1: a",
        "category": "Standards specifications",
    },
    "question 5": {
        "question": "What does RRC do? [3GPP Release 18]",
        "option 1": "a",
        "option 2": "b",
        "answer": "option 1: a",
        "category": "Standards specifications",
    },
}


def test_selection_rule_is_fixed_and_seeded():
    ids, pool = questions.select(RAW, CFG)
    assert pool == 2  # 1 and 3; not IEEE, not NRF (NR inside a word), not Release 18
    assert ids == ["question 1", "question 3"]
    assert questions.select(RAW, CFG) == (ids, pool)


def test_parse_question():
    q = questions.parse("question 3", RAW["question 3"])
    assert q.answer == 3 and q.options == {1: "a", 2: "b", 3: "c"}


# ---------- prompts and parsing ----------


def test_parse_answer():
    assert prompts.parse_answer("3", 5) == 3
    assert prompts.parse_answer("The answer is option 2.", 4) == 2
    assert prompts.parse_answer("5", 4) is None
    assert prompts.parse_answer("no idea", 4) is None


def test_prompt_only_context_changes():
    q = questions.parse("question 3", RAW["question 3"])
    a = prompts.user_prompt(q)
    c = specs.Chunk("TS 38.300 §9 #0", "38.300", "9", "Mobility", "handover text")
    b = prompts.user_prompt(q, [c], ["HO: Handover"])
    assert a in b and "TS 38.300 clause 9 Mobility" in b and "HO: Handover" in b


# ---------- run with a fake model and a tiny index ----------


def tiny_index():
    chunks = [
        specs.Chunk("TS 38.300 §1 #0", "38.300", "1", "Scope", "c answer text"),
        specs.Chunk("TS 38.331 §2 #0", "38.331", "2", "RRC", "other text"),
    ]
    vecs = np.eye(2, dtype=np.float32)
    return Index(chunks, vecs, lambda q: np.array([1.0, 0.0], dtype=np.float32))


def test_answer_one_records_everything_for_each_setup():
    q = questions.parse("question 3", RAW["question 3"])
    llm = FakeLLM(answer="3")
    for setup in prompts.SETUPS:
        r = answer_one(llm, q, setup, tiny_index(), {"RRC": ["Radio Resource Control"]}, CFG)
        assert r["correct"] and r["parsed"]
        assert (r["chunk_ids"] == []) == (setup == "closed_book")
    assert "Excerpts" not in llm.calls[0][1] and "Excerpts" in llm.calls[1][1]


def test_num_ctx_guard():
    with pytest.raises(ValueError):
        OllamaLLM(num_ctx=131072)


# ---------- analysis ----------


def test_mcnemar_and_bootstrap():
    assert analysis.mcnemar_exact(0, 0) == 1.0
    assert analysis.mcnemar_exact(10, 0) < 0.01
    assert 0.9 < analysis.mcnemar_exact(5, 6) <= 1.0
    lo, hi = analysis.bootstrap_ci(np.array([1.0] * 50 + [0.0] * 50), 500, 1)
    assert lo < 0.5 < hi


def test_context_hit():
    assert analysis.context_hit("Radio Resource Control", ["the radio resource control layer"], 0.6)
    assert not analysis.context_hit("Radio Resource Control", ["nothing related"], 0.6)


def test_latest_version_picks_newest_release_zip():
    html = '<a href="38300-h00.zip"> <a href="38300-hc0.zip"> <a href="38300-i10.zip">'
    assert latest_version(html, "38.300", "h") == "38300-hc0.zip"
    assert latest_version(html, "38.300", "g") is None


# ---------- live model (skipped on CI) ----------


@pytest.mark.ollama
def test_ollama_answers_a_digit():
    try:
        r = OllamaLLM().chat(prompts.SYSTEM, "Question: 1+1?\nOptions:\n1. 3\n2. 2\nAnswer.")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no ollama: {e}")
    assert prompts.parse_answer(r.text, 2) is not None


# ---------- manual review ----------


def test_review_summary_counts_labels_and_rejects_unknown(tmp_path, monkeypatch):
    import csv
    import json

    from speaks3gpp import review

    sheet = tmp_path / "sheet.csv"
    monkeypatch.setattr(review, "SHEET", sheet)
    monkeypatch.setattr(review, "RESULTS", tmp_path)
    rows = [
        {"qid": "q1", "auto_group": "missing_context", "label": "wrong_clause"},
        {"qid": "q2", "auto_group": "context_present", "label": "context_ignored"},
        {"qid": "q3", "auto_group": "missing_context", "label": ""},
    ]
    with open(sheet, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=review.FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in review.FIELDS})
    out = review.summarise()
    assert out["labelled"] == 2 and out["counts"]["wrong_clause"] == 1
    assert out["auto_vs_manual_agreement"] == 1.0
    assert json.loads((tmp_path / "manual_review.json").read_text())["sampled"] == 3
    rows[2]["label"] = "typo"
    with open(sheet, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=review.FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in review.FIELDS})
    with pytest.raises(SystemExit):
        review.summarise()
