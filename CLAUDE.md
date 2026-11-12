# Does the Model Speak 3GPP? Grounding a Local LLM in 5G NR Standards

Personal research project by Rakesh Reddy Karri. Measures how much a small local LLM gains on
5G NR standards questions when it can read the actual 3GPP specifications, and whether an acronym
glossary built from the 3GPP vocabulary helps on top of plain retrieval.

## Research questions
1. Closed book: how well does a 3B local model answer RAN standards questions on its own?
2. Retrieval: does retrieving clauses from the Release 17 NR specifications improve accuracy?
3. Glossary: does expanding 3GPP acronyms (TR 21.905 plus each spec's abbreviation clause) in the
   query and the prompt help on top of plain retrieval?
4. Errors: when the model is wrong with retrieval, was the needed text missing from the context, or
   was it there and the model still chose wrong? When does retrieval make a right answer wrong?

## Data
- Questions: TeleQnA (netop-team, MIT licence; arXiv 2310.15051). The dataset is password
  protected to avoid training contamination, so it is downloaded at run time into `data/` (git
  ignored) and never committed. Only question ids are committed.
- Selection (frozen before any model call): categories "Standards specifications" and "Standards
  overview", tagged `[3GPP Release 17]`, matching a fixed RAN keyword list (`configs/experiment.yaml`).
  Seeded sample (seed 42) of up to 300 questions. The id list is committed in
  `results/question_ids.json` before the first run.
- Corpus: latest Release 17 versions of TS 38.300, 38.331, 38.321, 38.211, 38.213, 38.214, 38.304,
  38.401 and 37.340 from the
  3GPP archive; TR 21.905 for the vocabulary. Downloaded by `make data`, git ignored.

## Setups (the only thing that changes between them is the prompt context)
- A closed_book: question and options only.
- B retrieval: top k clause chunks retrieved with the question as the query.
- C retrieval_glossary: query expanded with acronym definitions, same top k, plus a short glossary
  block in the prompt.

## Model and settings
- Ollama, `qwen2.5:3b`, temperature 0, fixed seed, `num_ctx` 8192 always sent, `num_predict` 8.
- The model must answer with the option number only; the parser takes the first digit 1 to 5.
- Unparseable replies count as wrong and are reported separately.
- Embeddings: sentence-transformers `all-MiniLM-L6-v2`, cosine similarity, chunk ~220 words with
  40 word overlap inside a clause. Every chunk keeps spec number and clause number.

## Analysis
- Accuracy per setup with 95% bootstrap CI over questions (2000 resamples, seed 42).
- Paired comparisons with an exact McNemar test (B vs A, C vs B, C vs A).
- Error groups for B and C: `missing_context` (answer text not in retrieved chunks),
  `context_present` (answer text found, still wrong). Context hit = at least 60% of the answer's
  content words appear in one retrieved chunk. This is a heuristic and is reported as one.
- Flips: right in A and wrong in B or C ("retrieval hurt") and the reverse ("retrieval helped").
- Acronym questions (question contains a glossary acronym) reported separately.

## Engineering rules
- uv, Python 3.12, ruff, pytest. Tests marked `ollama` are skipped on CI.
- Runs are resumable: answers append to `runs/answers.jsonl`, keyed by (setup, question id).
- `runs/` and `data/` are git ignored. Only aggregates, tables and figures are committed.
- Never print question text or answers into committed files (contamination rule of TeleQnA).
- Report every result, including negative ones.
