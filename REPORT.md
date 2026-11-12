# Does the Model Speak 3GPP? Grounding a Local LLM in 5G NR Standards

Rakesh Reddy Karri, November 2026

## 1. Why

3GPP specifications are the rulebook of mobile networks, and they are long, dense and full of
acronyms (TS 38.331 alone is well over a thousand pages). Engineers increasingly ask LLMs about
them. A small local model is attractive (cheap, private, runs on a laptop) but it guesses. This
project measures, on public data, how much a 3B model gains from reading the actual Release 17
NR specifications, whether explaining 3GPP acronyms helps further, and where the remaining errors
come from.

Research questions:

1. **Closed book:** how well does `qwen2.5:3b` answer RAN standards questions on its own?
2. **Retrieval:** does retrieving clauses from the Release 17 NR specs improve accuracy?
3. **Glossary:** does expanding acronyms (TR 21.905 and each spec's abbreviation clause) in the
   query and the prompt help on top of retrieval?
4. **Errors:** when retrieval answers are wrong, was the needed text missing, or present and
   misread? When does retrieval turn a right answer wrong?

## 2. Design

**Frozen before any model call.** The question rule, the setups, the model settings, the chunk size
and top k were fixed in `configs/experiment.yaml` and `CLAUDE.md` in the first commit, and the 300
question ids were committed (`results/question_ids.json`, commit "Freeze 300 question ids...")
before the first answer was generated. Nothing was tuned on the results.

**Questions.** TeleQnA (10,000 machine generated multiple choice questions from telecom standards
and papers). Kept: categories "Standards specifications" and "Standards overview", tag
`[3GPP Release 17]`, and a RAN term in the question (acronyms such as RRC, SSB, BWP, HARQ, gNB, UE,
or phrases such as handover, beam, paging). 336 matched; a seeded sample (seed 42) of 300: 269
specification and 31 overview questions, 197 with five options and 103 with four (chance 21.7%).
The TeleQnA authors protect the dataset with a password to avoid training contamination, so no
question text is committed anywhere in this repo.

**Corpus.** The newest Release 17 version of nine specs, downloaded from the 3GPP archive: TS 38.300
(NR overall description), 38.331 (RRC), 38.321 (MAC), 38.211 (physical channels), 38.213 and 38.214
(physical layer procedures for control and data), 38.304 (idle mode), 38.401 (NG-RAN architecture),
37.340 (multi-connectivity). The Word files are parsed by heading style; text under each numbered
heading becomes a clause, and clauses are cut into windows of 220 words with 40 words overlap.
Result: 5,164 chunks (38.331: 1,534; 38.213: 790; 38.214: 672; 38.321: 637; 38.300: 609; others
fewer). Each chunk keeps its spec and clause, for example `TS 38.331 §4.3.1 #0`.

**Glossary.** 390 acronyms from the abbreviation clauses of the nine specs and TR 21.905. When an
acronym has several meanings, spec-local ones come first and at most two are shown. Only tokens
with at least two capitals (or a capital and a digit) count as acronyms, so words like "Slot" are
not expanded.

**Setups.** Same system prompt ("reply with the option number only") and same question block; only
the context differs.

| | Query used for search | Context in the prompt |
|---|---|---|
| A closed book | none | none |
| B retrieval | the question | top 4 chunks, with spec and clause |
| C retrieval + glossary | question plus "(RRC: Radio Resource Control; ...)" | glossary lines plus top 4 chunks |

**Model and search.** Ollama `qwen2.5:3b`, temperature 0, seed 42, `num_ctx` 8192, at most 8 output
tokens; the first digit 1 to 5 in the reply is the answer. Embeddings: `all-MiniLM-L6-v2`,
normalised, exact cosine search in NumPy.

**Metrics.** Accuracy with a 95% bootstrap CI over questions (2,000 resamples, seed 42). Paired
exact McNemar tests on the same questions. A "context hit" means at least 60% of the content words
of the correct option appear in one retrieved chunk; wrong answers are split into missing context
(no hit) and context present (hit). Flips compare each question with closed book.

## 3. Results

| | A closed book | B retrieval | C retrieval + glossary |
|---|---|---|---|
| Accuracy | 52.0% | **65.7%** | 64.0% |
| 95% CI | 46.3 to 57.7 | 60.3 to 71.3 | 58.7 to 69.7 |
| Correct | 156 | 197 | 192 |
| Questions with a glossary acronym (n=235) | 53.6% | 65.1% | 63.0% |
| Questions without (n=65) | 46.2% | 67.7% | 67.7% |
| Median prompt tokens | 125 | 1,290 | 1,315 |
| Median seconds per question | 0.27 | 4.77 | 5.39 |

All 900 replies parsed.

**RQ1, closed book.** 52.0%, far above chance (21.7%) but wrong about half the time. The model
knows a lot of 3GPP vocabulary from pre-training and still guesses on details.

**RQ2, retrieval.** +13.7 points. 69 questions went from wrong to right and 28 from right to wrong
(exact McNemar p = 3.8e-05). Retrieval clearly helps, and it is not free: about 10 times more
prompt tokens and 5 s instead of 0.3 s per question on an 8 GB laptop.

**RQ3, glossary.** No measurable gain: 64.0% against 65.7% for plain retrieval; 12 questions only C
got right, 17 only B (p = 0.46). Even on the 235 questions that contain a glossary acronym, C was
not better (63.0% vs 65.1%). Section 4 looks at why.

**RQ4, errors and flips.**

| | B retrieval | C retrieval + glossary |
|---|---|---|
| Answer text in the top 4 chunks | 53.0% of questions | 51.0% |
| Accuracy when it was | 74.2% (n=159) | 75.8% (n=153) |
| Accuracy when it was not | 56.0% (n=141) | 51.7% (n=147) |
| Wrong answers: missing context | 62 | 71 |
| Wrong answers: context present | 41 | 37 |
| Right to wrong vs closed book | 28 (11 with the answer in context) | 30 (11) |

The clearest finding of the project: **when search finds the right text, the 3B model mostly reads
it correctly (74 to 76%); when it does not, accuracy falls back to closed book level.** About 60%
of the remaining errors are search errors, not reading errors. The 28 questions that retrieval
broke split in two: in 17 the answer text was not in the excerpts and they pulled a correct guess away; in 11 the
answer was in the excerpts and the model still moved away; in 9 of those 11 the option it chose
also appears in the excerpts (word overlap test), so it picked a plausible neighbour from the
same text.

## 4. Why the glossary did not help

(These diagnostics were added after the run, to explain the RQ3 result. They do not change any
pre-registered number.)

| Diagnostic | Value |
|---|---|
| Questions where B and C retrieved the same 4 chunks | 137 of 300 |
| Mean share of chunks shared by B and C | 81% |
| Questions with no glossary acronym (C prompt = B prompt plus nothing) | 65 |
| Mean glossary acronyms per question | 1.3 |
| Retrieved chunks that are definition or abbreviation lists, B / C | 2.9% / 3.0% |

My first guess was that the expanded query would pull in the abbreviation lists themselves and
crowd out real content. It did not (3% in both). The simpler explanation fits the data: the
expansion changed about one of the four excerpts, and the new one was as often worse as better
(context hit 51% vs 53%). The model did not need the definitions to read the excerpts; with the
right clause in front of it, B and C were equally accurate (74% vs 76%). So acronyms are not the
bottleneck for this model on these questions; finding the right clause is.

## 5. Manual review of wrong answers

`make review` writes a seeded sample of 50 wrong answers from setup B (28 automatically grouped as
missing context, 22 as context present) to a local sheet with the question, both options and the
start of each excerpt. Each one is read by hand and labelled: missing context, wrong clause (right
topic, wrong clause), misread acronym, context ignored, question issue, or other.
`make review-summary` writes the counts to `results/manual_review.json`.

*Status: labelling in progress; counts will be added here.*

One early observation from the sheet: some TeleQnA "standards" questions have more than one
defensible option. For example, a question asks what RRC offers to upper layers, the retrieved
TS 38.331 clause 4.3.1 lists several services, and two of the options are in that list. Such
questions cap the accuracy any model can reach and are labelled "question issue".

## 6. What a RAN specialist would need to check before trusting this

- **The source, not just the answer.** Every excerpt carries spec and clause, so an engineer can
  open the clause. Without that the 74% would not be usable.
- **Release and version.** Answers come from one Release 17 version per spec; a deployment on
  another release needs the matching documents.
- **Questions outside the nine specs.** About half the questions did not get the answer text into
  the context. Part of that is likely search, part a needed spec (for example a core network or
  sidelink spec) that is not in the corpus; the manual review separates the two. The tool should say "not found in the corpus" rather than guess.
- **Options that are all partly right.** Machine generated questions and lists of services make
  "pick one" ambiguous; a specialist should review these cases.

## 7. Limitations

- One 3B model, one embedding model, one chunk size, top 4; nothing tuned (by design), so the
  numbers are a baseline, not a best case.
- 300 questions: the CIs are about plus or minus 5.5 points, and B vs C (1.7 points) is noise.
- TeleQnA is machine generated; some questions are ambiguous or have debatable answers.
- The context hit test is word overlap, so it can miss paraphrases and count lucky overlaps; the
  manual review checks it on 50 cases.
- Nine specs only; the selection rule (RAN keywords incl. "UE") lets in some questions whose
  answer lives outside them.

## 8. Next steps

1. Hybrid search (BM25 keyword search plus embeddings) and re-ranking, since search is the
   bottleneck; acronyms and clause numbers suit keyword search.
2. Let the model say "not in the excerpts", and measure how often abstaining avoids the 17 flips
   where the answer was not in the excerpts.
3. Add the missing specs that the manual review points to, and repeat with a 7B model.
