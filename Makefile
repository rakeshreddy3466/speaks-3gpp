.PHONY: data questions index pilot run analyse report test lint

# 1. inputs into data/ (git ignored): Release 17 specs from the 3GPP archive, TeleQnA from GitHub
data:
	uv run python -m speaks3gpp.download

# 2. freeze the question ids (commit results/question_ids.json before any model call)
questions:
	uv run python -m speaks3gpp.questions

# 3. parse specs into clause chunks, build the acronym glossary, embed the chunks
index:
	uv run --group retrieval python -m speaks3gpp.retrieve

# 4. model runs (needs Ollama with qwen2.5:3b); resumable
pilot:
	uv run --group retrieval python -m speaks3gpp.run --limit 10
run:
	uv run --group retrieval python -m speaks3gpp.run

# 5. aggregates into results/summary.json (ids and numbers only)
analyse:
	uv run python -m speaks3gpp.analysis

# 6. tables T1-T4 and figures F1-F3 from results/summary.json only
report:
	uv run python -m speaks3gpp.report

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .
