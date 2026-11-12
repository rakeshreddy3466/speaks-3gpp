# speaks-3gpp

**Does the model speak 3GPP? Grounding a local LLM in 5G NR standards.**

A small local model (`qwen2.5:3b` on Ollama) answers 5G NR standards questions from the public
TeleQnA benchmark in three setups: closed book, retrieval over the Release 17 NR specifications,
and retrieval plus an acronym glossary built from the 3GPP vocabulary (TR 21.905). Work in
progress; results and the full write-up follow.

See [CLAUDE.md](CLAUDE.md) for the project brief.
