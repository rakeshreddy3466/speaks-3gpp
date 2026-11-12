"""Does the model speak 3GPP? Retrieval and an acronym glossary for a local LLM on NR standards."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CONFIG = ROOT / "configs" / "experiment.yaml"
