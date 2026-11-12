"""Every table and figure from the committed results/summary.json only (no data, no model)."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import FIGURES, RESULTS  # noqa: E402

LABEL = {
    "closed_book": "A closed book",
    "retrieval": "B retrieval",
    "retrieval_glossary": "C retrieval + glossary",
}


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write(df: pd.DataFrame, name: str) -> None:
    out = RESULTS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}.md").write_text(df.to_markdown(index=False) + "\n")


def tables(s: dict) -> None:
    rows = []
    for k, v in s["setups"].items():
        rows.append(
            {
                "setup": LABEL[k],
                "accuracy": pct(v["accuracy"]),
                "95% CI": f"{pct(v['ci95'][0])} to {pct(v['ci95'][1])}",
                "correct": f"{v['correct']}/{s['n_questions']}",
                "unparsed": v["unparsed"],
                f"acronym q (n={v['acronym_questions']['n']})": pct(
                    v["acronym_questions"]["accuracy"]
                ),
                f"other q (n={v['other_questions']['n']})": pct(v["other_questions"]["accuracy"]),
                "median prompt tokens": v["median_prompt_tokens"],
                "median s/question": v["median_seconds"],
            }
        )
    write(pd.DataFrame(rows), "T1_accuracy")
    write(pd.DataFrame(s["paired"]), "T2_paired_mcnemar")
    eg = [{"setup": LABEL[k], **v} for k, v in s["error_groups"].items()]
    write(pd.DataFrame(eg), "T3_error_groups")
    fl = [{"setup": LABEL[k], **v} for k, v in s["flips_vs_closed_book"].items()]
    write(pd.DataFrame(fl), "T4_flips_vs_closed_book")


def figures(s: dict) -> None:
    FIGURES.mkdir(exist_ok=True)
    names = list(s["setups"])
    acc = [100 * s["setups"][k]["accuracy"] for k in names]
    lo = [a - 100 * s["setups"][k]["ci95"][0] for a, k in zip(acc, names, strict=True)]
    hi = [100 * s["setups"][k]["ci95"][1] - a for a, k in zip(acc, names, strict=True)]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(
        [LABEL[k] for k in names],
        acc,
        yerr=[lo, hi],
        capsize=6,
        color=["#9aa5b1", "#3b7dd8", "#0e6e55"],
    )
    ax.axhline(100 * s["chance_accuracy"], ls="--", color="grey", lw=1)
    ax.text(2.45, 100 * s["chance_accuracy"] + 1, "chance", ha="right", color="grey", fontsize=8)
    for i, a in enumerate(acc):
        ax.text(i, a + hi[i] + 1.5, f"{a:.1f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("accuracy (%)")
    ax.set_title(
        f"{s['model']} on {s['n_questions']} NR standards questions (95% bootstrap CI)", fontsize=9
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "F1_accuracy.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    groups = ["missing_context", "context_present", "unparsed"]
    colors = ["#c0504d", "#e3a33b", "#9aa5b1"]
    keys = list(s["error_groups"])
    left = [0] * len(keys)
    for g, col in zip(groups, colors, strict=True):
        vals = [s["error_groups"][k][g] for k in keys]
        ax.barh([LABEL[k] for k in keys], vals, left=left, color=col, label=g.replace("_", " "))
        left = [a + b for a, b in zip(left, vals, strict=True)]
    ax.set_xlabel("wrong answers")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Why retrieval answers were wrong (answer text in context or not)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "F2_error_groups.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.0))
    keys = list(s["flips_vs_closed_book"])
    helped = [s["flips_vs_closed_book"][k]["helped"] for k in keys]
    hurt = [-s["flips_vs_closed_book"][k]["hurt"] for k in keys]
    ax.barh([LABEL[k] for k in keys], helped, color="#0e6e55", label="wrong to right")
    ax.barh([LABEL[k] for k in keys], hurt, color="#c0504d", label="right to wrong")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("questions that changed compared with closed book")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "F3_flips.png", dpi=160)
    plt.close(fig)


def main() -> None:
    s = json.loads((RESULTS / "summary.json").read_text())
    tables(s)
    figures(s)
    print("tables T1-T4 and figures F1-F3 written")


if __name__ == "__main__":
    main()
