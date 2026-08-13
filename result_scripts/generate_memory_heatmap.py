#############################################################################
# File: generate_memory_heatmap.py
#
# Description:
#   Builds the paper heatmap from the consolidated evaluation metrics text file.
#
#   - Parses LongMemEval question-type accuracy and benchmark-level accuracy.
#   - Uses one shared 0-100 percent scale across architectures and tasks.
#   - Annotates each cell with accuracy and its raw denominator.
#   - Saves publication PNG and SVG figures to the configured figures directory.
#############################################################################

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INPUT = Path("../asdrp/results/final_eval_metrics.txt")
OUT_PNG = Path("../figures/memory_architecture_accuracy_heatmap.png")
OUT_SVG = Path("../figures/memory_architecture_accuracy_heatmap.svg")

SYSTEMS = ["Condensed", "Propositional", "Episodic", "Graph", "Vector", "HVM"]

LME_METRICS = [
    ("Single Session User", "Single-Session\nUser"),
    ("Single Session Preference", "Single-Session\nPreference"),
    ("Single Session Assistant", "Single-Session\nAssistant"),
    ("Multi Session", "Multi-\nSession"),
    ("Temporal Reasoning", "Temporal\nReasoning"),
    ("Knowledge Update", "Knowledge\nUpdate"),
    ("Abstention", "Abstention"),
]

OTHER_METRICS = [
    ("Ecommerce", "Overall"),
    ("MAB - Accurate Retrieval", "Accurate\nRetrieval"),
    ("MAB - Long-Range Understanding", "Long-Range\nUnderstanding"),
    ("MAB - Test-Time Learning", "Test-Time\nLearning"),
    ("MAB - Conflict Resolution", "Conflict\nResolution"),
]


def _section(lines: list[str], system: str, dataset: str) -> list[str]:
    header = f"{system.upper() if system != 'HVM' else 'HVM'} | {dataset}"
    start = next(i for i, line in enumerate(lines) if line.strip() == header)
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("#" * 20)),
        len(lines),
    )
    return lines[start:end]


def _parse_lme(lines: list[str], system: str) -> dict[str, tuple[int, int, float]]:
    """Return question type -> (correct, denominator, accuracy%)."""
    sec = _section(lines, system, "LONGMEMEVAL")

    start = next(
        i
        for i, line in enumerate(sec)
        if line.strip() == "PERFORMANCE BY QUESTION TYPE"
    )
    end = next(
        i
        for i in range(start + 1, len(sec))
        if sec[i].strip() == "PERFORMANCE BY SOURCE"
    )

    out: dict[str, tuple[int, int, float]] = {}
    row_re = re.compile(r"^(.*?)\s{2,}(\d+)\s+(\d+)\s+([\d.]+)%\s*$")

    for line in sec[start + 1 : end]:
        m = row_re.match(line)
        if m and m.group(1).strip() != "Question Type":
            out[m.group(1).strip()] = (
                int(m.group(2)),
                int(m.group(3)),
                float(m.group(4)),
            )

    # Some sections list Abstention separately instead of in the question-type table.
    if "Abstention" not in out:
        try:
            a = next(i for i, line in enumerate(sec) if line.strip() == "ABSTENTION")
            abst_re = re.compile(r"^\s*(\d+)\s+(\d+)\s+([\d.]+)%\s*$")
            for line in sec[a + 1 : a + 8]:
                m = abst_re.match(line)
                if m:
                    out["Abstention"] = (
                        int(m.group(1)),
                        int(m.group(2)),
                        float(m.group(3)),
                    )
                    break
        except StopIteration:
            pass

    return out


def _parse_overview(lines: list[str]) -> dict[tuple[str, str], tuple[int, int, float]]:
    start = next(
        i for i, line in enumerate(lines) if line.strip() == "RESULTS OVERVIEW"
    )
    end = next(
        i
        for i in range(start + 1, len(lines))
        if lines[i].strip() == "EVALUATION POLICIES"
    )

    out: dict[tuple[str, str], tuple[int, int, float]] = {}

    for line in lines[start:end]:
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 7 or parts[2] != "OK":
            continue

        system, dataset = parts[0], parts[1]
        if system not in SYSTEMS:
            continue

        correct = int(parts[3].replace(",", ""))
        denom = int(parts[4].replace(",", ""))
        acc = float(parts[5].rstrip("%"))
        out[(system, dataset)] = (correct, denom, acc)

    return out


def main() -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    lines = INPUT.read_text(encoding="utf-8").splitlines()
    overview = _parse_overview(lines)

    columns = [display for _, display in LME_METRICS] + [
        display for _, display in OTHER_METRICS
    ]

    values = np.zeros((len(SYSTEMS), len(columns)), dtype=float)
    denoms = np.zeros((len(SYSTEMS), len(columns)), dtype=int)

    for r, system in enumerate(SYSTEMS):
        lme = _parse_lme(lines, system)

        for c, (metric, _) in enumerate(LME_METRICS):
            _, n, acc = lme[metric]
            values[r, c] = acc
            denoms[r, c] = n

        base = len(LME_METRICS)
        for j, (dataset, _) in enumerate(OTHER_METRICS):
            _, n, acc = overview[(system, dataset)]
            values[r, base + j] = acc
            denoms[r, base + j] = n

    # Sized for paper use but still readable when reduced.
    fig, ax = plt.subplots(figsize=(7.35, 4.95), dpi=360)
    im = ax.imshow(values, cmap="Blues", vmin=0, vmax=100, aspect="auto")

    # Main x-axis labels
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(
        columns,
        fontsize=6.9,
        linespacing=1.0,
        rotation=32,
        ha="right",
        rotation_mode="anchor",
    )

    # Y-axis labels
    ax.set_yticks(np.arange(len(SYSTEMS)))
    ax.set_yticklabels(SYSTEMS, fontsize=8.5, fontweight="semibold")

    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", length=0, pad=7)

    # Cell borders
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(SYSTEMS), 1), minor=True)
    ax.grid(which="minor", linewidth=0.9, color="white")
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_edgecolor("#57616a")

    # Stronger benchmark-family dividers INSIDE the heatmap only
    for x in (6.5, 7.5):
        ax.axvline(x, linewidth=1.8, color="#8797aa", zorder=4)

    # Cell annotations
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            v = values[r, c]
            n = denoms[r, c]
            color = "white" if v >= 57 else "#08306b"

            ax.text(
                c,
                r - 0.09,
                f"{v:.1f}%",
                ha="center",
                va="center",
                fontsize=7.0,
                fontweight="semibold",
                color=color,
            )
            ax.text(
                c,
                r + 0.19,
                f"(n={n:,})",
                ha="center",
                va="center",
                fontsize=6.0,
                color=color,
            )

    # -------- Group labels BELOW the x-axis --------
    group_y_rule = -0.25
    group_y_text = -0.35

    # True left/right edges of the heatmap.
    left_edge = -0.5
    right_edge = len(columns) - 0.5

    # Gap between each vertical tick and its neighboring horizontal rule.
    boundary_gap = 0.10

    group_specs = [
        (
            left_edge + boundary_gap,
            6.5 - boundary_gap,
            "LongMemEval",
        ),
        (
            6.5 + boundary_gap,
            7.5 - boundary_gap,
            "Ecommerce",
        ),
        (
            7.5 + boundary_gap,
            right_edge - boundary_gap,
            "MemoryAgentBench",
        ),
    ]

    for x0, x1, label in group_specs:
        ax.plot(
            [x0, x1],
            [group_y_rule, group_y_rule],
            transform=ax.get_xaxis_transform(),
            color="#9aaabd",
            linewidth=0.9,
            clip_on=False,
        )
        ax.text(
            (x0 + x1) / 2,
            group_y_text,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            color="#173f73",
            clip_on=False,
        )

    # Short upward ticks at BOTH outer edges and the two internal boundaries.
    # Outer ticks align exactly with the left/right edges of the heatmap.
    tick_len = 0.028

    for x in (left_edge, 6.5, 7.5, right_edge):
        ax.plot(
            [x, x],
            [group_y_rule, group_y_rule + tick_len],
            transform=ax.get_xaxis_transform(),
            color="#9aaabd",
            linewidth=0.9,
            clip_on=False,
        )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.018)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.set_ticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    cbar.ax.tick_params(labelsize=7.2, width=0.8, length=3)
    cbar.set_label(
        "Accuracy",
        rotation=270,
        labelpad=12,
        fontsize=8.2,
        fontweight="semibold",
    )
    cbar.outline.set_linewidth(0.8)

    # Extra room for rotated x labels + lower group labels.
    fig.subplots_adjust(left=0.125, right=0.94, bottom=0.42, top=0.97)

    fig.savefig(OUT_PNG, dpi=360, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_SVG, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_SVG}")


if __name__ == "__main__":
    main()
