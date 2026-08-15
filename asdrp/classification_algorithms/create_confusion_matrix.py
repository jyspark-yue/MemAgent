#############################################################################
# File: create_confusion_matrix.py
#
# Description:
#   Builds the publication confusion-matrix figure from saved router metrics.
#
#   - Loads labels, counts, accuracy, and macro-F1 from metrics.json.
#   - Supports normalized percentages or raw confusion-matrix counts.
#   - Formats LongMemEval labels for a compact paper figure.
#   - Saves a high-resolution PNG with configurable text and colormap settings.
#############################################################################

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence


def normalize_matrix(
    matrix: Sequence[Sequence[int]],
    normalize: bool,
) -> list[list[float]]:
    values = [[float(value) for value in row] for row in matrix]

    if not normalize:
        return values

    normalized: list[list[float]] = []

    for row in values:
        row_total = sum(row)
        normalized.append([value / row_total if row_total else 0.0 for value in row])

    return normalized


def get_summary_metrics(
    metrics: dict,
    matrix: Sequence[Sequence[int]],
) -> tuple[float, float, int]:
    """Return accuracy, macro-F1, and N, using the confusion matrix as fallback."""
    n = int(sum(sum(int(value) for value in row) for row in matrix))

    accuracy_value = metrics.get("accuracy")
    if isinstance(accuracy_value, (int, float)):
        accuracy = float(accuracy_value)
    else:
        correct = sum(
            int(matrix[index][index])
            for index in range(min(len(matrix), len(matrix[0]) if matrix else 0))
        )
        accuracy = correct / n if n else 0.0

    f1_value = metrics.get("macro_f1", metrics.get("f1"))
    if isinstance(f1_value, (int, float)):
        macro_f1 = float(f1_value)
    else:
        class_f1_scores: list[float] = []

        for class_index in range(len(matrix)):
            true_positive = float(matrix[class_index][class_index])
            false_positive = sum(
                float(matrix[row_index][class_index])
                for row_index in range(len(matrix))
                if row_index != class_index
            )
            false_negative = sum(
                float(matrix[class_index][column_index])
                for column_index in range(len(matrix[class_index]))
                if column_index != class_index
            )

            denominator = (2.0 * true_positive) + false_positive + false_negative
            class_f1_scores.append(
                (2.0 * true_positive) / denominator if denominator else 0.0
            )

        macro_f1 = (
            sum(class_f1_scores) / len(class_f1_scores) if class_f1_scores else 0.0
        )

    return accuracy, macro_f1, n


LONGMEMEVAL_LABEL_NAMES = {
    "knowledge-update": "Knowledge\nUpdate",
    "multi-session": "Multi\nSession",
    "single-session-assistant": "Single-Session\nAssistant",
    "single-session-preference": "Single-Session\nPreference",
    "single-session-user": "Single-Session\nUser",
    "temporal-reasoning": "Temporal\nReasoning",
}


def format_label(label: object) -> str:
    """Convert machine-readable labels into publication-ready labels."""
    raw_label = str(label)
    return LONGMEMEVAL_LABEL_NAMES.get(
        raw_label,
        raw_label.replace("_", " ").replace("-", " ").title(),
    )


def fit_text_to_axis_width(
    figure,
    axis,
    text_artist,
    *,
    max_width_fraction: float = 0.97,
    min_font_size: float = 18.0,
) -> None:
    """Shrink text only if absolutely necessary to keep it inside matrix edges."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axis_width = axis.get_window_extent(renderer=renderer).width
    max_width = axis_width * max_width_fraction

    while (
        text_artist.get_window_extent(renderer=renderer).width > max_width
        and text_artist.get_fontsize() > min_font_size
    ):
        text_artist.set_fontsize(max(text_artist.get_fontsize() - 0.25, min_font_size))
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()


def format_significant(value: float, digits: int = 4) -> str:
    """Format a finite number with exactly ``digits`` significant figures."""
    if not math.isfinite(value):
        return str(value)
    if value == 0:
        return "0." + ("0" * (digits - 1))

    magnitude = math.floor(math.log10(abs(value)))
    decimal_places = max(digits - magnitude - 1, 0)
    return f"{value:.{decimal_places}f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a publication-ready PNG confusion matrix "
            "from a classifier metrics.json file."
        )
    )

    parser.add_argument(
        "results",
        type=Path,
        help="Results directory or path to metrics.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output PNG path. Defaults to confusion_matrix.png " "beside metrics.json."
        ),
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help=(
            "Disable row normalization and display raw counts. "
            "By default, each true-label row is normalized."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help=(
            "Optional figure title. For publication, it is often better "
            "to omit this and use the manuscript caption."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="PNG resolution. Defaults to 400 DPI.",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=11.0,
        help=(
            "Base font-size control. Defaults to 11. "
            "Publication figure text is scaled upward from this value."
        ),
    )
    parser.add_argument(
        "--cmap",
        default="Blues",
        help=(
            "Continuous Matplotlib colormap. "
            "Examples: Blues, cividis, viridis, Greys."
        ),
    )
    parser.add_argument(
        "--raw-labels",
        action="store_true",
        help="Use labels exactly as stored in metrics.json.",
    )
    parser.add_argument(
        "--hide-counts",
        action="store_true",
        help=(
            "When normalized, show only percentages rather than "
            "percentages with raw counts."
        ),
    )
    parser.add_argument(
        "--x-rotation",
        type=float,
        default=0.0,
        help="Rotation angle for predicted-class labels. Defaults to 0 degrees.",
    )

    args = parser.parse_args()
    normalize = not args.no_normalize

    metrics_path = (
        args.results / "metrics.json" if args.results.is_dir() else args.results
    )

    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    labels = metrics.get("labels")
    matrix = metrics.get("confusion_matrix")

    if not isinstance(labels, list) or not labels:
        raise ValueError(f"Missing non-empty 'labels' list in {metrics_path}")

    if not isinstance(matrix, list) or len(matrix) != len(labels):
        raise ValueError(f"Invalid 'confusion_matrix' in {metrics_path}")

    if any(not isinstance(row, list) or len(row) != len(labels) for row in matrix):
        raise ValueError("Confusion matrix dimensions do not match the labels")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to create the confusion matrix"
        ) from exc

    display_labels = (
        [str(label) for label in labels]
        if args.raw_labels
        else [format_label(label) for label in labels]
    )

    display_matrix = normalize_matrix(
        matrix=matrix,
        normalize=normalize,
    )

    accuracy, macro_f1, sample_count = get_summary_metrics(metrics, matrix)

    label_count = len(labels)

    # Keep the source canvas compact so the figure does not have to be scaled
    # down as aggressively when placed in a single IEEE column.
    figure_width = max(8.6, label_count * 1.60)
    figure_height = max(7.6, label_count * 1.48)

    # Enlarged source fonts plus a compact canvas are intentional: at typical
    # single-column placement (~3.5 in wide), these remain roughly 8–10 pt.
    descriptor_font_size = args.font_size + 7  # 18 pt by default
    title_font_size = args.font_size + 10  # 21 pt by default
    cell_font_size = args.font_size + 10  # 21 pt by default

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": descriptor_font_size,
            "axes.labelsize": descriptor_font_size + 2,
            "axes.titlesize": title_font_size,
            "xtick.labelsize": descriptor_font_size - 1,
            "ytick.labelsize": descriptor_font_size - 1,
        }
    )

    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height),
    )

    image = axis.imshow(
        display_matrix,
        cmap=args.cmap,
        interpolation="nearest",
        aspect="equal",
        vmin=0,
        vmax=1 if normalize else None,
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.082,
        shrink=0.88,
    )

    if normalize:
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        colorbar.set_label(
            "Share of True-Label Samples",
            rotation=270,
            labelpad=22,
            fontsize=descriptor_font_size,
        )
    else:
        colorbar.set_label(
            "Sample Count",
            rotation=270,
            labelpad=22,
            fontsize=descriptor_font_size,
        )

    colorbar.outline.set_linewidth(0.8)
    colorbar.ax.tick_params(
        width=0.8,
        length=4,
        labelsize=descriptor_font_size,
        pad=5,
    )

    axis.set_xticks(range(label_count))
    axis.set_yticks(range(label_count))
    axis.set_xticklabels(display_labels)
    axis.set_yticklabels(display_labels)

    axis_label_pad = 30
    axis.set_xlabel(
        "Predicted Question Type",
        labelpad=axis_label_pad,
        fontsize=descriptor_font_size + 2,
        fontweight="semibold",
        color="#202020",
    )
    axis.set_ylabel(
        "True Question Type",
        labelpad=axis_label_pad,
        fontsize=descriptor_font_size + 2,
        fontweight="semibold",
        color="#202020",
    )

    accuracy_text = format_significant(accuracy * 100.0, 4) + "%"
    macro_f1_text = format_significant(macro_f1, 4)
    summary_text = (
        f"Accuracy: {accuracy_text} | "
        f"Macro F1: {macro_f1_text} | "
        f"N: {sample_count:,}"
    )

    summary_artist = axis.text(
        0.5,
        1.020,
        summary_text,
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=descriptor_font_size - 1,
        fontweight="normal",
        color="#3f3f3f",
        clip_on=False,
    )

    if args.title:
        figure.suptitle(
            args.title,
            x=0.5,
            y=0.965,
            ha="center",
            va="top",
            fontsize=title_font_size,
            fontweight="semibold",
        )

    plt.setp(
        axis.get_xticklabels(),
        rotation=args.x_rotation,
        ha="center",
        rotation_mode="anchor",
    )

    for tick_label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
        tick_label.set_multialignment("center")
        tick_label.set_linespacing(1.10)
        tick_label.set_fontweight("normal")
        tick_label.set_color("#444444")

    axis.set_xticks(
        [index - 0.5 for index in range(1, label_count)],
        minor=True,
    )
    axis.set_yticks(
        [index - 0.5 for index in range(1, label_count)],
        minor=True,
    )

    axis.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=0.7,
        alpha=0.55,
    )

    axis.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    axis.tick_params(
        axis="both",
        which="major",
        length=4,
        width=0.9,
        pad=14,
        colors="#444444",
    )

    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.15)
        spine.set_color("#444444")

    colormap = image.get_cmap()
    image_norm = image.norm

    for row_index, row in enumerate(display_matrix):
        for column_index, value in enumerate(row):
            raw_count = int(matrix[row_index][column_index])

            if normalize:
                if args.hide_counts:
                    annotation = f"{value:.1%}"
                else:
                    annotation = f"{value:.1%}\n({raw_count:,})"
            else:
                annotation = f"{raw_count:,}"

            red, green, blue, _ = colormap(image_norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            text_color = "white" if luminance < 0.48 else "#163A63"

            axis.text(
                column_index,
                row_index,
                annotation,
                ha="center",
                va="center",
                fontsize=cell_font_size,
                fontweight="medium",
                color=text_color,
                linespacing=1.08,
            )

    # Deliberate publication spacing:
    # - top leaves a clear band between title and summary;
    # - bottom/left separate axis descriptors from category labels;
    # - right leaves breathing room around the colorbar.
    figure.subplots_adjust(
        left=0.235,
        bottom=0.235,
        right=0.825,
        top=0.790,
    )

    # Preserve the requested descriptor size whenever possible. It may shrink
    # only slightly if the summary would otherwise cross the matrix edges.
    fit_text_to_axis_width(
        figure,
        axis,
        summary_artist,
        max_width_fraction=0.95,
        min_font_size=max(descriptor_font_size - 1.0, 15.0),
    )

    output_path = args.output or metrics_path.with_name("confusion_matrix.png")

    if output_path.suffix.lower() != ".png":
        raise ValueError(f"Output must be a PNG file, not: {output_path.suffix}")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=args.dpi,
        bbox_inches="tight",
        facecolor="white",
        format="png",
        pad_inches=0.22,
    )

    plt.close(figure)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
