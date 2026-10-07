"""Build the paper's few-shot figure from outputs/paper/results.json."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "paper" / "figures"
RESULTS_PATH = ROOT / "outputs" / "paper" / "results.json"
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral"],
        "font.size": 9,
        "mathtext.fontset": "stix",
        "text.color": "#111111",
        "axes.labelcolor": "#111111",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "pdf.fonttype": 42,
    }
)


def build_few_shot(output_dir: Path = FIGURE_DIR) -> Path:
    """Write Figure ``fig:few-shot`` as a compact four-panel ``few_shot.pdf``.

    Plot changes in entity F1 from each model's zero-shot baseline on the 30
    evaluation papers. Authors and affiliations score 1.000 throughout and
    are omitted. Each remaining entity collection has one panel.
    """

    if not RESULTS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing paper results: {RESULTS_PATH}. "
        )
    few_shot = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))["few_shot"]
    outcomes = {row["outcome"]: row for row in few_shot["outcomes"]}
    panels = ("Funding", "Facilities", "Samples", "Datasets")
    model_labels = few_shot["models"]
    model_colors = ("#c65a11", "#08519c", "#4292c6", "#78afd0")
    model_widths = (1.5, 1.2, 1.2, 1.2)
    model_markers = ("o", "s", "^", "D")
    model_styles = ("-", "--", "--", "--")
    x = (0, 1, 2)
    xticklabels = tuple(f"{shots}-shot" for shots in few_shot["shots"])

    fig, axes = plt.subplots(1, 4, figsize=(16 / 2.54, 2.0), sharey=True)
    for ax, name in zip(axes, panels, strict=True):
        outcome = outcomes[name]
        ax.axhline(0.0, color="#777777", linewidth=0.8, zorder=1)
        for deltas, color, width, marker, style, label in zip(
            outcome["deltas"],
            model_colors,
            model_widths,
            model_markers,
            model_styles,
            model_labels,
            strict=True,
        ):
            ax.plot(
                x,
                deltas,
                marker=marker,
                markersize=3.8,
                markeredgecolor="white",
                markeredgewidth=0.45,
                linewidth=width,
                linestyle=style,
                color=color,
                label=label,
                zorder=3,
            )
        ax.set_title(name, fontsize=10, pad=5)
        ax.set_ylim(-0.18, 0.18)
        ax.set_yticks([-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15])
        ax.set_xlim(-0.25, 2.25)
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)
        ax.grid(True, axis="y", color="#e4e4e4", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9, length=2.5, width=0.6, pad=2)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#aaaaaa")
            ax.spines[spine].set_linewidth(0.6)

    axes[0].set_ylabel(r"$\Delta$ entity F1", fontsize=9, labelpad=4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.52, 0.995),
        handlelength=2,
        handletextpad=0.4,
        columnspacing=1.4,
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.105, right=0.99, bottom=0.14, top=0.79, wspace=0.18)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "few_shot.pdf"
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)
    print(f"wrote {path}")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the manuscript's few-shot figure."""

    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    build_few_shot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
