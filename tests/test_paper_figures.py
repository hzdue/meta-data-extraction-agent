"""Check that the paper figure renders the prepared JSON results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_paper_figures as figures  # noqa: E402


@pytest.fixture
def paper_results(tmp_path, monkeypatch):
    few_shot = {
        "models": ["GPT-5.5", "Qwen3.5 27B", "Qwen3.5 9B", "Qwen3.5 4B"],
        "shots": [0, 1, 3],
        "outcomes": [
            {
                "outcome": name,
                "deltas": [[0, 0.1 * scale, 0.05 * scale],
                           [0, -0.1 * scale, -0.05 * scale],
                           [0, 0, 0], [0, 0.05 * scale, -0.1 * scale]],
            }
            for name, scale in zip(
                ("Affiliations", "Authors", "Funding", "Facilities", "Samples", "Datasets"),
                (0, 0, 0.25, 0.5, 0.75, 1), strict=True,
            )
        ],
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"n_papers": 30, "few_shot": few_shot}), encoding="utf-8")
    monkeypatch.setattr(figures, "RESULTS_PATH", path)
    return few_shot


def test_few_shot_figure_writes_a_compact_manuscript_pdf(tmp_path, monkeypatch, paper_results) -> None:
    saved = []
    original_savefig = Figure.savefig

    def record_savefig(fig, *args, **kwargs):
        saved.append(fig)
        return original_savefig(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", record_savefig)
    output_dir = tmp_path / "figures"
    path = figures.build_few_shot(output_dir=output_dir)

    assert path == output_dir / "few_shot.pdf"
    assert list(output_dir.iterdir()) == [path]
    assert path.read_bytes().startswith(b"%PDF-")
    figure, = saved
    assert tuple(figure.get_size_inches()) == pytest.approx((16 / 2.54, 2))
    assert tuple(ax.get_title() for ax in figure.axes) == (
        "Funding", "Facilities", "Samples", "Datasets"
    )
    assert [text.get_text() for text in figure.legends[0].get_texts()] == paper_results["models"]
    assert [ax.get_ylabel() for ax in figure.axes] == [r"$\Delta$ entity F1", "", "", ""]
    for column, ax in enumerate(figure.axes):
        subplot = ax.get_subplotspec()
        assert subplot.get_gridspec().nrows == 1
        assert subplot.get_gridspec().ncols == 4
        assert list(subplot.colspan) == [column]
        assert ax.get_ylim() == pytest.approx((-0.18, 0.18))
        assert tuple(label.get_text() for label in ax.get_xticklabels()) == (
            "0-shot", "1-shot", "3-shot"
        )

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    artists = [figure.legends[0], figure.axes[0].yaxis.label]
    for ax in figure.axes:
        artists.extend([ax.title, *ax.get_xticklabels(), *ax.get_yticklabels()])
    for artist in artists:
        bounds = artist.get_window_extent(renderer)
        assert bounds.x0 >= figure.bbox.x0 - 1
        assert bounds.y0 >= figure.bbox.y0 - 1
        assert bounds.x1 <= figure.bbox.x1 + 1
        assert bounds.y1 <= figure.bbox.y1 + 1


def test_few_shot_figure_plots_the_prepared_deltas(tmp_path, monkeypatch, paper_results) -> None:
    plotted = []
    original_plot = Axes.plot

    def record_plot(ax, x, y, **kwargs):
        plotted.append((tuple(x), tuple(y), kwargs["label"]))
        return original_plot(ax, x, y, **kwargs)

    monkeypatch.setattr(Axes, "plot", record_plot)
    figures.build_few_shot(output_dir=tmp_path / "figures")

    assert len(plotted) == 16
    panels = ("Funding", "Facilities", "Samples", "Datasets")
    outcomes = {row["outcome"]: row for row in paper_results["outcomes"]}
    for index, (x, y, label) in enumerate(plotted):
        assert x == (0, 1, 2)
        expected = outcomes[panels[index // 4]]["deltas"][index % 4]
        assert y == pytest.approx(expected)
        assert label == paper_results["models"][index % 4]


def test_missing_results_reports_the_missing_path(tmp_path, monkeypatch) -> None:
    path = tmp_path / "missing.json"
    monkeypatch.setattr(figures, "RESULTS_PATH", path)
    with pytest.raises(FileNotFoundError, match="Missing paper results") as error:
        figures.build_few_shot(output_dir=tmp_path / "figures")
    assert str(path) in str(error.value)
    assert not (tmp_path / "figures").exists()
