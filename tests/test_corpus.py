"""The release must retain the full gold corpus and only held-out observations."""

import json
from pathlib import Path

import pytest

from scripts.corpus import annotation_batches, load_corpus, paper_ids


ROOT = Path(__file__).resolve().parents[1]


def test_release_membership_and_exemplar_separation():
    calibration = set(paper_ids("calibration"))
    evaluation = set(paper_ids("evaluation"))
    assert len(calibration) == 10 and len(evaluation) == 30
    assert calibration.isdisjoint(evaluation)
    assert {p.stem for p in (ROOT / "data/gold").glob("*.json")} == calibration | evaluation

    expected_annotations = {
        (annotator, paper)
        for batch in annotation_batches().values()
        for annotator in batch["annotators"]
        for paper in batch["papers"]
    }
    assert len(expected_annotations) == 60
    assert {(p.parent.name, p.stem) for p in (ROOT / "data/annotations").glob("*/*.json")} == expected_annotations
    conditions = [p for p in (ROOT / "data/experiment").iterdir() if p.is_dir()]
    assert len(conditions) == 22
    for condition in conditions:
        assert {p.stem for p in condition.glob("*.json")} == evaluation
    examples = {p.stem for p in (ROOT / "data/few_shot").glob("*.json")}
    assert len(examples) == 3
    assert examples <= calibration


@pytest.mark.parametrize("invalid", ["duplicate", "split", "missing_pair", "calibration_batch", "self_pair"])
def test_invalid_split_or_pairing_is_rejected(tmp_path, invalid):
    corpus = load_corpus()
    evaluation = next(p for p in corpus["papers"] if p["split"] == "evaluation")
    if invalid == "duplicate":
        corpus["papers"].append(dict(evaluation))
    elif invalid == "split":
        evaluation["split"] = "unknown"
    elif invalid == "missing_pair":
        evaluation.pop("annotation_batch")
    elif invalid == "calibration_batch":
        evaluation["split"] = "calibration"
    else:
        corpus["annotation_pairs"][evaluation["annotation_batch"]] = ["annotator_1"] * 2
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(corpus), encoding="utf-8")
    with pytest.raises(ValueError):
        load_corpus(path)


def test_unknown_split_cannot_silently_produce_an_empty_evaluation():
    with pytest.raises(ValueError, match="Unknown corpus split"):
        paper_ids("test")
