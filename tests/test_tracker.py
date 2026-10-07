"""Check per-condition extraction paths and saved JSON."""
from __future__ import annotations

import json

from meta_data_extraction.components.tracker import LocalTracker
from meta_data_extraction.data_models import ExtractionResult, Provenance


def make_result(
    document_id: str = "2405.14324",
    config_name: str = "photon_and_neutron",
) -> ExtractionResult:
    return ExtractionResult(
        document_id=document_id,
        domain_name="generic",
        provenance=Provenance(config_name=config_name),
    )


def test_result_path_and_saved_json(tmp_path):
    tracker = LocalTracker(output_dir=str(tmp_path))
    tracker.log_run(make_result())
    target = tmp_path / "photon_and_neutron" / "2405.14324.json"
    assert tracker.result_path("2405.14324", "photon_and_neutron") == target
    content = json.loads(target.read_text(encoding="utf-8"))
    assert content["document_id"] == "2405.14324"
    assert content["provenance"]["config_name"] == "photon_and_neutron"


def test_special_characters_in_config_name_are_sanitised(tmp_path):
    tracker = LocalTracker(output_dir=str(tmp_path))
    tracker.log_run(make_result(config_name="experiments/pan-v1"))
    target = tmp_path / "experiments_pan-v1" / "2405.14324.json"
    assert list(tmp_path.iterdir()) == [target.parent]
    assert target.is_file()
    assert tracker.result_path("2405.14324", "experiments/pan-v1") == target


def test_different_configs_produce_separate_folders(tmp_path):
    tracker = LocalTracker(output_dir=str(tmp_path))
    tracker.log_run(make_result(config_name="default"))
    tracker.log_run(make_result(config_name="photon_and_neutron"))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["default", "photon_and_neutron"]
