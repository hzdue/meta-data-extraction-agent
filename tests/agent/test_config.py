import os

import pytest
from pydantic import ValidationError

from meta_data_extraction.agent.config import OfflineRunConfig, load_offline_config
from meta_data_extraction.agent.paths import PathPolicyError


@pytest.mark.parametrize("source", [
    "data/gold/example.json", "data/annotations/example.json", "data/experiment/example.json",
    "data/few_shot/example.json", "data/corpus.json", "configs/alignment_helper.yaml",
    "configs/evaluation_metrics.yaml", "outputs/paper/results.json", "outputs/evaluation/report.json",
    "paper/main.tex", ".", "src/meta_data_extraction/evaluation.py", "data/raw/../../outputs/a.json",
    "data/raw/*.pdf", "data/raw", "tests/agent/fixtures", "/outside/paper.pdf",
    "C:\\outside\\paper.pdf", "data/raw/missing.pdf",
])
def test_runtime_sources_reject_offline_assets_and_broad_selection(workspace, source):
    config = OfflineRunConfig(run_id="safe", sources=[source])
    with pytest.raises(PathPolicyError):
        config.source_files(workspace)


@pytest.mark.parametrize("extra", [
    {"mode": "live"}, {"output_dir": "outputs"}, {"tracker": {"output_dir": "data/experiment"}},
    {"llm": {"base_url": "https://invalid.example"}}, {"gold_path": "data/gold"},
    {"alignment_helper": "configs/alignment_helper.yaml"}, {"standards": ["data/gold/example.json"]},
    {"allow_release_write": True}, {"schemas": ["../configs/alignment_helper.yaml"]},
    {"sources": []}, {"schemas": []}, {"sources": ["data/raw/a", "data/raw/a"]},
])
def test_config_cannot_enable_inference_release_writes_or_gold(workspace, extra):
    with pytest.raises(ValidationError):
        OfflineRunConfig.model_validate({"run_id": "safe", "sources": ["data/raw/a"], **extra})


@pytest.mark.parametrize("config_path", [
    "configs/alignment_helper.yaml", "configs/evaluation_metrics.yaml", "data/gold/example.json",
    "outputs/paper/results.json", "configs/pan_gpt4o_full_0shot.yaml",
])
def test_config_loader_refuses_offline_files_before_open(workspace, config_path):
    with pytest.raises(PathPolicyError):
        load_offline_config(config_path, workspace)


def test_duplicate_yaml_keys_rejected(workspace):
    path = workspace / "configs/agent/duplicate.yaml"
    path.write_text("run_id: first\nrun_id: second\n")
    with pytest.raises(ValueError, match="Duplicate YAML key"):
        load_offline_config("configs/agent/duplicate.yaml", workspace)


def test_only_explicit_source_is_selected(workspace):
    source = workspace / "tests/agent/fixtures/synthetic_source.md"
    (source.parent / "unselected.md").write_text("must not be selected")
    config = load_offline_config("configs/agent/offline.yaml", workspace)
    assert config.source_files(workspace) == (source,)


def test_source_link_cannot_disguise_offline_data(workspace, link_directory):
    release = workspace / "data/gold"
    release.mkdir(parents=True)
    (release / "synthetic.json").write_text("synthetic private answer")
    alias = workspace / "data/raw"
    link_directory(alias, release)
    try:
        config = OfflineRunConfig(run_id="safe", sources=["data/raw/synthetic.json"])
        with pytest.raises(PathPolicyError):
            config.source_files(workspace)
    finally:
        alias.rmdir() if os.name == "nt" else alias.unlink()


def test_hard_link_cannot_disguise_offline_data(workspace):
    release = workspace / "synthetic-gold.json"
    release.write_text("synthetic private answer")
    source = workspace / "tests/agent/fixtures/alias.md"
    os.link(release, source)
    config = OfflineRunConfig(run_id="safe", sources=["tests/agent/fixtures/alias.md"])
    with pytest.raises(PathPolicyError):
        config.source_files(workspace)
