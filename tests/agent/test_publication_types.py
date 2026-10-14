"""Offline publication run-scope validation; constructing scope dispatches nothing."""
import json

import pytest

from meta_data_extraction.agent.publication_types import (
    PublicationLiveSettings, PublicationSpec, load_publication_spec,
)


def payload(**updates):
    data = dict(run_id="publication-unit", mode="mock", split="synthetic",
        source={"path": "tests/agent/fixtures/invented.md", "sha256": "1" * 64},
        mock_responses=[{"path": "tests/agent/fixtures/invented-output.json", "sha256": "2" * 64}])
    data.update(updates)
    return data


def live_payload(**updates):
    return payload(mode="live", split="development", mock_responses=[],
        live={"endpoint": "https://chat.example.invalid/v1"}, **updates)


def test_mock_defaults_are_bounded_and_automatic_edits_disabled():
    scope = PublicationSpec.model_validate(payload())
    assert scope.mode == "mock" and scope.split == "synthetic"
    assert scope.max_calls == 5 and scope.max_targets == 12
    assert not scope.allow_automatic_patches and scope.generate_proposal
    assert scope.live is None and scope.raw_retention == "hash_only"
    assert scope.policy == "issue_aware"
    assert PublicationSpec.model_validate(scope.model_dump()) == scope


def test_live_scope_does_not_read_credentials_or_create_requests(monkeypatch):
    monkeypatch.delenv("CHATAI_API_KEY", raising=False)
    scope = PublicationSpec.model_validate(live_payload())
    assert scope.mode == "live" and scope.live.model == "qwen3.8-27b"
    assert scope.live.api_key_env == "CHATAI_API_KEY"
    assert not scope.mock_responses
    assert "api_key" not in scope.live.model_dump()
    assert scope.source.path == "tests/agent/fixtures/invented.md"


@pytest.mark.parametrize("patch", [
    {"mode": "mock", "split": "development"},
    {"mode": "mock", "split": "calibration"},
    {"mock_responses": []},
    {"live": {"endpoint": "https://chat.example.invalid/v1"}},
    {"mode": "live", "mock_responses": [], "live": None},
    {"mode": "live", "live": {"endpoint": "https://chat.example.invalid/v1"}},
    {"mode": "local"}, {"split": "final_test"}, {"split": "test"},
], ids=[f"scope-{i}" for i in range(9)])
def test_provider_and_split_boundaries(patch):
    with pytest.raises(ValueError):
        PublicationSpec.model_validate(payload(**patch))


@pytest.mark.parametrize("name", ["allow_automatic_patches", "generate_proposal"])
@pytest.mark.parametrize("value", ["true", "false", 1, 0, None, [], {}], ids=["text-true", "text-false", "one", "zero", "none", "list", "dict"])
def test_permission_flags_are_explicit_booleans(name, value):
    with pytest.raises(ValueError):
        PublicationSpec.model_validate(payload(**{name: value}))


def test_automatic_edits_require_proposal_and_five_slots():
    valid = PublicationSpec.model_validate(payload(allow_automatic_patches=True))
    assert valid.max_calls == 5
    for patch in ({"generate_proposal": False}, {"max_calls": 4}):
        with pytest.raises(ValueError):
            PublicationSpec.model_validate(payload(allow_automatic_patches=True, **patch))
    assert PublicationSpec.model_validate(payload(max_calls=2, generate_proposal=False)).max_calls == 2


@pytest.mark.parametrize("name,value", [
    ("max_calls", 1), ("max_calls", 11), ("max_calls", True), ("max_calls", "5"), ("max_calls", 5.0),
    ("max_targets", 0), ("max_targets", 13), ("max_targets", False), ("max_targets", "12"),
    ("evidence_chars", 999), ("evidence_chars", 40001), ("evidence_chars", True), ("evidence_chars", "1000"),
    ("workflow_timeout_seconds", 0), ("workflow_timeout_seconds", 301),
    ("workflow_timeout_seconds", float("inf")), ("workflow_timeout_seconds", float("nan")),
], ids=[f"budget-{i}" for i in range(17)])
def test_run_budgets_reject_outside_bounds_and_integer_coercion(name, value):
    with pytest.raises(ValueError):
        PublicationSpec.model_validate(payload(**{name: value}))


@pytest.mark.parametrize("run_id", [
    "", "../outside", "..", "outputs/result", "D:/outside", "UPPERCASE", "with spaces", "x" * 33, "_control", "a.b",
], ids=[f"run-id-{i}" for i in range(10)])
def test_run_ids_cannot_be_output_paths(run_id):
    with pytest.raises(ValueError):
        PublicationSpec.model_validate(payload(run_id=run_id))


def test_maximum_safe_budgets_and_run_id():
    scope = PublicationSpec.model_validate(payload(run_id="a" * 32, max_calls=10, max_targets=12,
        evidence_chars=40000, workflow_timeout_seconds=300))
    assert scope.max_calls == 10


@pytest.mark.parametrize("value", [True, False, "60", "1.5", None, [], {}, float("inf"), float("nan")],
    ids=["true", "false", "int-text", "float-text", "none", "list", "dict", "infinity", "nan"])
def test_timeouts_require_finite_explicit_numbers(value):
    with pytest.raises(ValueError):
        PublicationSpec.model_validate(payload(workflow_timeout_seconds=value))
    with pytest.raises(ValueError):
        PublicationLiveSettings(endpoint="https://chat.example.invalid/v1", timeout_seconds=value)


@pytest.mark.parametrize("value", [1, 1.5, 300])
def test_timeouts_accept_numeric_durations(value):
    assert PublicationSpec.model_validate(payload(workflow_timeout_seconds=value)).workflow_timeout_seconds == value
    assert PublicationLiveSettings(endpoint="https://chat.example.invalid/v1", timeout_seconds=value).timeout_seconds == value


@pytest.mark.parametrize("schemas", [[], ["generic.json", "generic.json"], ["other.json"], ["../generic.json"]])
def test_canonical_schemas_explicit_distinct_allowlist(schemas):
    with pytest.raises(ValueError):
        PublicationSpec.model_validate(payload(schemas=schemas))


@pytest.mark.parametrize("name,value", [
    ("policy", "generic_llm"), ("raw_retention", "discard"), ("unknown_usage", "ignore"),
    ("version", "publication-run-99"), ("output_dir", "outputs"), ("api_key", "not-a-real-key"),
], ids=[f"setting-{i}" for i in range(6)])
def test_unknown_runtime_and_output_settings_rejected(name, value):
    with pytest.raises(ValueError):
        PublicationSpec.model_validate(payload(**{name: value}))


@pytest.mark.parametrize("pin", [
    {"path": "x", "sha256": "short"}, {"path": "x", "sha256": "A" * 64},
    {"path": "x", "sha256": "g" * 64}, {"path": "", "sha256": "0" * 64},
    {"path": "x" * 501, "sha256": "0" * 64}, {"path": "x", "sha256": "0" * 64, "approved": True},
    {"path": "x"},
], ids=[f"pin-{i}" for i in range(7)])
def test_source_and_response_pins_have_strict_shape(pin):
    with pytest.raises(ValueError):
        PublicationSpec.model_validate(payload(source=pin))
    with pytest.raises(ValueError):
        PublicationSpec.model_validate(payload(mock_responses=[pin]))


def test_mock_response_catalog_is_bounded():
    one = payload()["mock_responses"][0]
    with pytest.raises(ValueError):
        PublicationSpec.model_validate(payload(mock_responses=[one] * 11))


@pytest.mark.parametrize("endpoint", [
    "http://chat.example.invalid", "https://user:secret@chat.example.invalid/v1",
    "https://chat.example.invalid/v1?key=secret", "https://chat.example.invalid/v1#secret",
    " https://chat.example.invalid", "https://", "file:///tmp/service", "",
], ids=[f"endpoint-{i}" for i in range(8)])
def test_live_endpoint_uses_existing_https_credential_free_boundary(endpoint):
    with pytest.raises(ValueError):
        PublicationLiveSettings(endpoint=endpoint)


@pytest.mark.parametrize("patch", [
    {"model": "other-model"}, {"api_key_env": "lowercase"}, {"api_key_env": "KEY=value"},
    {"api_key_env": "KEY WITH SPACES"}, {"api_key_env": "A" * 101}, {"api_key": "secret"},
    {"max_output_tokens": 0}, {"max_output_tokens": 32769}, {"max_output_tokens": True},
    {"max_prompt_bytes": 0}, {"max_prompt_bytes": 100001}, {"max_prompt_bytes": "1000"},
    {"timeout_seconds": 0}, {"timeout_seconds": 301}, {"timeout_seconds": float("inf")},
], ids=[f"live-setting-{i}" for i in range(15)])
def test_live_model_environment_and_budget_boundary(patch):
    with pytest.raises(ValueError):
        PublicationLiveSettings(endpoint="https://chat.example.invalid/v1", **patch)


def test_loader_accepts_only_local_scoped_config_and_does_not_read_source(workspace):
    location = workspace / "configs/agent/publication.yaml"
    location.write_text(json.dumps(payload()), encoding="utf-8")
    scope = load_publication_spec(workspace, "configs/agent/publication.yaml")
    assert scope == PublicationSpec.model_validate(payload())
    assert not (workspace / scope.source.path).exists()
    # Source bytes and their path policy are checked by runner preparation, not
    # by this pure typed config layer. A missing source cannot dispatch a model.


def test_loader_accepts_existing_run_config_copy(workspace):
    location = workspace / "runs/agent/unit/config.yaml"
    location.parent.mkdir(parents=True)
    location.write_text(json.dumps(payload()), encoding="utf-8")
    assert load_publication_spec(workspace, "runs/agent/unit/config.yaml").run_id == "publication-unit"


@pytest.mark.parametrize("relative", [
    "../config.yaml", "data/gold/config.yaml", "outputs/config.yaml", "paper/config.yaml",
    "schemas/config.yaml", "configs/alignment_helper.yaml", "D:/outside/config.yaml",
])
def test_loader_refuses_unapproved_roots(workspace, relative):
    with pytest.raises((ValueError, OSError)):
        load_publication_spec(workspace, relative)


@pytest.mark.parametrize("text", [
    "run_id: one\nrun_id: two\n", "source:\n  path: one\n  path: two\n",
    "[]", "null", "- one\n- two\n", "!!python/object/apply:os.system ['must-not-execute']",
], ids=["duplicate-root", "duplicate-nested", "list", "null", "sequence", "unsafe-yaml-tag"])
def test_loader_rejects_duplicate_keys_and_nonmapping_or_unsafe_yaml(workspace, text):
    location = workspace / "configs/agent/publication.yaml"
    location.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_publication_spec(workspace, "configs/agent/publication.yaml")


def test_loader_rejects_oversized_spec_before_parsing(workspace):
    location = workspace / "configs/agent/publication.yaml"
    location.write_bytes(b" " * 1000001)
    with pytest.raises(ValueError, match="size bound"):
        load_publication_spec(workspace, "configs/agent/publication.yaml")
