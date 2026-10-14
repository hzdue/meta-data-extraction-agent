import hashlib
import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from meta_data_extraction.agent.paths import PathPolicyError
from meta_data_extraction.offline.manifest import (
    ArtifactRef, ManifestError, PredictionRef, Publication, SplitManifest,
    artifact_json, load_manifest, manifest_digest, read_artifact, strict_json,
)


def ref(path="tests/offline/fixtures/example.json", raw=b"{}"):
    return ArtifactRef(path=path, sha256=hashlib.sha256(raw).hexdigest())


def publication(identifier="paper-1", **updates):
    values = dict(publication_id=identifier, split="calibration", data_kind="synthetic",
                  alignment_status="synthetic", predictions=())
    return Publication(**(values | updates))


def manifest(*publications, **updates):
    values = dict(experiment_id="synthetic-comparison", description="Invented unit fixture",
                  primary_outcome="Separate inherited metrics with support",
                  variants=("baseline", "candidate"), publications=publications,
                  split_notes="Synthetic fixtures only; no natural papers inspected.",
                  held_out_policy="Future natural papers remain unselected; no final gold access.")
    return SplitManifest(**(values | updates))


def write_fixture(workspace: Path, name: str, raw: bytes) -> ArtifactRef:
    relative = "tests/offline/fixtures/" + name
    path = workspace / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return ref(relative, raw)


def test_explicit_manifest_parse_never_reads_declared_gold(workspace, monkeypatch):
    item = manifest(publication(gold=ref("data/gold/not-opened.json")))
    selected = write_fixture(workspace, "manifest.json", item.model_dump_json().encode())
    original = Path.read_bytes
    reads = []

    def tracked(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", tracked)
    loaded = load_manifest(workspace, selected.path)
    assert loaded == item
    assert reads == [workspace / selected.path]


def test_held_out_manifest_can_be_prepared_without_selecting_or_reading_papers():
    item = manifest()
    assert item.publications == ()
    assert len(manifest_digest(item)) == 64


@pytest.mark.parametrize("path", [
    "../private.json", "data/gold/../../secret", "/outside.json", "C:/secret.json",
    "tests/offline/fixtures/*.json", "tests/offline/fixtures/a:secret", "tests/offline/fixtures/NUL.json",
    "tests/offline/fixtures/a.", "tests/offline/fixtures/a ", "tests//fixtures/a.json",
])
def test_artifact_rejects_unsafe_portable_paths(path):
    with pytest.raises(ValidationError):
        ref(path)


def test_backslashes_canonicalize_to_one_path():
    assert ref(r"tests\offline\fixtures\example.json").path == "tests/offline/fixtures/example.json"


@pytest.mark.parametrize("hash_value", ["A" * 64, "a" * 63, "not-a-hash", "0" * 65])
def test_artifact_requires_exact_sha256(hash_value):
    with pytest.raises(ValidationError):
        ArtifactRef(path="tests/offline/fixtures/a", sha256=hash_value)


@pytest.mark.parametrize("value", [
    {"variant": "baseline", "status": "ready"},
    {"variant": "baseline", "status": "failed"},
    {"variant": "baseline", "status": "failed", "failure_code": "timeout", "artifact": ref()},
    {"variant": "baseline", "status": "ready", "artifact": ref(), "failure_code": "timeout"},
    {"variant": "baseline", "status": "unprocessed", "reference_report": ref()},
    {"variant": "baseline", "status": "unprocessed", "request_count": True},
    {"variant": "baseline", "status": "unprocessed", "latency_ms": float("inf")},
    {"variant": "baseline", "status": "unprocessed", "cost": -1.0},
    {"variant": "baseline", "status": "unprocessed", "request_count": "1"},
    {"variant": "baseline", "status": "unprocessed", "execution_mode": "cached"},
])
def test_prediction_rejects_ambiguous_or_invalid_states(value):
    with pytest.raises(ValidationError):
        PredictionRef(**value)


def test_unknown_resource_usage_is_preserved_not_zero():
    item = PredictionRef(variant="baseline", status="failed", failure_code="timeout")
    assert all(getattr(item, name) is None for name in
               ("request_count", "input_tokens", "output_tokens", "latency_ms", "cost"))


def test_publication_aliases_cannot_cross_splits():
    with pytest.raises(ValidationError, match="common lineage"):
        manifest(publication(aliases=("DOI:10.1234/AbC",)),
                 publication("paper-2", split="future_held_out", aliases=("https://doi.org/10.1234/abc",)))


def test_preprint_versions_resolve_to_same_declared_identity():
    with pytest.raises(ValidationError, match="common lineage"):
        manifest(publication(aliases=("arXiv:2301.12345v1",)),
                 publication("paper-2", aliases=("https://arxiv.org/pdf/2301.12345v2.pdf",)))


@pytest.mark.parametrize("alias", ["", " ", " PAPER-1 "])
def test_empty_and_self_aliases_fail(alias):
    with pytest.raises(ValidationError):
        publication(aliases=(alias,))


@pytest.mark.parametrize("kind", ["path", "hash"])
def test_shared_source_cannot_be_split_or_double_counted(kind):
    left = ref("tests/offline/fixtures/a.md", b"one source")
    right = ref("tests/offline/fixtures/b.md" if kind == "hash" else left.path,
                b"one source" if kind == "hash" else b"different bytes")
    with pytest.raises(ValidationError, match="common lineage"):
        manifest(publication(sources=(left,)), publication("paper-2", sources=(right,)))


def test_derivatives_group_without_extra_independent_papers():
    source = ref("tests/offline/fixtures/source.md", b"one source")
    item = manifest(publication(sources=(source,)),
                    publication("paper-chunk", derived_from="paper-1", sources=(source,)),
                    publication("paper-synthetic-edit", derived_from="paper-chunk"))
    assert {item.publication_group(pub.publication_id) for pub in item.publications} == {"paper-1"}


def test_publication_matching_uses_only_declared_document_and_original_aliases():
    item = manifest(publication(aliases=("doi:10.1234/ABC",)),
                    publication("paper-chunk", derived_from="paper-1", aliases=("chunk-source",)))
    assert item.matches_publication("paper-chunk", "https://doi.org/10.1234/abc")
    assert item.matches_publication("paper-chunk", "chunk-source")
    assert item.matches_publication("paper-chunk", "paper-1")
    assert not item.matches_publication("paper-chunk", "unknown-id")
    assert not item.matches_publication("paper-1", "chunk-source")


@pytest.mark.parametrize("publications", [
    (publication(), publication("child", derived_from="paper-1", split="historical")),
    (publication(derived_from="missing"),),
    (publication(derived_from="paper-1"),),
    (publication(derived_from="paper-2"), publication("paper-2", derived_from="paper-1")),
    (publication(), publication()),
])
def test_invalid_lineage_rejected(publications):
    with pytest.raises(ValidationError):
        manifest(*publications)


def test_unknown_variant_and_duplicate_predictions_fail():
    predicted = PredictionRef(variant="other", status="unprocessed")
    with pytest.raises(ValidationError, match="undeclared"):
        manifest(publication(predictions=(predicted,)))
    with pytest.raises(ValidationError, match="one prediction"):
        publication(predictions=(predicted, predicted))


@pytest.mark.parametrize("updates", [
    {"alignment_status": "adjudication_required"},
    {"data_kind": "natural", "alignment_status": "synthetic"},
    {"identity_gold": ref()},
    {"blocked_collections": ("samples", "samples")},
    {"sources": (ref(), ref())},
    {"publication_id": "  paper"},
    {"alignment_status": "frozen_historical", "split": "future_held_out"},
])
def test_invalid_publication_declarations_fail(updates):
    with pytest.raises(ValidationError):
        publication(**updates)


def test_adjudication_block_preserves_metric_scope():
    item = publication(alignment_status="adjudication_required", blocked_collections=("samples",))
    assert item.blocked_collections == ("samples",)


def test_manifest_fingerprint_is_deterministic_and_sensitive_to_split():
    first = manifest(publication())
    restored = SplitManifest.model_validate_json(first.model_dump_json())
    assert manifest_digest(first) == manifest_digest(restored)
    assert manifest_digest(first) != manifest_digest(manifest(publication(split="historical")))
    with pytest.raises(ValidationError):
        first.primary_outcome = "changed"


def test_same_input_path_cannot_claim_conflicting_frozen_content():
    predictions = (
        PredictionRef(variant="baseline", status="ready", artifact=ref(raw=b"old")),
        PredictionRef(variant="candidate", status="ready", artifact=ref(raw=b"new")),
    )
    with pytest.raises(ValidationError, match="conflicting frozen hashes"):
        manifest(publication(predictions=predictions))


@pytest.mark.parametrize("role", ["gold", "identity_gold", "artifact", "sidecar", "reference_report"])
def test_every_declared_artifact_stays_with_its_publication_group(role):
    first_gold = ref("tests/offline/fixtures/first-gold.json")
    second_gold = ref("tests/offline/fixtures/second-gold.json")
    shared = ref("tests/offline/fixtures/shared.json")
    if role in {"gold", "identity_gold"}:
        left = publication(**({"gold": first_gold} | {role: shared}))
        right = publication("paper-2", split="historical", **({"gold": second_gold} | {role: shared}))
    else:
        left_prediction = PredictionRef(**(dict(variant="baseline", status="ready",
                                                artifact=first_gold) | {role: shared}))
        right_prediction = PredictionRef(**(dict(variant="baseline", status="ready",
                                                 artifact=second_gold) | {role: shared}))
        left = publication(predictions=(left_prediction,))
        right = publication("paper-2", split="historical", predictions=(right_prediction,))
    with pytest.raises(ValidationError, match="unrelated publication"):
        manifest(left, right)


def test_read_only_explicit_hash_verified_artifact(workspace):
    selected = write_fixture(workspace, "chosen.json", b'{"value": 3}')
    write_fixture(workspace, "not-selected.json", b"invalid JSON")
    assert artifact_json(workspace, selected, role="prediction") == {"value": 3}
    (workspace / selected.path).write_bytes(b'{"value": 4}')
    with pytest.raises(ManifestError, match="frozen hash"):
        read_artifact(workspace, selected, role="prediction")


def test_future_gold_rejected_before_any_file_access(workspace, monkeypatch):
    import meta_data_extraction.offline.manifest as module

    def forbidden(*args, **kwargs):
        raise AssertionError("Must reject before accessing gold path")

    monkeypatch.setattr(module, "input_file", forbidden)
    with pytest.raises(ManifestError, match="authorization"):
        read_artifact(workspace, ref("data/gold/future.json"), role="gold", split="future_held_out")


def test_gold_requires_split_and_explicit_heldout_opt_in(workspace):
    selected = write_fixture(workspace, "synthetic-gold.json", b"{}")
    with pytest.raises(ManifestError, match="explicit publication split"):
        read_artifact(workspace, selected, role="gold")
    assert read_artifact(workspace, selected, role="gold", split="calibration") == b"{}"
    assert read_artifact(workspace, selected, role="gold", split="historical") == b"{}"
    assert read_artifact(workspace, selected, role="gold", split="future_held_out", allow_final_gold=True) == b"{}"


@pytest.mark.parametrize("role,path", [
    ("prediction", "data/gold/a.json"), ("gold", "data/experiment/a.json"),
    ("source", "data/gold/a.json"), ("source", "configs/alignment_helper.yaml"),
    ("report", "paper/a.json"), ("report", "outputs/paper/a.json"),
    ("gold", ".env"), ("source", "tests/offline/fixtures"),
])
def test_role_allowlist_rejects_inappropriate_paths(workspace, role, path):
    with pytest.raises(PathPolicyError):
        read_artifact(workspace, ref(path), role=role, split="historical")


def test_hardlinked_artifact_rejected(workspace):
    selected = write_fixture(workspace, "original.json", b"{}")
    target = workspace / "tests/offline/fixtures/alias.json"
    os.link(workspace / selected.path, target)
    with pytest.raises(PathPolicyError, match="Hard-linked"):
        read_artifact(workspace, ref("tests/offline/fixtures/alias.json"), role="prediction")


@pytest.mark.parametrize("raw", [
    b'{"a": 1, "a": 2}', b'{"x": NaN}', b'{"x": Infinity}', b'{"x": 1e999}',
])
def test_duplicate_or_nonfinite_json_cannot_enter_manifest(raw):
    with pytest.raises(ManifestError):
        strict_json(raw)


def test_artifact_json_requires_object(workspace):
    selected = write_fixture(workspace, "array.json", b"[]")
    with pytest.raises(ManifestError, match="object"):
        artifact_json(workspace, selected, role="prediction")


def test_manifest_rejects_unknown_fields(workspace):
    value = manifest().model_dump(mode="json") | {"allow_final_gold": True}
    selected = write_fixture(workspace, "manifest.json", json.dumps(value).encode())
    with pytest.raises(ValidationError, match="Extra inputs"):
        load_manifest(workspace, selected.path)
