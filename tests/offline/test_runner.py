"""Runner unit tests with an explicitly mocked scorer and synthetic records.

The real inherited evaluator and its fingerprint checks have separate integration
tests. These tests exercise isolation, accounting, failure visibility and replay.
"""
import hashlib
import json
from pathlib import Path

import pytest

from meta_data_extraction.agent.contracts import canonical_json
from meta_data_extraction.agent.evidence import ParserProvenance, bundle_document
from meta_data_extraction.agent.extraction import extract_fixed
from meta_data_extraction.agent.profiles import load_profile
from meta_data_extraction.agent.providers import MockProvider, ProviderResult
from meta_data_extraction.agent.storage import RunConflictError, RunIncompleteError
from meta_data_extraction.data_models import Document
from meta_data_extraction.offline import runner
from meta_data_extraction.offline.manifest import ArtifactRef, PredictionRef, Publication, SplitManifest
from meta_data_extraction.offline.scoring import EvaluatorSnapshot, ScoringError


def artifact(workspace, name, value):
    path = "tests/offline/fixtures/" + name
    target = workspace / path
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else json.dumps(value).encode()
    target.write_bytes(raw)
    return ArtifactRef(path=path, sha256=hashlib.sha256(raw).hexdigest())


def report(quality=1):
    return {"fields": {
        "samples": {"tp": 1 + quality, "fp": 0, "fn": 1 - quality, "n_gold": 2},
        "samples.attrs.description": {"mean_score": float(quality), "n_pairs": 1 + quality,
                                        "n_pairs_non_null": 1 + quality},
        "datasets.attrs.sample_refs": {"tp": quality, "fp": 0, "fn": 1 - quality, "n_pairs": 1},
        "bibliographic.title": {"score": float(quality), "gold_null": False},
    }, "evaluator": {"plan_file": "synthetic-fixture"}, "identity_sources": {}}


def synthetic_extraction(*, failed=False):
    """Reuse the shipped M02 synthetic source/response for a fully typed sidecar."""
    repository = Path(__file__).resolve().parents[2]
    source = repository / "tests/agent/fixtures/m02_source.md"
    raw = source.read_bytes()
    bundle = bundle_document(Document(id="synthetic", text=raw.decode()),
                             source_id="tests/agent/fixtures/m02_source.md", source_bytes=raw,
                             source_format="markdown", parser=ParserProvenance(name="fixture", version="1"))
    response = (repository / "tests/agent/fixtures/m02_inherited_response.json").read_text()
    provider = MockProvider([ProviderResult(status="failed" if failed else "ok",
                                           content=None if failed else response,
                                           finish_reason=None, input_tokens=12, latency_ms=0.0)])
    return extract_fixed(repository, bundle, load_profile(repository), provider, variant="inherited")


@pytest.fixture
def stub_scorer(monkeypatch):
    fields = [
        {"field": "samples", "metric": "entity", "parent_entity": None},
        {"field": "samples.attrs.description", "metric": "text", "parent_entity": "samples"},
        {"field": "datasets.attrs.sample_refs", "metric": "reference", "parent_entity": "datasets"},
        {"field": "bibliographic.title", "metric": "text", "parent_entity": None},
    ]
    payload = dict(version="m03-evaluator-1", files=[], dependencies=[], field_metrics=fields,
                   python_version="synthetic-test", metadata_json="{}")
    payload["evaluator_id"] = "evaluator:" + hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    snapshot = EvaluatorSnapshot.model_validate_json(json.dumps(payload))
    calls = []

    def score(root, gold, pred, evaluator, identity_gold=None):
        assert evaluator == snapshot
        assert gold.is_relative_to(root / "runs/agent")
        assert pred.is_relative_to(root / "runs/agent")
        calls.append((gold, pred, identity_gold))
        value = json.loads(pred.read_bytes())
        if "document_id" in value:
            value = value["output"]
        return report(value.get("fixture_quality", 1))

    monkeypatch.setattr(runner, "verify_evaluator", lambda snapshot, root: None)
    monkeypatch.setattr(runner, "_implementation", lambda root: {"synthetic_test_stub": "a" * 64})
    monkeypatch.setattr(runner, "evaluate_pair", score)
    return snapshot, calls


def ready(workspace, variant="baseline", name=None, value=None, **kwargs):
    return PredictionRef(variant=variant, status="ready", artifact=artifact(
        workspace, name or variant + ".json", {"fixture_quality": 1} if value is None else value,
    ), **kwargs)


def paper(workspace, identifier="paper-1", predictions=(), **kwargs):
    return Publication(**(dict(publication_id=identifier, split="calibration", data_kind="synthetic",
                              gold=artifact(workspace, identifier + "-gold.json", {"fixture_gold": True}),
                              predictions=predictions, alignment_status="synthetic") | kwargs))


def manifest(*publications, variants=("baseline", "candidate")):
    return SplitManifest(experiment_id="runner-unit-fixture", description="Synthetic offline harness test",
                         primary_outcome="Inherited fields separately with support", variants=variants,
                         publications=publications, split_notes="Invented fixtures only",
                         held_out_policy="No natural paper gold selected")


def execute(workspace, scope, stub_scorer, run_id="unit-run", **kwargs):
    return runner.run_offline(workspace, scope, run_id=run_id, evaluator=stub_scorer[0], **kwargs)


def test_fresh_run_scores_only_isolated_copies_and_replays_without_rescoring(workspace, stub_scorer):
    scope = manifest(paper(workspace, predictions=(ready(workspace), ready(workspace, "candidate"))))
    first = execute(workspace, scope, stub_scorer)
    assert first["execution"] == "offline_scoring"
    assert first["new_evaluations"] == 2
    assert first["result"]["model_calls"] == 0
    assert len(stub_scorer[1]) == 2
    assert all(gold.name == "paper-1-gold.json" for gold, _, _ in stub_scorer[1])
    completion = workspace / "runs/agent/unit-run/result.json"
    before = completion.read_bytes()
    replay = execute(workspace, scope, stub_scorer)
    assert replay["execution"] == "cached_replay"
    assert replay["new_evaluations"] == 0
    assert replay["result"] == first["result"]
    assert len(stub_scorer[1]) == 2
    assert completion.read_bytes() == before
    assert not (workspace / "outputs").exists()


def test_changed_run_request_conflicts_and_incomplete_run_never_rescores(workspace, stub_scorer):
    original = manifest(paper(workspace, predictions=(ready(workspace),)), variants=("baseline",))
    execute(workspace, original, stub_scorer)
    changed = SplitManifest.model_validate(original.model_dump(mode="json") | {"primary_outcome": "changed"})
    with pytest.raises(RunConflictError):
        execute(workspace, changed, stub_scorer)
    (workspace / "runs/agent/unit-run/result.json").unlink()
    with pytest.raises(RunIncompleteError):
        execute(workspace, original, stub_scorer)
    assert len(stub_scorer[1]) == 1


def test_changed_input_blocks_verified_replay(workspace, stub_scorer):
    pred = ready(workspace)
    scope = manifest(paper(workspace, predictions=(pred,)), variants=("baseline",))
    execute(workspace, scope, stub_scorer)
    (workspace / pred.artifact.path).write_text('{"changed": true}')
    with pytest.raises(ValueError, match="frozen scope"):
        execute(workspace, scope, stub_scorer)
    assert len(stub_scorer[1]) == 1


def test_result_tampering_rejected_before_rescoring(workspace, stub_scorer):
    scope = manifest(paper(workspace, predictions=(ready(workspace),)), variants=("baseline",))
    execute(workspace, scope, stub_scorer)
    (workspace / "runs/agent/unit-run/sidecar.json").write_text("{}")
    with pytest.raises(RunConflictError):
        execute(workspace, scope, stub_scorer)
    assert len(stub_scorer[1]) == 1


def test_metric_categories_support_and_harmed_comparison_remain_separate(workspace, stub_scorer):
    scope = manifest(paper(workspace, predictions=(
        ready(workspace), ready(workspace, "candidate", value={"fixture_quality": 0}),
    )))
    result = execute(workspace, scope, stub_scorer)["result"]
    assert {row["category"] for row in result["metric_rows"]} == {
        "entity_recovery", "conditional_attributes", "references", "scalar_fields",
    }
    assert all(row["support_count"] > 0 for row in result["metric_rows"])
    assert all(row["status"] == "harmed" and row["delta"] < 0 for row in result["comparisons"])
    assert not any("average" in row["field"] for row in result["metric_rows"])


def test_improvement_uses_the_same_field_and_positive_support(workspace, stub_scorer):
    scope = manifest(paper(workspace, predictions=(
        ready(workspace, value={"fixture_quality": 0}), ready(workspace, "candidate"),
    )))
    comparisons = execute(workspace, scope, stub_scorer)["result"]["comparisons"]
    assert all(row["status"] == "improved" and row["delta"] > 0 for row in comparisons)
    assert {row["metric"] for row in comparisons} == {"f1", "mean_score"}
    assert all(row["baseline_support"]["support_count"] > 0 and row["variant_support"]["support_count"] > 0
               for row in comparisons)


def test_failed_unprocessed_and_unknown_resources_are_visible(workspace, stub_scorer):
    failed = PredictionRef(variant="baseline", status="failed", failure_code="timeout", request_count=1)
    absent = PredictionRef(variant="candidate", status="unprocessed")
    result = execute(workspace, manifest(paper(workspace, predictions=(failed, absent))), stub_scorer)["result"]
    assert [item["status"] for item in result["records"]] == ["failed", "unprocessed"]
    assert not stub_scorer[1]
    assert result["new_evaluations"] == 0
    assert result["comparisons"][0]["status"] == "unpaired"
    for row in result["coverage"]:
        assert row["publications"] == 1
        assert row["resources_by_execution_mode"]["unknown"]["cost"] == {
            "total": None, "known_subtotal": None, "unknown_records": 1,
        }


def test_partial_known_usage_does_not_become_complete_total(workspace, stub_scorer):
    first = ready(workspace, name="first.json", request_count=1, input_tokens=20)
    second = ready(workspace, name="second.json")
    scope = manifest(paper(workspace, predictions=(first,)),
                     paper(workspace, "paper-2", predictions=(second,)), variants=("baseline",))
    resources = execute(workspace, scope, stub_scorer)["result"]["coverage"][0]["resources_by_execution_mode"]["unknown"]
    assert resources["input_tokens"] == {"total": None, "known_subtotal": 20, "unknown_records": 1}
    assert resources["request_count"] == {"total": None, "known_subtotal": 1, "unknown_records": 1}


def test_natural_and_synthetic_results_never_pool(workspace, stub_scorer):
    scope = manifest(paper(workspace, predictions=(ready(workspace, name="synthetic.json"),)),
                     paper(workspace, "natural-fixture-label", data_kind="natural", alignment_status="frozen_historical",
                           aliases=("natural-fixture-label-gold",),
                           predictions=(ready(workspace, name="natural.json"),)), variants=("baseline",))
    result = execute(workspace, scope, stub_scorer)["result"]
    assert {(row["data_kind"], row["publications"]) for row in result["coverage"]} == {("synthetic", 1), ("natural", 1)}
    assert all(row["n_papers"] == 1 for row in result["metric_rows"])
    assert all(row["status"] == "scored" for row in result["records"])


def test_historical_gold_basename_must_match_declared_publication_before_read(workspace, stub_scorer, monkeypatch):
    scope = manifest(paper(workspace, data_kind="natural", alignment_status="frozen_historical",
                           predictions=(ready(workspace),)), variants=("baseline",))
    original = runner.read_artifact
    reads = []
    def observed(*args, **kwargs):
        reads.append(kwargs["role"])
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, "read_artifact", observed)
    result = execute(workspace, scope, stub_scorer)["result"]
    assert result["records"][0]["status"] == "blocked"
    assert result["records"][0]["diagnostics"] == ["gold_publication_identity_mismatch"]
    assert reads == []
    assert not stub_scorer[1]


@pytest.mark.parametrize("document_id", ["another-paper", None, 3])
def test_historical_prediction_document_id_cannot_silently_score_another_paper(workspace, stub_scorer, document_id):
    value = {"document_id": document_id, "output": {"fixture_quality": 1}, "warnings": []}
    scope = manifest(paper(workspace, data_kind="natural", alignment_status="frozen_historical",
                           aliases=("paper-1-gold",), predictions=(ready(workspace, value=value),)),
                     variants=("baseline",))
    result = execute(workspace, scope, stub_scorer)["result"]
    assert result["records"][0]["status"] == "failed"
    assert result["records"][0]["diagnostics"] == ["prediction_publication_identity_mismatch"]
    assert not stub_scorer[1]


def test_historical_prediction_accepts_declared_alias(workspace, stub_scorer):
    value = {"document_id": "https://doi.org/10.1234/example", "output": {"fixture_quality": 1}, "warnings": []}
    scope = manifest(paper(workspace, data_kind="natural", alignment_status="frozen_historical",
                           aliases=("paper-1-gold", "doi:10.1234/example"),
                           predictions=(ready(workspace, value=value),)), variants=("baseline",))
    assert execute(workspace, scope, stub_scorer)["result"]["records"][0]["status"] == "scored"


def test_derivatives_cannot_supply_duplicate_observations(workspace, stub_scorer):
    scope = manifest(paper(workspace, predictions=(ready(workspace, name="original.json"),)),
                     paper(workspace, "chunk", derived_from="paper-1", predictions=(ready(workspace, name="chunk.json"),)),
                     variants=("baseline",))
    with pytest.raises(ValueError, match="one record per publication group"):
        execute(workspace, scope, stub_scorer)
    assert not stub_scorer[1]
    assert not (workspace / "runs").exists()


def test_new_paper_unreviewed_identity_metrics_are_blocked(workspace, stub_scorer):
    scope = manifest(paper(workspace, predictions=(ready(workspace),), alignment_status="unreviewed"),
                     variants=("baseline",))
    result = execute(workspace, scope, stub_scorer)["result"]
    row = result["records"][0]
    assert row["status"] == "partial"
    assert set(row["blocked_fields"]) == {"samples", "samples.attrs.description", "datasets.attrs.sample_refs"}
    assert {row["field"] for row in result["metric_rows"]} == {"bibliographic.title"}
    saved = json.loads((workspace / "runs/agent/unit-run/sidecar.json").read_bytes())
    assert "samples" in saved["reports"][0]["report"]["fields"]


def test_adjudication_block_includes_references_to_blocked_entities(workspace, stub_scorer):
    scope = manifest(paper(workspace, predictions=(ready(workspace),),
                           alignment_status="adjudication_required", blocked_collections=("samples",)),
                     variants=("baseline",))
    result = execute(workspace, scope, stub_scorer)["result"]
    assert {row["field"] for row in result["metric_rows"]} == {"bibliographic.title"}


def test_final_gold_flag_required_before_creating_run_or_reading_gold(workspace, stub_scorer, monkeypatch):
    scope = manifest(paper(workspace, predictions=(ready(workspace),), split="future_held_out"),
                     variants=("baseline",))
    original = runner.read_artifact
    reads = []

    def observed(*args, **kwargs):
        reads.append(kwargs["role"])
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "read_artifact", observed)
    with pytest.raises(ValueError, match="explicitly authorized"):
        execute(workspace, scope, stub_scorer)
    assert reads == []
    assert not (workspace / "runs").exists()
    assert execute(workspace, scope, stub_scorer, allow_final_gold=True)["new_evaluations"] == 1


@pytest.mark.parametrize("value", [
    b'[]', b'{"x": NaN}', b'{"x": 1, "x": 2}', b'{"x": 1e999}',
    {"document_id": "synthetic", "output": [], "warnings": []},
    {"document_id": "synthetic", "output": {},
     "warnings": ["JSON parse failed after fallback extraction; using empty output."]},
    {"document_id": "synthetic", "output": {},
     "warnings": ["LLM output parsed but is not a JSON object; using empty output."]},
])
def test_invalid_predictions_never_become_successful_empty_records(workspace, stub_scorer, value):
    scope = manifest(paper(workspace, predictions=(ready(workspace, value=value),)), variants=("baseline",))
    result = execute(workspace, scope, stub_scorer)["result"]
    assert result["records"][0]["status"] == "failed"
    assert not stub_scorer[1]


@pytest.mark.parametrize("extraction", [
    None, [], "not an extraction object",
    {"status": "failed"}, {"response_truncated": True},
    {"status": "succeeded", "snapshot": {"canonical_json": '{"different":true}'}},
])
def test_failed_or_mismatching_extraction_sidecar_is_not_scored(workspace, stub_scorer, extraction):
    sidecar = artifact(workspace, "sidecar.json", {"extraction": extraction})
    scope = manifest(paper(workspace, predictions=(ready(workspace, sidecar=sidecar),)), variants=("baseline",))
    result = execute(workspace, scope, stub_scorer)["result"]
    assert result["records"][0]["status"] == "failed"
    assert not stub_scorer[1]


def test_matching_sidecar_preserves_usage_and_execution_mode(workspace, stub_scorer):
    extraction = synthetic_extraction()
    assert extraction.status == "needs_review"
    value = extraction.snapshot.canonical
    sidecar = artifact(workspace, "sidecar.json", {"extraction": extraction.model_dump(mode="json")})
    scope = manifest(paper(workspace, predictions=(ready(workspace, value=value, sidecar=sidecar),)), variants=("baseline",))
    result = execute(workspace, scope, stub_scorer)["result"]
    resources = result["records"][0]["resources"]
    assert resources["input_tokens"] == 12
    assert resources["output_tokens"] is None
    assert resources["execution_mode"] == "mock"
    assert resources["request_count"] is None
    assert result["records"][0]["status"] == "partial"
    assert result["records"][0]["extraction_status"] == "needs_review"
    assert result["records"][0]["evidence_support"] == "not_assessed_by_this_evaluator"


def test_failed_sidecar_usage_preserved_without_reading_gold(workspace, stub_scorer, monkeypatch):
    extraction = synthetic_extraction(failed=True)
    sidecar = artifact(workspace, "failure-sidecar.json", {"extraction": extraction.model_dump(mode="json")})
    failed = PredictionRef(variant="baseline", status="failed", failure_code="provider_failed", sidecar=sidecar)
    scope = manifest(paper(workspace, predictions=(failed,)), variants=("baseline",))
    original = runner.read_artifact
    roles = []
    def observed(*args, **kwargs):
        roles.append(kwargs["role"])
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, "read_artifact", observed)
    result = execute(workspace, scope, stub_scorer)["result"]
    assert roles == ["sidecar"]
    assert result["records"][0]["resources"]["input_tokens"] == 12
    assert result["records"][0]["resources"]["execution_mode"] == "mock"
    assert result["records"][0]["status"] == "failed"


def test_omitted_variant_has_explicit_unprocessed_record_and_denominator(workspace, stub_scorer):
    scope = manifest(paper(workspace, predictions=(ready(workspace),)))
    result = execute(workspace, scope, stub_scorer)["result"]
    candidate = next(row for row in result["records"] if row["variant"] == "candidate")
    assert candidate["status"] == "unprocessed"
    assert candidate["diagnostics"] == ["prediction_not_declared"]
    assert all(row["publications"] == 1 for row in result["coverage"])


def test_cached_and_live_resource_measurements_are_not_added_together(workspace, stub_scorer):
    scope = manifest(paper(workspace, predictions=(ready(workspace, name="live.json", execution_mode="live", request_count=1),)),
                     paper(workspace, "paper-2", predictions=(ready(workspace, name="cached.json", execution_mode="cached_replay", request_count=0),)),
                     variants=("baseline",))
    result = execute(workspace, scope, stub_scorer)["result"]
    resources = result["coverage"][0]["resources_by_execution_mode"]
    assert set(resources) == {"live", "cached_replay"}
    assert resources["live"]["request_count"]["total"] == 1
    assert resources["cached_replay"]["request_count"]["total"] == 0


def test_tampered_staged_scoring_input_rejects_replay(workspace, stub_scorer):
    scope = manifest(paper(workspace, predictions=(ready(workspace),)), variants=("baseline",))
    result = execute(workspace, scope, stub_scorer)["result"]
    (workspace / result["staged_inputs"][0]["path"]).write_bytes(b'{"tampered": true}')
    with pytest.raises(ValueError, match="Staged scoring input"):
        execute(workspace, scope, stub_scorer)
    assert len(stub_scorer[1]) == 1


def test_scoring_failure_is_sanitized_and_other_records_continue(workspace, stub_scorer, monkeypatch):
    original = runner.evaluate_pair
    def fail_one(root, gold, pred, snapshot, identity_gold=None):
        if gold.name == "paper-1-gold.json":
            raise ScoringError("synthetic_scorer_unavailable")
        return original(root, gold, pred, snapshot, identity_gold=identity_gold)

    monkeypatch.setattr(runner, "evaluate_pair", fail_one)
    scope = manifest(paper(workspace, predictions=(ready(workspace, name="first.json"),)),
                     paper(workspace, "paper-2", predictions=(ready(workspace, name="second.json"),)), variants=("baseline",))
    result = execute(workspace, scope, stub_scorer)["result"]
    assert [row["status"] for row in result["records"]] == ["failed", "scored"]
    assert result["records"][0]["diagnostics"] == ["synthetic_scorer_unavailable"]


@pytest.mark.parametrize("matching", [True, False])
def test_reference_report_parity_is_visible_per_document(workspace, stub_scorer, matching):
    reference = artifact(workspace, "reference.json", report(1 if matching else 0))
    scope = manifest(paper(workspace, predictions=(ready(workspace, reference_report=reference),)), variants=("baseline",))
    result = execute(workspace, scope, stub_scorer)["result"]
    assert result["records"][0]["parity"]["status"] == ("matched" if matching else "mismatch")


def test_missing_gold_remains_blocked_without_scoring(workspace, stub_scorer):
    scope = manifest(paper(workspace, gold=None, predictions=(ready(workspace),)), variants=("baseline",))
    result = execute(workspace, scope, stub_scorer)["result"]
    assert result["records"][0]["status"] == "blocked"
    assert result["records"][0]["diagnostics"] == ["gold_not_supplied"]
    assert not stub_scorer[1]
