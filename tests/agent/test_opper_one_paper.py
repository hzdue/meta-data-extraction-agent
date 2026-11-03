"""Offline tests for the one-paper Opper launcher; no credential or network use."""
from pathlib import Path
import json
import shutil
from types import SimpleNamespace

import pytest

from meta_data_extraction.agent.evidence import sha256
from meta_data_extraction.agent import opper_one_paper as pilot


@pytest.fixture
def pilot_workspace(workspace, monkeypatch):
    repository = Path(__file__).resolve().parents[2]
    shutil.copyfile(repository / "schemas/prompt_template.md", workspace / "schemas/prompt_template.md")
    source = workspace / "data/raw/one/paper.md"
    source.parent.mkdir(parents=True)
    source.write_text("# Synthetic paper\n\nThe beam energy was 12 keV.\n", encoding="utf-8")
    monkeypatch.setattr(pilot, "SOURCE", "data/raw/one/paper.md")
    monkeypatch.setattr(pilot, "SOURCE_SHA256", sha256(source.read_bytes()))
    return workspace


def test_prepare_freezes_one_paper_model_budgets_and_token_plan(pilot_workspace, monkeypatch):
    monkeypatch.delenv("OPPER_API_KEY", raising=False)
    result = pilot.prepare(pilot_workspace, "m10-opper-unit-v1")
    prep = pilot_workspace / result["prep_relative"]
    spec = json.loads((prep / "publication-spec.json").read_text(encoding="utf-8"))
    plan = json.loads((prep / "token-plan.json").read_text(encoding="utf-8"))
    protocol = json.loads((prep / "protocol.json").read_text(encoding="utf-8"))
    assert spec["source"]["path"] == pilot.SOURCE and spec["split"] == "calibration"
    assert spec["live"]["provider"] == "opper"
    assert spec["live"]["model"] == "openai/gpt-4o-2024-08-06"
    assert spec["max_calls"] == 3 and spec["live"]["max_output_tokens"] == 8192
    assert plan["whole_run_output_tokens_reserved"] == 24576
    assert plan["first_prompt_utf8_bytes_exact"] > 0
    assert plan["measurement_status"] == "preflight_estimates_not_tokenizer_measurements"
    assert protocol["source"]["publication_count"] == 1 and protocol["evaluator"] is None
    assert len(protocol["launcher_sha256"]) == 64
    assert result["network_requests"] == result["model_requests"] == 0
    assert b"OPPER_API_KEY" not in (prep / "readiness.json").read_bytes()


def test_catalog_check_is_one_redacted_administrative_attempt(pilot_workspace, monkeypatch):
    run_id = "m10-opper-unit-v2"
    pilot.prepare(pilot_workspace, run_id)
    calls = []

    def request(key):
        calls.append(key)
        body = {"data":[{"id":pilot.MODEL, "context_length":128000,
            "pricing":{"prompt":"0.0000025", "completion":"0.00001"},
            "opper":{"max_output_tokens":16384, "region":"us", "gdpr_residency":"us"}}]}
        return 200, {"x-ratelimit-remaining-requests":"99", "server":"private"}, json.dumps(body).encode()

    monkeypatch.setattr(pilot, "_models_request", request)
    result = pilot.check_catalog(pilot_workspace, run_id, "synthetic-secret-key")
    assert calls == ["synthetic-secret-key"] and result["model_found"] is True
    assert result["model_metadata"]["context_length"] == 128000
    assert result["numeric_limit_headers"] == {"x-ratelimit-remaining-requests":99}
    prep = pilot_workspace / "runs/agent" / pilot._prep_id(run_id)
    retained = b"".join(path.read_bytes() for path in prep.glob("*.json"))
    assert b"synthetic-secret-key" not in retained and b"private" not in retained
    with pytest.raises(FileExistsError):
        pilot.check_catalog(pilot_workspace, run_id, "synthetic-secret-key")


def _successful_catalog(workspace, run_id, monkeypatch):
    body = {"data":[{"id":pilot.MODEL, "context_length":128000,
                     "pricing":{}, "opper":{"max_output_tokens":16384}}]}
    monkeypatch.setattr(pilot, "_models_request", lambda key: (200, {}, json.dumps(body).encode()))
    pilot.check_catalog(workspace, run_id, "synthetic-secret-key")


def test_execute_fails_closed_before_publication_dispatch(pilot_workspace, monkeypatch):
    run_id = "m10-opper-unit-v3"
    pilot.prepare(pilot_workspace, run_id)
    monkeypatch.setattr(pilot, "run_publication", lambda *a, **k: pytest.fail("must not dispatch"))
    with pytest.raises(pilot.OpperPilotError, match="live_confirmation_required"):
        pilot.execute(pilot_workspace, run_id, confirm_live=False,
                      retention_reviewed=True, load_dotenv_file=False)
    with pytest.raises(pilot.OpperPilotError, match="retention_review_required"):
        pilot.execute(pilot_workspace, run_id, confirm_live=True,
                      retention_reviewed=False, load_dotenv_file=False)
    with pytest.raises(pilot.OpperPilotError, match="successful_catalog_check_required"):
        pilot.execute(pilot_workspace, run_id, confirm_live=True,
                      retention_reviewed=True, load_dotenv_file=False)
    _successful_catalog(pilot_workspace, run_id, monkeypatch)
    monkeypatch.delenv("OPPER_API_KEY", raising=False)
    with pytest.raises(pilot.OpperPilotError, match="missing_credentials"):
        pilot.execute(pilot_workspace, run_id, confirm_live=True,
                      retention_reviewed=True, load_dotenv_file=False)


@pytest.mark.parametrize(
    ("context_length", "max_output_tokens", "error"),
    [(108_319, 16_384, "model_context_too_small"),
     (128_000, 8_191, "model_output_limit_too_small"),
     (None, 16_384, "model_limits_unknown")],
)
def test_execute_checks_catalog_limits_before_credentials(
        pilot_workspace, monkeypatch, context_length, max_output_tokens, error):
    run_id = "m10-opper-limit-v1"
    pilot.prepare(pilot_workspace, run_id)
    body = {"data":[{"id":pilot.MODEL, "context_length":context_length,
                     "pricing":{}, "opper":{"max_output_tokens":max_output_tokens}}]}
    monkeypatch.setattr(
        pilot, "_models_request", lambda key: (200, {}, json.dumps(body).encode()))
    pilot.check_catalog(pilot_workspace, run_id, "synthetic-secret-key")
    monkeypatch.delenv("OPPER_API_KEY", raising=False)
    monkeypatch.setattr(pilot, "run_publication", lambda *a, **k: pytest.fail("must not dispatch"))
    with pytest.raises(pilot.OpperPilotError, match=error):
        pilot.execute(pilot_workspace, run_id, confirm_live=True,
                      retention_reviewed=True, load_dotenv_file=False)


def test_execute_reports_provider_usage_without_exposing_content(pilot_workspace, monkeypatch):
    run_id = "m10-opper-unit-v4"
    pilot.prepare(pilot_workspace, run_id)
    _successful_catalog(pilot_workspace, run_id, monkeypatch)
    monkeypatch.setenv("OPPER_API_KEY", "synthetic-secret-key")
    monkeypatch.setattr(pilot, "OpperRequestLedger", lambda root: SimpleNamespace(
        reserve=lambda *a, **k: SimpleNamespace(error_code=None)))
    monkeypatch.setattr(pilot, "run_publication", lambda *a, **k: {
        "status":"pending_review", "workflow_status":"pending_review",
        "new_model_calls":3, "new_live_requests":3,
        "known_input_tokens":21000, "known_output_tokens":5000,
        "usage_unknown":False, "cost":None, "cost_status":"unknown",
        "canonical_relative":"runs/agent/x/canonical.json",
        "sidecar_relative":"runs/agent/x/sidecar.json",
    })
    result = pilot.execute(pilot_workspace, run_id, confirm_live=True,
                           retention_reviewed=True, load_dotenv_file=False)
    assert result["known_input_tokens"] == 21000 and result["known_output_tokens"] == 5000
    assert result["maximum_reserved_output_tokens"] == 24576
    assert "content" not in result and "synthetic-secret-key" not in json.dumps(result)


def test_execute_can_record_explicit_run_scoped_history_override(
        pilot_workspace, monkeypatch):
    run_id = "m10-opper-override-v1"
    pilot.prepare(pilot_workspace, run_id)
    _successful_catalog(pilot_workspace, run_id, monkeypatch)
    monkeypatch.setenv("OPPER_API_KEY", "synthetic-secret-key")
    monkeypatch.setattr(pilot, "run_publication", lambda *a, **k: {
        "status":"pending_review", "workflow_status":"pending_review",
        "new_model_calls":0, "new_live_requests":0,
        "known_input_tokens":0, "known_output_tokens":0,
        "usage_unknown":False, "cost":None, "cost_status":"unknown",
        "canonical_relative":None, "sidecar_relative":None,
    })
    result = pilot.execute(
        pilot_workspace, run_id, confirm_live=True, retention_reviewed=True,
        load_dotenv_file=False, acknowledge_untracked_history=True)
    assert result["untracked_history_acknowledged"] is True
    expected = pilot._history_override_run_ids(pilot._fixed_spec(pilot.SOURCE_SHA256, run_id))
    assert len(expected) == 5
    spec = pilot._fixed_spec(pilot.SOURCE_SHA256, run_id)
    bundle, profile, prompt = pilot._inputs(pilot_workspace, spec)
    scope = pilot._first_scope(spec, bundle, profile, prompt)
    ledger = pilot.OpperRequestLedger(pilot_workspace)
    for override_run_id in expected:
        override = (pilot_workspace / "runs/agent/_control" /
                    f"opper-history-override-{override_run_id}.json")
        payload = json.loads(override.read_text(encoding="utf-8"))
        assert payload["run_id"] == override_run_id
        assert payload["local_attempts"] == 0
        assert ledger.reserve(
            override_run_id, scope, commit_attempt=False).error_code is None


def test_report_preserves_unknown_usage(pilot_workspace, monkeypatch):
    run_id = "m10-opper-unit-v5"
    pilot.prepare(pilot_workspace, run_id)
    monkeypatch.setattr(pilot, "read_publication", lambda *a: {
        "status":"pending_review", "workflow_status":"pending_review",
        "total_model_calls":2, "total_live_requests":2,
        "known_input_tokens":0, "known_output_tokens":0,
        "usage_unknown":True, "cost":None, "cost_status":"unknown",
    })
    result = pilot.report(pilot_workspace, run_id)
    assert result["usage_unknown"] is True
    assert result["known_input_tokens"] == result["known_output_tokens"] == 0
    assert result["token_plan"]["authoritative_after_run"].endswith("unknown remains unknown")
