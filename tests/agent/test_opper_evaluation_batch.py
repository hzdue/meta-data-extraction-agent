"""Offline tests for the 30-paper Opper batch; no credential or network use."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from meta_data_extraction.agent import opper_evaluation_batch as batch


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def batch_workspace(workspace):
    papers = []
    config_papers = []
    for index in range(1, 31):
        arxiv_id = f"{1800 + index:04d}.{index:05d}"
        raw = f"synthetic PDF bytes {index}".encode()
        source = workspace / f"data/raw/{arxiv_id}/paper.pdf"
        source.parent.mkdir(parents=True)
        source.write_bytes(raw)
        papers.append({"arxiv_id":arxiv_id, "split":"evaluation"})
        config_papers.append({
            "ordinal":index, "arxiv_id":arxiv_id,
            "source":{"path":f"data/raw/{arxiv_id}/paper.pdf", "sha256":_hash(raw)},
        })
    corpus = workspace / "data/corpus.json"
    corpus.parent.mkdir(parents=True, exist_ok=True)
    corpus.write_text(json.dumps({"papers":papers}), encoding="utf-8")
    config = {
        "version":"opper-evaluation-batch-1", "name":"historical-evaluation-30",
        "corpus":{"path":"data/corpus.json", "sha256":_hash(corpus.read_bytes())},
        "split":"evaluation",
        "provider":{"name":"opper", "endpoint":batch.ENDPOINT, "model":batch.MODEL,
                    "api_key_env":"OPPER_API_KEY"},
        "limits":{"max_calls_per_paper":3, "max_output_tokens_per_call":8192,
                  "max_prompt_bytes_per_call":130000,
                  "max_input_tokens_per_call":119000, "timeout_seconds":240,
                  "preflight_timeout_seconds":30, "maximum_papers":30,
                  "maximum_total_calls":90},
        "papers":config_papers,
    }
    path = workspace / "configs/agent/batch.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config), encoding="utf-8")
    return workspace


def _ready_preflight(paper, *, eligible=True, error=None, prompt_bytes=1000,
                     exact_tokens=200):
    return {
        "version":"opper-paper-preflight-1", "ordinal":paper.ordinal,
        "arxiv_id":paper.arxiv_id, "status":"ready" if error is None else "unavailable",
        "error_code":error, "eligible":eligible,
        "first_prompt_utf8_bytes_exact":prompt_bytes,
        "first_input_tokens_chars_div_4_estimate":250,
        "first_input_tokens_o200k_exact":exact_tokens,
    }


def _prepare_ready(batch_workspace, batch_id, monkeypatch):
    batch.prepare(batch_workspace, batch_id, "configs/agent/batch.json")
    monkeypatch.setattr(
        batch, "_run_preflight_worker",
        lambda root, bid, paper, timeout: _ready_preflight(paper),
    )
    batch.preflight(batch_workspace, batch_id)


def _catalog_ready(batch_workspace, batch_id, monkeypatch):
    body = {"data":[{
        "id":batch.MODEL, "context_length":128000,
        "pricing":{"prompt":"0.0000025", "completion":"0.00001"},
        "opper":{"max_output_tokens":16384},
    }]}
    monkeypatch.setattr(
        batch, "_models_request", lambda key: (200, {}, json.dumps(body).encode()))
    batch.check_catalog(batch_workspace, batch_id, "synthetic-secret-key")


def test_prepare_freezes_exact_evaluation_sources_without_gold(batch_workspace):
    gold = batch_workspace / "data/gold/do-not-read.json"
    gold.parent.mkdir(parents=True)
    gold.write_text("not json and not an input", encoding="utf-8")
    result = batch.prepare(
        batch_workspace, "m10-e30-unit-v1", "configs/agent/batch.json")
    prep = batch_workspace / result["prep_relative"]
    manifest = json.loads((prep / "source-manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((prep / "protocol.json").read_text(encoding="utf-8"))
    assert result["paper_count"] == 30 and result["model_requests"] == 0
    assert len(manifest["papers"]) == 30
    assert protocol["gold_access"] is False and protocol["evaluator"] is None
    assert protocol["budgets"]["maximum_total_calls"] == 90
    assert protocol["budgets"]["conservative_batch_input_token_equivalents"] == 10710000
    assert protocol["budgets"]["reserved_batch_output_tokens"] == 737280


def test_prepare_rejects_split_or_source_tampering(batch_workspace):
    corpus = batch_workspace / "data/corpus.json"
    payload = json.loads(corpus.read_text(encoding="utf-8"))
    payload["papers"][0]["split"] = "calibration"
    corpus.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(batch.BatchError, match="corpus_hash_mismatch"):
        batch.prepare(batch_workspace, "m10-e30-unit-v2", "configs/agent/batch.json")


def test_preflight_is_partial_replay_safe_and_credential_free(batch_workspace, monkeypatch):
    run_id = "m10-e30-unit-v3"
    batch.prepare(batch_workspace, run_id, "configs/agent/batch.json")
    calls = []

    def worker(root, bid, paper, timeout):
        calls.append(paper.ordinal)
        return _ready_preflight(paper)

    monkeypatch.setattr(batch, "_run_preflight_worker", worker)
    first = batch.preflight(batch_workspace, run_id)
    second = batch.preflight(batch_workspace, run_id)
    assert calls == list(range(1, 31))
    assert first == second and first["batch_eligible"] is True
    assert first["credential_read"] is False and first["network_requests"] == 0


def test_preflight_keeps_timeout_and_oversize_explicit(batch_workspace, monkeypatch):
    run_id = "m10-e30-unit-v4"
    batch.prepare(batch_workspace, run_id, "configs/agent/batch.json")

    def worker(root, bid, paper, timeout):
        if paper.ordinal == 1:
            return _ready_preflight(paper, eligible=False, error="preflight_timeout")
        if paper.ordinal == 2:
            return _ready_preflight(paper, eligible=False, prompt_bytes=130001)
        return _ready_preflight(paper)

    monkeypatch.setattr(batch, "_run_preflight_worker", worker)
    result = batch.preflight(batch_workspace, run_id)
    assert result["batch_eligible"] is False and result["eligible_count"] == 28
    assert result["failed"][0]["error_code"] == "preflight_timeout"
    assert result["failed"][1]["error_code"] == "prompt_size_limit"


def test_catalog_is_redacted_and_produces_hard_cost_ceiling(batch_workspace, monkeypatch):
    run_id = "m10-e30-unit-v5"
    _prepare_ready(batch_workspace, run_id, monkeypatch)
    _catalog_ready(batch_workspace, run_id, monkeypatch)
    prep, config = batch._load_prepared(batch_workspace, run_id)
    _, ceiling = batch._catalog_and_cost(prep, config)
    retained = b"".join(path.read_bytes() for path in prep.path.glob("*.json"))
    assert ceiling == Decimal("34.1478000")
    assert b"synthetic-secret-key" not in retained


def test_run_requires_cost_acknowledgement_and_three_shared_slots(
        batch_workspace, monkeypatch):
    run_id = "m10-e30-unit-v6"
    _prepare_ready(batch_workspace, run_id, monkeypatch)
    _catalog_ready(batch_workspace, run_id, monkeypatch)
    monkeypatch.setenv("OPPER_API_KEY", "synthetic-secret-key")
    monkeypatch.setattr(batch, "run_publication", lambda *a, **k: pytest.fail("no dispatch"))
    with pytest.raises(batch.BatchError, match="cost_ceiling_acknowledgement_required"):
        batch.run_available(
            batch_workspace, run_id, confirm_live=True, retention_reviewed=True,
            load_dotenv_file=False, accepted_cost_ceiling_usd=Decimal("29"))
    monkeypatch.setattr(batch, "OpperRequestLedger", lambda root: SimpleNamespace(
        capacity=lambda: SimpleNamespace(error_code=None, remaining_attempts=2)))
    result = batch.run_available(
        batch_workspace, run_id, confirm_live=True, retention_reviewed=True,
        load_dotenv_file=False, accepted_cost_ceiling_usd=Decimal("35"))
    assert result["stop_reason"] == "waiting_for_rolling_request_capacity"
    assert result["completed_this_invocation"] == []


def test_run_is_sequential_and_bounded_by_invocation_limit(batch_workspace, monkeypatch):
    run_id = "m10-e30-unit-v7"
    _prepare_ready(batch_workspace, run_id, monkeypatch)
    _catalog_ready(batch_workspace, run_id, monkeypatch)
    monkeypatch.setenv("OPPER_API_KEY", "synthetic-secret-key")
    monkeypatch.setattr(batch, "OpperRequestLedger", lambda root: SimpleNamespace(
        capacity=lambda: SimpleNamespace(error_code=None, remaining_attempts=10)))
    seen = []

    def run(root, spec, **kwargs):
        seen.append(spec.run_id)
        return {"status":"pending_review", "new_model_calls":2,
                "new_live_requests":2, "known_input_tokens":100,
                "known_output_tokens":10, "usage_unknown":False}

    monkeypatch.setattr(batch, "run_publication", run)
    monkeypatch.setattr(batch, "_status", lambda *a: {"completed_papers":len(seen)})
    result = batch.run_available(
        batch_workspace, run_id, confirm_live=True, retention_reviewed=True,
        load_dotenv_file=False, accepted_cost_ceiling_usd=Decimal("35"),
        max_papers=2)
    assert len(seen) == 2 and len(result["completed_this_invocation"]) == 2
    assert result["stop_reason"] == "invocation_paper_limit"
    assert seen[0].endswith("-p01-180100001")
