"""Prepare and operate the guarded 30-paper Opper evaluation batch."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from tiktoken import get_encoding

from .evidence import FrozenModel, sha256
from .extraction import render_prompt
from .opper_one_paper import (
    CATALOG_MAX_BYTES, _model_metadata, _models_request, _numeric_headers,
)
from .paths import RunDirectory, checked_path, input_file, workspace_root
from .profiles import load_profile
from .providers import OpperRequestLedger
from .publication_run import read_publication, run_publication
from .publication_types import PublicationSpec
from .sources import prepare_sources


VERSION = "m10-opper-evaluation-30-1"
DEFAULT_CONFIG = "configs/agent/opper_evaluation_30.json"
ENDPOINT = "https://api.opper.ai/v3/compat"
MODEL = "openai/gpt-4o-2024-08-06"
WORKER_MARKER = "M10_OPPER_PREFLIGHT_JSON="
CHAT_MESSAGE_TOKEN_OVERHEAD = 16


class BatchError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class InputPin(FrozenModel):
    path: str = Field(min_length=1, max_length=300)
    sha256: str

    @field_validator("sha256")
    @classmethod
    def _hash(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("Expected lowercase SHA-256")
        return value


class ProviderConfig(FrozenModel):
    name: Literal["opper"]
    endpoint: Literal["https://api.opper.ai/v3/compat"]
    model: Literal["openai/gpt-4o-2024-08-06"]
    api_key_env: Literal["OPPER_API_KEY"]


class BatchLimits(FrozenModel):
    max_calls_per_paper: Literal[3]
    max_output_tokens_per_call: Literal[8192]
    max_prompt_bytes_per_call: Literal[130000]
    max_input_tokens_per_call: Literal[119000]
    timeout_seconds: Literal[240]
    preflight_timeout_seconds: int = Field(ge=30, le=600, strict=True)
    maximum_papers: Literal[30]
    maximum_total_calls: Literal[90]


class BatchPaper(FrozenModel):
    ordinal: int = Field(ge=1, le=30, strict=True)
    arxiv_id: str
    source: InputPin

    @field_validator("arxiv_id")
    @classmethod
    def _arxiv(cls, value: str) -> str:
        if not re.fullmatch(r"\d{4}\.\d{5}", value):
            raise ValueError("Expected a modern arXiv identifier")
        return value

    @model_validator(mode="after")
    def _source_matches(self):
        if self.source.path != f"data/raw/{self.arxiv_id}/paper.pdf":
            raise ValueError("Source path must match the paper identifier")
        return self


class EvaluationBatchConfig(FrozenModel):
    version: Literal["opper-evaluation-batch-1"]
    name: Literal["historical-evaluation-30"]
    corpus: InputPin
    split: Literal["evaluation"]
    provider: ProviderConfig
    limits: BatchLimits
    papers: tuple[BatchPaper, ...]

    @model_validator(mode="after")
    def _papers(self):
        if len(self.papers) != 30:
            raise ValueError("Expected exactly 30 papers")
        if tuple(p.ordinal for p in self.papers) != tuple(range(1, 31)):
            raise ValueError("Paper ordinals must be exactly 1..30")
        if len({p.arxiv_id for p in self.papers}) != 30:
            raise ValueError("Paper identifiers must be distinct")
        if self.limits.maximum_total_calls != len(self.papers) * self.limits.max_calls_per_paper:
            raise ValueError("Total call budget does not match paper budgets")
        return self


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _batch_id(value: str) -> str:
    if not re.fullmatch(r"m10-e30-[a-z0-9][a-z0-9_-]{0,9}", value):
        raise argparse.ArgumentTypeError(
            "Use an m10-e30- batch ID of at most 18 lowercase characters.")
    return value


def _prep_id(batch_id: str) -> str:
    return batch_id + "-prep"


def _paper_run_id(batch_id: str, paper: BatchPaper) -> str:
    return f"{batch_id}-p{paper.ordinal:02d}-{paper.arxiv_id.replace('.', '')}"


def _strict_json(path: Path, *, maximum: int = 4_000_000) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > maximum:
        raise BatchError("artifact_unavailable")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_bytes(), object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")),
        )
    except (OSError, ValueError):
        raise BatchError("artifact_invalid") from None
    if not isinstance(value, dict):
        raise BatchError("artifact_invalid")
    return value


def _load_config(root: Path, relative: str = DEFAULT_CONFIG) -> tuple[EvaluationBatchConfig, str]:
    path = input_file(root, relative, allowed_roots=("configs/agent",))
    raw = path.read_bytes()
    try:
        config = EvaluationBatchConfig.model_validate(_strict_json(path))
    except (ValueError, BatchError):
        raise BatchError("batch_config_invalid") from None
    return config, sha256(raw)


def _prep(root: Path, batch_id: str) -> RunDirectory:
    run = RunDirectory(root, _prep_id(batch_id))
    if not run.path.is_dir():
        raise BatchError("preparation_not_found")
    return run


def _load_prepared(root: Path, batch_id: str) -> tuple[RunDirectory, EvaluationBatchConfig]:
    prep = _prep(root, batch_id)
    try:
        config = EvaluationBatchConfig.model_validate(
            _strict_json(prep.path / "batch-config.json"))
        protocol = _strict_json(prep.path / "protocol.json")
    except ValueError:
        raise BatchError("prepared_scope_invalid") from None
    if (protocol.get("version") != VERSION
            or protocol.get("batch_id") != batch_id
            or protocol.get("launcher_sha256") != sha256(Path(__file__).read_bytes())):
        raise BatchError("prepared_implementation_changed")
    return prep, config


def _corpus_evaluation_ids(root: Path, pin: InputPin) -> tuple[str, ...]:
    path = input_file(root, pin.path, allowed_roots=("data",))
    raw = path.read_bytes()
    if sha256(raw) != pin.sha256:
        raise BatchError("corpus_hash_mismatch")
    try:
        payload = json.loads(raw)
        papers = payload["papers"]
        rows = tuple(
            row["arxiv_id"] for row in papers
            if isinstance(row, dict) and row.get("split") == "evaluation"
        )
    except (KeyError, TypeError, ValueError):
        raise BatchError("corpus_manifest_invalid") from None
    if len(rows) != 30 or len(set(rows)) != 30:
        raise BatchError("evaluation_split_invalid")
    return rows


def _verify_sources(root: Path, config: EvaluationBatchConfig) -> list[dict[str, Any]]:
    expected = _corpus_evaluation_ids(root, config.corpus)
    configured = tuple(p.arxiv_id for p in config.papers)
    if configured != expected:
        raise BatchError("evaluation_membership_changed")
    rows = []
    for paper in config.papers:
        path = input_file(root, paper.source.path, allowed_roots=("data/raw",))
        raw_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if raw_hash != paper.source.sha256:
            raise BatchError("source_hash_mismatch")
        rows.append({
            "ordinal": paper.ordinal, "arxiv_id": paper.arxiv_id,
            "path": paper.source.path, "sha256": raw_hash,
            "bytes": path.stat().st_size,
            "run_id": _paper_run_id("BATCH_ID", paper).replace("BATCH_ID", "{batch_id}"),
        })
    return rows


def prepare(root: Path | str, batch_id: str,
            config_relative: str = DEFAULT_CONFIG) -> dict[str, Any]:
    root = workspace_root(root)
    config, config_sha256 = _load_config(root, config_relative)
    sources = _verify_sources(root, config)
    prep = RunDirectory.create(root, _prep_id(batch_id))
    hard_input = (config.limits.maximum_total_calls
                  * config.limits.max_input_tokens_per_call)
    hard_output = (config.limits.maximum_total_calls
                   * config.limits.max_output_tokens_per_call)
    protocol = {
        "version": VERSION, "batch_id": batch_id, "prepared_at": _utc_now(),
        "launcher_sha256": sha256(Path(__file__).read_bytes()),
        "config_sha256": config_sha256,
        "question": "Can the guarded workflow produce reviewable outputs for all 30 evaluation papers through Opper?",
        "split_manifest_only": config.corpus.model_dump(mode="json"),
        "gold_access": False, "evaluator": None, "comparison": None,
        "provider": config.provider.model_dump(mode="json"),
        "policy": {"controller":"issue_aware", "automatic_patches":False,
                   "raw_retention":"hash_only", "workers":1},
        "budgets": {
            **config.limits.model_dump(mode="json"),
            "conservative_batch_input_token_equivalents": hard_input,
            "reserved_batch_output_tokens": hard_output,
            "shared_rolling_request_limit": 10,
            "minimum_slots_before_paper": config.limits.max_calls_per_paper,
        },
        "empirical_status": "not_run",
    }
    prep.write_json("batch-config.json", config.model_dump(mode="json"))
    prep.write_json("source-manifest.json", {
        "version":"opper-evaluation-sources-1", "count":len(sources),
        "papers":[row | {"run_id":_paper_run_id(batch_id, config.papers[index])}
                  for index, row in enumerate(sources)],
    })
    prep.write_json("protocol.json", protocol)
    return {
        "version":VERSION, "status":"prepared", "batch_id":batch_id,
        "paper_count":len(config.papers),
        "prep_relative":f"runs/agent/{prep.run_id}",
        "conservative_batch_input_token_equivalents":hard_input,
        "reserved_batch_output_tokens":hard_output,
        "network_requests":0, "model_requests":0, "gold_access":False,
    }


def _preflight_filename(paper: BatchPaper) -> str:
    return f"preflight-{paper.ordinal:02d}-{paper.arxiv_id.replace('.', '')}.json"


def _preflight_one(root: Path, config: EvaluationBatchConfig,
                   paper: BatchPaper) -> dict[str, Any]:
    source = input_file(root, paper.source.path, allowed_roots=("data/raw",))
    if sha256(source.read_bytes()) != paper.source.sha256:
        raise BatchError("source_hash_mismatch")
    prepared = prepare_sources(root, [paper.source.path])[0]
    if prepared.bundle is None:
        return {
            "version":"opper-paper-preflight-1", "ordinal":paper.ordinal,
            "arxiv_id":paper.arxiv_id, "status":"unavailable",
            "error_code":"source_preparation_failed", "eligible":False,
        }
    profile = load_profile(root, ("generic.json", "photon_and_neutron.json"))
    prompt = render_prompt(prepared.bundle, profile, "evidence")
    prompt_bytes = len(prompt.encode("utf-8"))
    exact_tokens = len(get_encoding("o200k_base").encode(prompt))
    return {
        "version":"opper-paper-preflight-1", "ordinal":paper.ordinal,
        "arxiv_id":paper.arxiv_id, "status":"ready",
        "error_code":None,
        "eligible":(
            prompt_bytes <= config.limits.max_prompt_bytes_per_call
            and exact_tokens + CHAT_MESSAGE_TOKEN_OVERHEAD
            <= config.limits.max_input_tokens_per_call),
        "source_sha256":prepared.bundle.source_sha256,
        "first_prompt_characters_exact":len(prompt),
        "first_prompt_utf8_bytes_exact":prompt_bytes,
        "first_prompt_sha256":sha256(prompt.encode("utf-8")),
        "first_input_tokens_chars_div_4_estimate":math.ceil(len(prompt) / 4),
        "first_input_tokens_conservative_byte_bound_plus_overhead":prompt_bytes + 128,
        "first_input_tokens_o200k_exact":exact_tokens,
        "warnings":[warning.model_dump(mode="json") for warning in prepared.bundle.warnings],
    }


def _worker_command(root: Path, batch_id: str, ordinal: int) -> list[str]:
    return [
        sys.executable, "-m", "meta_data_extraction.agent.opper_evaluation_batch",
        "_preflight-worker", "--root", str(root), "--batch-id", batch_id,
        "--ordinal", str(ordinal),
    ]


def _run_preflight_worker(root: Path, batch_id: str, paper: BatchPaper,
                          timeout_seconds: int) -> dict[str, Any]:
    environment = dict(os.environ)
    environment.pop("OPPER_API_KEY", None)
    environment.pop("CHATAI_API_KEY", None)
    try:
        completed = subprocess.run(
            _worker_command(root, batch_id, paper.ordinal), cwd=root,
            env=environment, capture_output=True, timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "version":"opper-paper-preflight-1", "ordinal":paper.ordinal,
            "arxiv_id":paper.arxiv_id, "status":"unavailable",
            "error_code":"preflight_timeout", "eligible":False,
        }
    marker = WORKER_MARKER.encode("ascii")
    lines = [line for line in completed.stdout.splitlines() if line.startswith(marker)]
    if completed.returncode != 0 or len(lines) != 1:
        return {
            "version":"opper-paper-preflight-1", "ordinal":paper.ordinal,
            "arxiv_id":paper.arxiv_id, "status":"unavailable",
            "error_code":"preflight_worker_failed", "eligible":False,
        }
    try:
        payload = json.loads(lines[0][len(marker):])
    except ValueError:
        raise BatchError("preflight_worker_invalid") from None
    if (not isinstance(payload, dict) or payload.get("ordinal") != paper.ordinal
            or payload.get("arxiv_id") != paper.arxiv_id):
        raise BatchError("preflight_worker_invalid")
    return payload


def preflight(root: Path | str, batch_id: str) -> dict[str, Any]:
    root = workspace_root(root)
    prep, config = _load_prepared(root, batch_id)
    summary_path = prep.path / "preflight-summary.json"
    if summary_path.is_file():
        return _strict_json(summary_path)
    rows = []
    for paper in config.papers:
        path = prep.path / _preflight_filename(paper)
        if path.is_file():
            row = _strict_json(path)
        else:
            row = _run_preflight_worker(
                root, batch_id, paper, config.limits.preflight_timeout_seconds)
            prep.write_json(_preflight_filename(paper), row)
        rows.append(row)
    ready = [row for row in rows if row.get("status") == "ready"]
    eligible = [row for row in rows if row.get("eligible") is True]
    result = {
        "version":"opper-evaluation-preflight-1", "batch_id":batch_id,
        "paper_count":len(rows), "ready_count":len(ready),
        "eligible_count":len(eligible),
        "batch_eligible":len(eligible) == len(config.papers),
        "failed":[{"ordinal":row.get("ordinal"), "arxiv_id":row.get("arxiv_id"),
                   "error_code":row.get("error_code") or (
                       "prompt_size_limit"
                       if int(row.get("first_prompt_utf8_bytes_exact", 0))
                       > config.limits.max_prompt_bytes_per_call
                       else "input_token_limit")}
                  for row in rows if row.get("eligible") is not True],
        "first_prompt_utf8_bytes_total":sum(
            int(row.get("first_prompt_utf8_bytes_exact", 0)) for row in ready),
        "first_input_tokens_chars_div_4_estimate_total":sum(
            int(row.get("first_input_tokens_chars_div_4_estimate", 0)) for row in ready),
        "first_input_tokens_o200k_exact_total":sum(
            int(row.get("first_input_tokens_o200k_exact", 0)) for row in ready),
        "maximum_first_input_tokens_o200k_exact":max(
            (int(row.get("first_input_tokens_o200k_exact", 0)) for row in ready),
            default=0),
        "network_requests":0, "model_requests":0, "credential_read":False,
    }
    prep.write_json("preflight-summary.json", result)
    return result


def check_catalog(root: Path | str, batch_id: str, api_key: str) -> dict[str, Any]:
    root = workspace_root(root)
    if not api_key:
        raise BatchError("missing_credentials")
    prep, config = _load_prepared(root, batch_id)
    prep.write_json("catalog-attempt.json", {
        "version":"opper-batch-catalog-attempt-1", "attempted_at":_utc_now(),
        "method":"GET", "url":ENDPOINT + "/models", "prompt_sent":False,
        "document_sent":False, "retries":0, "follow_redirects":False,
    })
    try:
        status_code, headers, raw = _models_request(api_key)
        if len(raw) > CATALOG_MAX_BYTES:
            raise BatchError("catalog_response_too_large")
        metadata = _model_metadata(raw) if status_code == 200 else None
        error = (None if status_code == 200 and metadata is not None else
                 "model_not_found" if status_code == 200 else "catalog_http_error")
        result = {
            "version":"opper-batch-catalog-result-1", "completed_at":_utc_now(),
            "status_code":status_code, "model":MODEL,
            "model_found":metadata is not None, "model_metadata":metadata,
            "numeric_limit_headers":_numeric_headers(headers),
            "prompt_sent":False, "document_sent":False, "model_requests":0,
            "error_code":error,
        }
    except BatchError as error:
        result = {
            "version":"opper-batch-catalog-result-1", "completed_at":_utc_now(),
            "status_code":None, "model":MODEL, "model_found":False,
            "model_metadata":None, "numeric_limit_headers":{},
            "prompt_sent":False, "document_sent":False, "model_requests":0,
            "error_code":error.code,
        }
    except Exception:
        result = {
            "version":"opper-batch-catalog-result-1", "completed_at":_utc_now(),
            "status_code":None, "model":MODEL, "model_found":False,
            "model_metadata":None, "numeric_limit_headers":{},
            "prompt_sent":False, "document_sent":False, "model_requests":0,
            "error_code":"catalog_request_failed",
        }
    prep.write_json("catalog-result.json", result)
    if result["error_code"] is not None:
        raise BatchError(result["error_code"])
    return result


def _catalog_and_cost(prep: RunDirectory, config: EvaluationBatchConfig) -> tuple[dict, Decimal]:
    try:
        catalog = _strict_json(prep.path / "catalog-result.json")
    except BatchError:
        raise BatchError("successful_catalog_check_required") from None
    metadata = catalog.get("model_metadata")
    if (catalog.get("error_code") is not None or catalog.get("status_code") != 200
            or catalog.get("model") != MODEL or catalog.get("model_found") is not True
            or not isinstance(metadata, dict)):
        raise BatchError("successful_catalog_check_required")
    context = metadata.get("context_length")
    output = metadata.get("max_output_tokens")
    if type(context) is not int or type(output) is not int:
        raise BatchError("model_limits_unknown")
    per_request_input = config.limits.max_input_tokens_per_call
    if context < per_request_input + config.limits.max_output_tokens_per_call:
        raise BatchError("model_context_too_small")
    if output < config.limits.max_output_tokens_per_call:
        raise BatchError("model_output_limit_too_small")
    try:
        prompt_price = Decimal(metadata["pricing_prompt"])
        completion_price = Decimal(metadata["pricing_completion"])
    except (InvalidOperation, KeyError, TypeError):
        raise BatchError("catalog_pricing_unknown") from None
    ceiling = (
        Decimal(config.limits.maximum_total_calls * per_request_input) * prompt_price
        + Decimal(config.limits.maximum_total_calls
                  * config.limits.max_output_tokens_per_call) * completion_price
    )
    return catalog, ceiling


def _publication_spec(batch_id: str, config: EvaluationBatchConfig,
                      paper: BatchPaper) -> PublicationSpec:
    return PublicationSpec(
        run_id=_paper_run_id(batch_id, paper), mode="live", split="evaluation",
        source=paper.source.model_dump(mode="json"), policy="issue_aware",
        max_calls=config.limits.max_calls_per_paper, max_targets=12,
        evidence_chars=24_000, retrieval_strategy="lexical_v1",
        allow_automatic_patches=False, generate_proposal=True,
        raw_retention="hash_only", unknown_usage="allow_bounded_calls",
        workflow_timeout_seconds=300, mock_responses=(),
        live={
            "provider":"opper", "endpoint":config.provider.endpoint,
            "model":config.provider.model, "api_key_env":config.provider.api_key_env,
            "max_output_tokens":config.limits.max_output_tokens_per_call,
            "max_prompt_bytes":config.limits.max_prompt_bytes_per_call,
            "max_input_tokens":config.limits.max_input_tokens_per_call,
            "timeout_seconds":config.limits.timeout_seconds,
        },
        panet=None, nexus=None,
    )


def _completed(root: Path, run_id: str) -> dict[str, Any] | None:
    path = checked_path(root, f"runs/agent/{run_id}/result.json", allowed_roots=("runs/agent",))
    if not path.is_file():
        return None
    return read_publication(root, run_id)


def _status(root: Path, batch_id: str, prep: RunDirectory,
            config: EvaluationBatchConfig) -> dict[str, Any]:
    rows = []
    for paper in config.papers:
        run_id = _paper_run_id(batch_id, paper)
        result = _completed(root, run_id)
        if result is None:
            state = "incomplete" if checked_path(
                root, f"runs/agent/{run_id}", allowed_roots=("runs/agent",)).is_dir() else "pending"
            rows.append({"ordinal":paper.ordinal, "arxiv_id":paper.arxiv_id,
                         "run_id":run_id, "status":state})
        else:
            rows.append({
                "ordinal":paper.ordinal, "arxiv_id":paper.arxiv_id,
                "run_id":run_id, "status":result.get("status"),
                "model_calls":result.get("total_model_calls"),
                "live_requests":result.get("total_live_requests"),
                "known_input_tokens":result.get("known_input_tokens"),
                "known_output_tokens":result.get("known_output_tokens"),
                "usage_unknown":result.get("usage_unknown"),
            })
    completed = [row for row in rows if row["status"] not in {"pending", "incomplete"}]
    counts = Counter(row["status"] for row in rows)
    known_input = sum(int(row.get("known_input_tokens") or 0) for row in completed)
    known_output = sum(int(row.get("known_output_tokens") or 0) for row in completed)
    usage_unknown = any(row.get("usage_unknown") is True for row in completed)
    cost_estimate = None
    try:
        catalog, ceiling = _catalog_and_cost(prep, config)
        metadata = catalog["model_metadata"]
        if not usage_unknown:
            cost_estimate = str(
                Decimal(known_input) * Decimal(metadata["pricing_prompt"])
                + Decimal(known_output) * Decimal(metadata["pricing_completion"])
            )
    except BatchError:
        ceiling = None
    return {
        "version":VERSION, "batch_id":batch_id, "paper_count":len(rows),
        "status_counts":dict(sorted(counts.items())), "papers":rows,
        "completed_papers":len(completed),
        "total_model_calls":sum(int(row.get("model_calls") or 0) for row in completed),
        "total_live_requests":sum(int(row.get("live_requests") or 0) for row in completed),
        "known_input_tokens":known_input, "known_output_tokens":known_output,
        "usage_unknown":usage_unknown,
        "catalog_price_usage_estimate_usd":cost_estimate,
        "catalog_price_hard_ceiling_usd":str(ceiling) if ceiling is not None else None,
        "cost_note":"catalog multiplication is a planning estimate, not a billing receipt",
    }


def status(root: Path | str, batch_id: str) -> dict[str, Any]:
    root = workspace_root(root)
    prep, config = _load_prepared(root, batch_id)
    return _status(root, batch_id, prep, config)


def run_available(root: Path | str, batch_id: str, *, confirm_live: bool,
                  retention_reviewed: bool, load_dotenv_file: bool,
                  accepted_cost_ceiling_usd: Decimal | None,
                  max_papers: int = 1) -> dict[str, Any]:
    if confirm_live is not True:
        raise BatchError("live_confirmation_required")
    if retention_reviewed is not True:
        raise BatchError("retention_review_required")
    if type(max_papers) is not int or not 1 <= max_papers <= 30:
        raise BatchError("max_papers_invalid")
    root = workspace_root(root)
    prep, config = _load_prepared(root, batch_id)
    preflight_summary = _strict_json(prep.path / "preflight-summary.json")
    if (preflight_summary.get("batch_id") != batch_id
            or preflight_summary.get("batch_eligible") is not True
            or preflight_summary.get("eligible_count") != len(config.papers)):
        raise BatchError("complete_eligible_preflight_required")
    _, cost_ceiling = _catalog_and_cost(prep, config)
    if (accepted_cost_ceiling_usd is None
            or accepted_cost_ceiling_usd < cost_ceiling):
        raise BatchError("cost_ceiling_acknowledgement_required")
    if load_dotenv_file:
        load_dotenv(checked_path(root, ".env", allowed_roots=(".env",)), override=False)
    if not os.environ.get(config.provider.api_key_env):
        raise BatchError("missing_credentials")
    ledger = OpperRequestLedger(root)
    completed_now = []
    stop_reason = None
    for paper in config.papers:
        run_id = _paper_run_id(batch_id, paper)
        existing = _completed(root, run_id)
        if existing is not None:
            continue
        capacity = ledger.capacity()
        if capacity.error_code:
            stop_reason = capacity.error_code
            break
        if capacity.remaining_attempts < config.limits.max_calls_per_paper:
            stop_reason = "waiting_for_rolling_request_capacity"
            break
        result = run_publication(
            root, _publication_spec(batch_id, config, paper),
            configured_provider_endpoint=config.provider.endpoint,
            load_dotenv_file=False,
        )
        completed_now.append({
            "ordinal":paper.ordinal, "arxiv_id":paper.arxiv_id,
            "run_id":run_id, "status":result.get("status"),
            "new_model_calls":result.get("new_model_calls"),
            "new_live_requests":result.get("new_live_requests"),
            "known_input_tokens":result.get("known_input_tokens"),
            "known_output_tokens":result.get("known_output_tokens"),
            "usage_unknown":result.get("usage_unknown"),
        })
        if len(completed_now) >= max_papers:
            stop_reason = "invocation_paper_limit"
            break
    aggregate = _status(root, batch_id, prep, config)
    return {
        "version":VERSION, "batch_id":batch_id,
        "completed_this_invocation":completed_now,
        "stop_reason":stop_reason or "batch_complete",
        "accepted_cost_ceiling_usd":str(accepted_cost_ceiling_usd),
        "aggregate":aggregate,
    }


def _parse_decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        raise argparse.ArgumentTypeError("Expected a decimal USD amount") from None
    if not parsed.is_finite() or parsed < 0:
        raise argparse.ArgumentTypeError("Expected a finite non-negative USD amount")
    return parsed


def _worker(root: str, batch_id: str, ordinal: int) -> int:
    try:
        root_path = workspace_root(root)
        _, config = _load_prepared(root_path, batch_id)
        if not 1 <= ordinal <= len(config.papers):
            raise BatchError("paper_ordinal_invalid")
        result = _preflight_one(root_path, config, config.papers[ordinal - 1])
        print(WORKER_MARKER + json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception:
        return 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "preflight", "catalog", "run-available", "status"):
        command = commands.add_parser(name)
        command.add_argument("--root", default=str(Path(__file__).resolve().parents[3]))
        command.add_argument("--batch-id", required=True, type=_batch_id)
        if name == "prepare":
            command.add_argument("--config", default=DEFAULT_CONFIG)
        if name in {"catalog", "run-available"}:
            command.add_argument("--load-dotenv", action="store_true")
        if name == "run-available":
            command.add_argument("--execute-live", action="store_true")
            command.add_argument("--retention-reviewed", action="store_true")
            command.add_argument("--accept-cost-ceiling-usd", type=_parse_decimal)
            command.add_argument("--max-papers", type=int, default=1)
    worker = commands.add_parser("_preflight-worker", help=argparse.SUPPRESS)
    worker.add_argument("--root", required=True)
    worker.add_argument("--batch-id", required=True, type=_batch_id)
    worker.add_argument("--ordinal", required=True, type=int)
    args = parser.parse_args(argv)
    if args.command == "_preflight-worker":
        return _worker(args.root, args.batch_id, args.ordinal)
    try:
        if args.command == "prepare":
            result = prepare(args.root, args.batch_id, args.config)
        elif args.command == "preflight":
            result = preflight(args.root, args.batch_id)
        elif args.command == "catalog":
            root = workspace_root(args.root)
            if args.load_dotenv:
                load_dotenv(checked_path(root, ".env", allowed_roots=(".env",)), override=False)
            result = check_catalog(root, args.batch_id, os.environ.get("OPPER_API_KEY", ""))
        elif args.command == "run-available":
            result = run_available(
                args.root, args.batch_id, confirm_live=args.execute_live,
                retention_reviewed=args.retention_reviewed,
                load_dotenv_file=args.load_dotenv,
                accepted_cost_ceiling_usd=args.accept_cost_ceiling_usd,
                max_papers=args.max_papers,
            )
        else:
            result = status(args.root, args.batch_id)
    except BatchError as error:
        parser.exit(2, f"Opper evaluation batch stopped ({error.code}).\n")
    except FileExistsError:
        parser.exit(2, "Opper evaluation batch stopped (artifact_exists_use_fresh_batch_id).\n")
    except Exception:
        parser.exit(2, "Opper evaluation batch stopped (operation_failed).\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
