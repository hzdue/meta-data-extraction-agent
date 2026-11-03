"""Prepare and explicitly run one bounded Opper publication pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any
import uuid

import httpx
from dotenv import load_dotenv

from .evidence import sha256
from .extraction import render_prompt
from .paths import RunDirectory, checked_path, input_file, workspace_root
from .profiles import load_profile
from .providers import OpperRequestLedger, OpperRunSpec
from .publication_run import _child as _publication_child
from .publication_run import read_publication, run_publication
from .publication_types import PublicationOpperSettings, PublicationSpec, load_publication_spec
from .sources import prepare_sources


VERSION = "m10-opper-one-paper-1"
ENDPOINT = "https://api.opper.ai/v3/compat"
MODEL = "openai/gpt-4o-2024-08-06"
SOURCE = "data/raw/1812.03956/paper.pdf"
SOURCE_SHA256 = "158984e801673f2ce019d9fd793684183bbc288e1b27856039aa37b6c9845bce"
MAX_CALLS = 3
MAX_OUTPUT_TOKENS = 8192
MAX_PROMPT_BYTES = 100_000
TIMEOUT_SECONDS = 240
CATALOG_MAX_BYTES = 8_000_000
MODEL_STAGE_NAMES = ("extract", "check", "proposal", "provider")


class OpperPilotError(ValueError):
    """Safe operator-facing failure code; never contains credentials or source text."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id(value: str) -> str:
    if not re.fullmatch(r"m10-[a-z0-9][a-z0-9_-]{0,27}", value):
        raise argparse.ArgumentTypeError("Use an m10- run ID of at most 32 lowercase characters.")
    return value


def _prep_id(run_id: str) -> str:
    return run_id + "-opper-prep"


def _strict_json(path: Path, *, maximum: int = 2_000_000) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > maximum:
        raise OpperPilotError("artifact_unavailable")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
    except (ValueError, OSError):
        raise OpperPilotError("artifact_invalid") from None
    if not isinstance(value, dict):
        raise OpperPilotError("artifact_invalid")
    return value


def _prep(root: Path, run_id: str) -> RunDirectory:
    run = RunDirectory(root, _prep_id(run_id))
    if not run.path.is_dir():
        raise OpperPilotError("preparation_not_found")
    return run


def _fixed_spec(source_sha256: str, run_id: str) -> PublicationSpec:
    return PublicationSpec(
        run_id=run_id, mode="live", split="calibration",
        source={"path": SOURCE, "sha256": source_sha256},
        policy="issue_aware", max_calls=MAX_CALLS, max_targets=12,
        evidence_chars=24_000, retrieval_strategy="lexical_v1",
        allow_automatic_patches=False, generate_proposal=True,
        raw_retention="hash_only", unknown_usage="allow_bounded_calls",
        workflow_timeout_seconds=300, mock_responses=(),
        live={"provider":"opper", "endpoint":ENDPOINT, "model":MODEL,
              "api_key_env":"OPPER_API_KEY", "max_output_tokens":MAX_OUTPUT_TOKENS,
              "max_prompt_bytes":MAX_PROMPT_BYTES, "timeout_seconds":TIMEOUT_SECONDS},
        panet=None, nexus=None,
    )


def _inputs(root: Path, spec: PublicationSpec):
    source = input_file(root, spec.source.path, allowed_roots=("data/raw",))
    raw_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    if raw_sha256 != spec.source.sha256 or raw_sha256 != SOURCE_SHA256:
        raise OpperPilotError("source_hash_mismatch")
    prepared = prepare_sources(root, [spec.source.path])[0]
    if prepared.bundle is None:
        raise OpperPilotError("source_preparation_failed")
    profile = load_profile(root, spec.schemas)
    prompt = render_prompt(prepared.bundle, profile, "evidence")
    return prepared.bundle, profile, prompt


def _token_plan(prompt: str, spec: PublicationSpec) -> dict[str, Any]:
    prompt_bytes = len(prompt.encode("utf-8"))
    prompt_chars = len(prompt)
    message_overhead_allowance = 128
    return {
        "version": "opper-token-plan-1",
        "measurement_status": "preflight_estimates_not_tokenizer_measurements",
        "first_prompt_characters_exact": prompt_chars,
        "first_prompt_utf8_bytes_exact": prompt_bytes,
        "first_prompt_sha256": sha256(prompt.encode("utf-8")),
        "first_input_tokens_chars_div_4_estimate": math.ceil(prompt_chars / 4),
        "first_input_tokens_conservative_byte_bound_plus_overhead": prompt_bytes + message_overhead_allowance,
        "per_request_prompt_bytes_limit": spec.live.max_prompt_bytes,
        "per_request_input_tokens_conservative_bound_plus_overhead": spec.live.max_prompt_bytes + message_overhead_allowance,
        "maximum_calls": spec.max_calls,
        "whole_run_prompt_bytes_limit": spec.max_calls * spec.live.max_prompt_bytes,
        "whole_run_input_tokens_conservative_bound_plus_overhead": spec.max_calls * (spec.live.max_prompt_bytes + message_overhead_allowance),
        "per_request_output_tokens_reserved": spec.live.max_output_tokens,
        "whole_run_output_tokens_reserved": spec.max_calls * spec.live.max_output_tokens,
        "authoritative_after_run": "provider usage input_tokens/output_tokens; unknown remains unknown",
    }


def _first_scope(spec: PublicationSpec, bundle, profile, prompt: str) -> OpperRunSpec:
    if type(spec.live) is not PublicationOpperSettings:
        raise OpperPilotError("opper_scope_required")
    return OpperRunSpec(
        **spec.live.model_dump(mode="json"),
        sources=({"path":bundle.source_id, "sha256":bundle.source_sha256},),
        split=spec.split, profile_sha256=profile.profile_id.removeprefix("profile:"),
        approved_prompt_sha256=(sha256(prompt.encode("utf-8")),),
        max_calls=1, max_total_output_tokens=spec.live.max_output_tokens,
    )


def _verify_fixed(spec: PublicationSpec, run_id: str) -> None:
    expected = _fixed_spec(SOURCE_SHA256, run_id)
    if spec != expected:
        raise OpperPilotError("prepared_scope_changed")


def _history_override_run_ids(spec: PublicationSpec) -> tuple[str, ...]:
    """Parent readiness plus every deterministic child that can own a provider."""
    return (spec.run_id, *(_publication_child(spec, stage) for stage in MODEL_STAGE_NAMES))


def prepare(root: Path | str, run_id: str) -> dict[str, Any]:
    root = workspace_root(root)
    source = input_file(root, SOURCE, allowed_roots=("data/raw",))
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha256 != SOURCE_SHA256:
        raise OpperPilotError("source_hash_mismatch")
    spec = _fixed_spec(source_sha256, run_id)
    bundle, profile, prompt = _inputs(root, spec)
    plan = _token_plan(prompt, spec)
    scope = _first_scope(spec, bundle, profile, prompt)
    ledger = OpperRequestLedger(root)
    readiness = ledger.reserve(run_id, scope, commit_attempt=False)
    prep = RunDirectory.create(root, _prep_id(run_id))
    protocol = {
        "version": VERSION, "prepared_at": _utc_now(),
        "launcher_sha256": sha256(Path(__file__).read_bytes()),
        "question": "Can the guarded issue-aware workflow produce an inspectable one-paper result through Opper?",
        "comparison": "engineering smoke test only; no comparative performance claim",
        "model": MODEL, "endpoint": ENDPOINT,
        "source": {"path":SOURCE, "sha256":source_sha256, "split":"calibration", "publication_count":1},
        "budgets": {"max_calls":MAX_CALLS, "max_output_tokens_per_call":MAX_OUTPUT_TOKENS,
                    "max_prompt_bytes_per_call":MAX_PROMPT_BYTES, "timeout_seconds":TIMEOUT_SECONDS,
                    "automatic_retries":0},
        "policy": {"controller":"issue_aware", "automatic_patches":False,
                   "raw_retention":"hash_only", "external_tracing":"must_be_reviewed_before_run"},
        "primary_outcome": "structural completion with evidence-linked diagnostics and review state",
        "evaluator": None, "empirical_status": "not_run",
        "catalog_check_required": True,
    }
    prep.write_json("publication-spec.json", spec.model_dump(mode="json"))
    prep.write_json("protocol.json", protocol)
    prep.write_json("token-plan.json", plan)
    prep.write_json("readiness.json", {
        "version":"opper-readiness-1", "checked_at":_utc_now(),
        "error_code":readiness.error_code, "request_reserved":False,
        "credential_read":False, "network_requests":0, "model_requests":0,
    })
    return {
        "version": VERSION, "status":"prepared", "run_id":run_id,
        "prep_relative":f"runs/agent/{prep.run_id}",
        "spec_relative":f"runs/agent/{prep.run_id}/publication-spec.json",
        "ledger_readiness":readiness.error_code or "ready",
        "token_plan":plan, "network_requests":0, "model_requests":0,
    }


def _models_request(api_key: str) -> tuple[int, dict[str, str], bytes]:
    headers = {"Authorization": "Bearer " + api_key, "Accept": "application/json"}
    with httpx.Client(timeout=15.0, follow_redirects=False) as client:
        with client.stream("GET", ENDPOINT + "/models", headers=headers) as response:
            chunks = []
            size = 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > CATALOG_MAX_BYTES:
                    raise OpperPilotError("catalog_response_too_large")
                chunks.append(chunk)
            return response.status_code, dict(response.headers), b"".join(chunks)


def _numeric_headers(headers: dict[str, str]) -> dict[str, int | float]:
    retained = {}
    for name, raw in headers.items():
        lower = name.lower()
        if not (lower.startswith("x-ratelimit-") or lower.startswith("ratelimit-") or lower == "retry-after"):
            continue
        if re.fullmatch(r"[0-9]+", raw):
            retained[lower] = int(raw)
        elif re.fullmatch(r"[0-9]+\.[0-9]+", raw):
            retained[lower] = float(raw)
    return retained


def _model_metadata(raw: bytes) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw)
        rows = payload["data"]
    except (ValueError, KeyError, TypeError):
        raise OpperPilotError("catalog_response_invalid") from None
    if not isinstance(rows, list):
        raise OpperPilotError("catalog_response_invalid")
    matches = [row for row in rows if isinstance(row, dict) and row.get("id") == MODEL]
    if len(matches) != 1:
        return None
    row = matches[0]
    opper = row.get("opper") if isinstance(row.get("opper"), dict) else {}
    pricing = row.get("pricing") if isinstance(row.get("pricing"), dict) else {}

    def integer(value):
        return value if type(value) is int and value >= 0 else None

    def numeric_text(value):
        return value if isinstance(value, str) and re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value) else None

    return {
        "id": MODEL,
        "context_length": integer(row.get("context_length")),
        "max_output_tokens": integer(opper.get("max_output_tokens")),
        "region": opper.get("region") if isinstance(opper.get("region"), str) else None,
        "gdpr_residency": opper.get("gdpr_residency") if isinstance(opper.get("gdpr_residency"), str) else None,
        "pricing_prompt": numeric_text(pricing.get("prompt")),
        "pricing_completion": numeric_text(pricing.get("completion")),
        "pricing_units": "provider_metadata_not_interpreted_by_launcher",
    }


def check_catalog(root: Path | str, run_id: str, api_key: str) -> dict[str, Any]:
    root = workspace_root(root)
    if not api_key:
        raise OpperPilotError("missing_credentials")
    prep = _prep(root, run_id)
    spec = load_publication_spec(root, f"runs/agent/{prep.run_id}/publication-spec.json")
    _verify_fixed(spec, run_id)
    prep.write_json("catalog-attempt.json", {
        "version":"opper-catalog-attempt-1", "attempt_id":uuid.uuid4().hex,
        "attempted_at":_utc_now(), "method":"GET", "url":ENDPOINT + "/models",
        "prompt_sent":False, "document_sent":False, "retries":0, "follow_redirects":False,
    })
    try:
        status, headers, raw = _models_request(api_key)
        metadata = _model_metadata(raw) if status == 200 else None
        result = {
            "version":"opper-catalog-result-1", "completed_at":_utc_now(),
            "status_code":status, "model":MODEL, "model_found":metadata is not None,
            "model_metadata":metadata, "numeric_limit_headers":_numeric_headers(headers),
            "prompt_sent":False, "document_sent":False, "model_requests":0,
            "error_code":None if status == 200 and metadata is not None else
                         "model_not_found" if status == 200 else "catalog_http_error",
        }
    except OpperPilotError as error:
        result = {"version":"opper-catalog-result-1", "completed_at":_utc_now(),
                  "status_code":None, "model":MODEL, "model_found":False,
                  "model_metadata":None, "numeric_limit_headers":{},
                  "prompt_sent":False, "document_sent":False, "model_requests":0,
                  "error_code":error.code}
    except Exception:
        result = {"version":"opper-catalog-result-1", "completed_at":_utc_now(),
                  "status_code":None, "model":MODEL, "model_found":False,
                  "model_metadata":None, "numeric_limit_headers":{},
                  "prompt_sent":False, "document_sent":False, "model_requests":0,
                  "error_code":"catalog_request_failed"}
    prep.write_json("catalog-result.json", result)
    if result["error_code"] is not None:
        raise OpperPilotError(result["error_code"])
    return result


def execute(root: Path | str, run_id: str, *, confirm_live: bool,
            retention_reviewed: bool, load_dotenv_file: bool,
            acknowledge_untracked_history: bool = False) -> dict[str, Any]:
    if confirm_live is not True:
        raise OpperPilotError("live_confirmation_required")
    if retention_reviewed is not True:
        raise OpperPilotError("retention_review_required")
    root = workspace_root(root)
    prep = _prep(root, run_id)
    spec = load_publication_spec(root, f"runs/agent/{prep.run_id}/publication-spec.json")
    _verify_fixed(spec, run_id)
    try:
        catalog = _strict_json(prep.path / "catalog-result.json")
    except OpperPilotError:
        raise OpperPilotError("successful_catalog_check_required") from None
    if (catalog.get("error_code") is not None or catalog.get("model_found") is not True
            or catalog.get("model") != MODEL or catalog.get("status_code") != 200):
        raise OpperPilotError("successful_catalog_check_required")
    metadata = catalog.get("model_metadata")
    if not isinstance(metadata, dict):
        raise OpperPilotError("model_limits_unknown")
    context_length = metadata.get("context_length")
    provider_max_output = metadata.get("max_output_tokens")
    if type(context_length) is not int or type(provider_max_output) is not int:
        raise OpperPilotError("model_limits_unknown")
    if context_length < MAX_PROMPT_BYTES + 128 + MAX_OUTPUT_TOKENS:
        raise OpperPilotError("model_context_too_small")
    if provider_max_output < MAX_OUTPUT_TOKENS:
        raise OpperPilotError("model_output_limit_too_small")
    if load_dotenv_file:
        env_path = checked_path(root, ".env", allowed_roots=(".env",))
        load_dotenv(env_path, override=False)
    if not os.environ.get("OPPER_API_KEY"):
        raise OpperPilotError("missing_credentials")
    bundle, profile, prompt = _inputs(root, spec)
    scope = _first_scope(spec, bundle, profile, prompt)
    ledger = OpperRequestLedger(root)
    if acknowledge_untracked_history:
        for override_run_id in _history_override_run_ids(spec):
            acknowledgement = ledger.acknowledge_untracked_history(override_run_id)
            if acknowledgement.error_code:
                raise OpperPilotError(acknowledgement.error_code)
    readiness = ledger.reserve(run_id, scope, commit_attempt=False)
    if readiness.error_code:
        raise OpperPilotError(readiness.error_code)
    result = run_publication(root, spec, configured_provider_endpoint=ENDPOINT,
                             load_dotenv_file=False)
    return {
        "version":VERSION, "run_id":run_id, "status":result.get("status"),
        "workflow_status":result.get("workflow_status"),
        "new_model_calls":result.get("new_model_calls"),
        "new_live_requests":result.get("new_live_requests"),
        "known_input_tokens":result.get("known_input_tokens"),
        "known_output_tokens":result.get("known_output_tokens"),
        "usage_unknown":result.get("usage_unknown"),
        "maximum_reserved_output_tokens":MAX_CALLS * MAX_OUTPUT_TOKENS,
        "untracked_history_acknowledged":acknowledge_untracked_history,
        "cost":result.get("cost"), "cost_status":result.get("cost_status"),
        "canonical_relative":result.get("canonical_relative"),
        "sidecar_relative":result.get("sidecar_relative"),
    }


def report(root: Path | str, run_id: str) -> dict[str, Any]:
    root = workspace_root(root)
    prep = _prep(root, run_id)
    plan = _strict_json(prep.path / "token-plan.json")
    result = read_publication(root, run_id)
    return {
        "version":VERSION, "run_id":run_id, "status":result.get("status"),
        "workflow_status":result.get("workflow_status"),
        "total_model_calls":result.get("total_model_calls"),
        "total_live_requests":result.get("total_live_requests"),
        "known_input_tokens":result.get("known_input_tokens"),
        "known_output_tokens":result.get("known_output_tokens"),
        "usage_unknown":result.get("usage_unknown"),
        "cost":result.get("cost"), "cost_status":result.get("cost_status"),
        "token_plan":plan,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "catalog", "run", "report"):
        command = commands.add_parser(name)
        command.add_argument("--root", default=str(Path(__file__).resolve().parents[3]))
        command.add_argument("--run-id", required=True, type=_run_id)
        if name in {"catalog", "run"}:
            command.add_argument("--load-dotenv", action="store_true")
        if name == "run":
            command.add_argument("--execute-live", action="store_true")
            command.add_argument("--retention-reviewed", action="store_true")
            command.add_argument("--acknowledge-untracked-history", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.root, args.run_id)
        elif args.command == "catalog":
            root = workspace_root(args.root)
            if args.load_dotenv:
                load_dotenv(checked_path(root, ".env", allowed_roots=(".env",)), override=False)
            result = check_catalog(root, args.run_id, os.environ.get("OPPER_API_KEY", ""))
        elif args.command == "run":
            result = execute(args.root, args.run_id, confirm_live=args.execute_live,
                             retention_reviewed=args.retention_reviewed,
                             load_dotenv_file=args.load_dotenv,
                             acknowledge_untracked_history=args.acknowledge_untracked_history)
        else:
            result = report(args.root, args.run_id)
    except OpperPilotError as error:
        parser.exit(2, f"Opper one-paper command stopped ({error.code}).\n")
    except FileExistsError:
        parser.exit(2, "Opper one-paper command stopped (artifact_exists_use_fresh_run_id).\n")
    except Exception:
        parser.exit(2, "Opper one-paper command stopped (operation_failed).\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
