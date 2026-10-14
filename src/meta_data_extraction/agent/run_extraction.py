"""Isolated fixed-run execution and an offline-only fixture CLI for M02."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time
from typing import Literal

from meta_data_extraction.agent.config import load_offline_config
from meta_data_extraction.agent.contracts import FixedExtraction, canonical_json, digest
from meta_data_extraction.agent.evidence import FrozenModel, resolve_evidence, sha256
from meta_data_extraction.agent.extraction import EVIDENCE_PROMPT_VERSION, Variant, extract_fixed, render_prompt
from meta_data_extraction.agent.paths import input_file, workspace_root
from meta_data_extraction.agent.profiles import DEFAULT_SCHEMAS, load_profile, validate_record
from meta_data_extraction.agent.providers import LiveRunSpec, MockProvider, ProviderResult, create_live_provider
from meta_data_extraction.agent.sources import prepare_sources
from meta_data_extraction.agent.storage import RunConflictError, RunStore


class RunReport(FrozenModel):
    run_id: str
    execution: Literal["mock", "live", "cached_replay"]
    new_live_requests: int
    elapsed_ms: float
    extraction: FixedExtraction


def _implementation(root: Path) -> dict[str, str]:
    # Explicit code provenance, never a corpus/retrieval index.
    names = ["agent/" + name + ".py" for name in (
        "config", "paths", "evidence", "sources", "profiles", "providers", "contracts",
        "extraction", "storage", "run_extraction",
    )] + ["components/extractor.py", "components/llm.py", "components/validator.py",
          "components/parser.py", "base/llm.py", "utils.py", "data_models.py"]
    return {name: sha256(input_file(root, "src/meta_data_extraction/" + name,
                                   allowed_roots=("src/meta_data_extraction",)).read_bytes())
            for name in names}


def run_fixed(root: Path | str, *, run_id: str, source: str,
              schemas: tuple[str, ...] = DEFAULT_SCHEMAS, variant: Variant = "evidence",
              mock: MockProvider | None = None, live_spec: LiveRunSpec | None = None,
              configured_chatai_endpoint: str | None = None, load_dotenv_file: bool = False,
              retain_raw: bool) -> RunReport:
    """Prepare one explicit source and execute once, or verify a completed replay.

    Live callers must supply the authorized ChatAI endpoint from their trusted
    configuration, plus an exact source/profile/prompt run specification. Merely
    finding a key or accepting a model/document-supplied URL is not authorization.
    The CLI below deliberately exposes only offline synthetic fixture execution.
    """
    started = time.perf_counter()
    root = workspace_root(root)
    if (mock is None) == (live_spec is None):
        raise ValueError("Choose exactly one explicit mock provider or live run specification")
    if mock is not None and not isinstance(mock, MockProvider):
        raise TypeError("Mock execution requires the deterministic mock adapter")
    if live_spec is not None:
        live_spec = LiveRunSpec.model_validate(live_spec.model_dump(mode="json"))
    if type(retain_raw) is not bool:
        raise TypeError("Choose an explicit raw-output retention policy")
    prepared = prepare_sources(root, [source])[0]
    if prepared.bundle is None:
        raise ValueError("Source preparation failed: " + prepared.error.code)
    bundle = prepared.bundle
    profile = load_profile(root, schema_names=schemas)
    prompt = render_prompt(bundle, profile, variant)
    prompt_hash = sha256(prompt.encode("utf-8"))
    if live_spec is not None:
        if not configured_chatai_endpoint or live_spec.endpoint != configured_chatai_endpoint:
            raise ValueError("Live scope must match the explicitly configured ChatAI endpoint")
        if (live_spec.profile_sha256 != profile.profile_id.removeprefix("profile:")
                or {(s.path, s.sha256) for s in live_spec.sources} != {(bundle.source_id, bundle.source_sha256)}
                or live_spec.approved_prompt_sha256 != (prompt_hash,)):
            raise ValueError("This fixed run requires exactly its source, profile and prompt")
        descriptor = {"mode": "live", "model": live_spec.model, "spec": live_spec.model_dump(mode="json")}
    else:
        descriptor = {"mode": "mock", "model": mock.model_name,
                      "configuration_sha256": mock.configuration_fingerprint}
    request = {
        "state_version": "m02-run-1", "source_id": bundle.source_id,
        "source_sha256": bundle.source_sha256, "representation_sha256": bundle.representation_sha256,
        "source_snapshot_sha256": digest(bundle.model_dump(mode="json")),
        "profile": profile.model_dump(mode="json"), "variant": variant,
        "prompt_version": profile.prompt_version if variant == "inherited" else EVIDENCE_PROMPT_VERSION,
        "prompt_sha256": prompt_hash, "provider": descriptor,
        "retain_raw": retain_raw, "implementation": _implementation(root),
        "dependencies": {name: version(name) for name in ("pydantic", "openai", "httpx")},
        "standards_resources": [], "evaluator": None,
    }
    store = RunStore.begin(root, run_id, request)
    # Match RunStore's serialized request bytes, including its terminal newline.
    request_hash = sha256((canonical_json(request) + "\n").encode("utf-8"))
    if store.completed is not None:
        try:
            outcome = FixedExtraction.model_validate(store.completed["extraction"])
            if store.completed["request_sha256"] != request_hash:
                raise ValueError("Replay request differs")
            if outcome.snapshot:
                snap = outcome.snapshot
                if (snap.source_id, snap.source_sha256, snap.representation_sha256, snap.profile_id) != (
                    bundle.source_id, bundle.source_sha256, bundle.representation_sha256, profile.profile_id,
                ) or validate_record(snap.canonical, profile):
                    raise ValueError("Replay source/profile or canonical structure differs")
                for assertion in outcome.assertions:
                    for ref in assertion.evidence:
                        resolve_evidence(bundle, ref)
        except (ValueError, TypeError, KeyError) as exc:
            raise RunConflictError("Completed extraction no longer validates") from exc
        return RunReport(run_id=run_id, execution="cached_replay", new_live_requests=0,
                         elapsed_ms=(time.perf_counter() - started) * 1000, extraction=outcome)
    provider = mock if mock is not None else create_live_provider(store.run, live_spec,
                                                                 load_dotenv_file=load_dotenv_file)
    outcome = extract_fixed(root, bundle, profile, provider, variant=variant)
    new_live_requests = int(outcome.provider.mode == "live" and outcome.provider.request_id is not None)
    elapsed = (time.perf_counter() - started) * 1000
    result = {
        "state_version": "m02-run-1", "request_sha256": request_hash,
        "execution": descriptor["mode"], "new_live_requests": new_live_requests,
        "elapsed_ms": elapsed, "cost": None, "cost_status": "unknown" if live_spec else "not_applicable",
        "raw_retention": "unavailable" if outcome.raw_output is None else "retained" if retain_raw else "discarded",
        "extraction": outcome.model_dump(mode="json"),
    }
    store.complete(result, canonical=outcome.snapshot.canonical if outcome.snapshot else None,
                   sidecar={"source": bundle.model_dump(mode="json"), "profile": profile.model_dump(mode="json"),
                            "extraction": outcome.model_dump(mode="json")},
                   raw_response=outcome.raw_output, retain_raw=retain_raw)
    return RunReport(run_id=run_id, execution=descriptor["mode"], new_live_requests=new_live_requests,
                     elapsed_ms=elapsed, extraction=outcome)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", required=True, help="An existing agent offline source/schema config")
    parser.add_argument("--response", required=True, help="Explicit synthetic raw response fixture")
    parser.add_argument("--variant", choices=("inherited", "evidence"), default="evidence")
    parser.add_argument("--run-id", help="Optional fresh run ID override")
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument("--retain-raw", action="store_true")
    policy.add_argument("--discard-raw", action="store_true")
    args = parser.parse_args()
    root = workspace_root(args.root)
    config = load_offline_config(args.config, root)
    if len(config.sources) != 1:
        parser.error("M02 fixed execution accepts exactly one source")
    response = input_file(root, args.response, allowed_roots=("tests/agent/fixtures",)).read_text(encoding="utf-8")
    mock = MockProvider([ProviderResult(status="ok", content=response, finish_reason="stop")])
    report = run_fixed(root, run_id=args.run_id or config.run_id, source=config.sources[0],
                       schemas=config.schemas, variant=args.variant, mock=mock, retain_raw=args.retain_raw)
    # Only safe run diagnostics go to stdout; extracted text/raw output stay local.
    print(json.dumps({"run_id": report.run_id, "execution": report.execution,
                      "status": report.extraction.status, "parse_status": report.extraction.parse_status,
                      "schema_status": report.extraction.schema_status,
                      "evidence_status": report.extraction.evidence_status,
                      "new_live_requests": report.new_live_requests}))
    return 1 if report.extraction.status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
