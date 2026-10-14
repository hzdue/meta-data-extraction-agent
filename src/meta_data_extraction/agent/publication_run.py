"""Source-to-review integration using existing extraction, checking and workflow APIs.

Each model stage has a fresh, immutable execution claim. Resuming reuses completed
stages; uncertain dispatch never authorizes a second attempt. All live stages share
the original checkout's RequestLedger. No evaluator or gold input is imported.
"""
from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path
import platform
from typing import Any

from .config import SOURCE_ROOTS
from .contracts import FixedExtraction, ProviderObservation, digest
from .evidence import sha256
from .extraction import render_prompt
from .paths import RunDirectory, checked_path, input_file, workspace_root
from .profiles import load_profile
from .providers import LiveRunSpec, MockProvider, ProviderResult, create_live_provider
from .publication_checks import (
    PublicationCheckError, eligible_patch_paths, issues_from_support, parse_patch_response,
    plan_support, render_patch_prompt,
)
from .publication_standards import load_reviewed_nexus, map_record_techniques, standards_input_pins
from .publication_types import PublicationSpec
from .record_patches import PatchPolicy, build_candidate
from .run_extraction import run_fixed
from .sources import prepare_sources
from .storage import RunConflictError, RunIncompleteError, RunStore, _check_request, _digest, _json_bytes, _read_completion
from .support_checks import SupportReport, SupportTarget, check_support, render_support_prompt, verify_report
from .workflow import WorkflowMockProvider, WorkflowRuntime
from .workflow_graph import _lease, run_workflow_graph
from .workflow_tools import ToolContext, execute_tool
from .workflow_types import RetrieveSource, WorkflowBudget, WorkflowState

VERSION = "publication-parent-1"
STAGES = ("extract", "check", "proposal", "provider", "workflow", "wc1", "wc2")
FILES = ("request.json", "result.json", "canonical.json", "sidecar.json", "raw-response.json",
         "live-run-spec.json", "dispatch.sqlite", "checkpoints.sqlite")
MAX_BYTES = 8_000_000


def _child(spec: PublicationSpec, stage: str) -> str:
    if stage not in STAGES:
        raise ValueError("Unknown publication stage")
    return "pub-" + digest([spec.run_id, spec.version])[:16] + "-" + stage


def _read(path: Path) -> dict:
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Publication artifact exceeds the size bound")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate publication key")
            result[key] = value
        return result
    value = json.loads(path.read_bytes(), object_pairs_hook=pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    if not isinstance(value, dict):
        raise ValueError("Expected publication object")
    return value


def _implementation() -> dict:
    base = Path(__file__).parent
    names = ("publication_types", "publication_checks", "publication_standards", "publication_run",
             "run_extraction", "extraction", "sources", "config", "contracts", "evidence", "profiles",
             "providers", "paths", "storage", "record_checks", "record_patches", "support_checks",
             "retrieval", "workflow", "workflow_graph", "workflow_types", "workflow_tools",
             "workflow_policies", "panet_resources", "panet_lookup", "panet_mapping",
             "nexus_resources", "nexus_registry", "nexus_units")
    return {name:sha256((base/(name+".py")).read_bytes()) for name in names}


def _prepare(root: Path, spec: PublicationSpec):
    source = input_file(root, spec.source.path, allowed_roots=SOURCE_ROOTS)
    if sha256(source.read_bytes()) != spec.source.sha256:
        raise RunConflictError("Publication source hash differs")
    prepared = prepare_sources(root, [spec.source.path])[0]
    if prepared.bundle is None:
        raise ValueError("Publication source preparation failed")
    bundle = prepared.bundle
    if bundle.source_sha256 != spec.source.sha256:
        raise RunConflictError("Publication source changed during preparation")
    profile = load_profile(root, spec.schemas)
    responses = []
    for pin in spec.mock_responses:
        path = input_file(root, pin.path, allowed_roots=("tests/agent/fixtures",))
        raw = path.read_bytes()
        if len(raw) > 1_000_000 or sha256(raw) != pin.sha256:
            raise RunConflictError("Mock response pin differs")
        responses.append(ProviderResult(status="ok", content=raw.decode("utf-8"), finish_reason="stop"))
    standard_pins = standards_input_pins(root, panet=spec.panet, nexus=spec.nexus)
    request = {"version":VERSION, "spec":spec.model_dump(mode="json"),
               "spec_sha256":digest(spec.model_dump(mode="json")),
               "source_snapshot_sha256":digest(bundle.model_dump(mode="json")),
               "profile":profile.model_dump(mode="json"), "standards_inputs":standard_pins,
               "implementation":_implementation(), "python":platform.python_version(),
               "dependencies":{name:version(name) for name in ("pydantic", "openai", "httpx", "pymupdf4llm", "pymupdf",
                   "langgraph", "langgraph-checkpoint-sqlite")},
               "children":{stage:_child(spec,stage) for stage in STAGES},
               "max_reserved_output_tokens":spec.max_calls*spec.live.max_output_tokens if spec.live else None,
               "external_tracing":"disabled", "evaluator":None}
    return bundle, profile, tuple(responses), request


def _parent(root: Path, spec: PublicationSpec, request: dict) -> RunStore:
    try:
        return RunStore.begin(root, spec.run_id, request)
    except RunIncompleteError:
        run = RunDirectory(root, spec.run_id)
        fingerprint = _digest(_json_bytes(request))
        _check_request(run, request, fingerprint)
        # Only stage-bound resume is supported; a half-published parent is uncertain.
        if any((run.path/name).exists() for name in ("canonical.json", "sidecar.json", "result.pending.json")):
            raise RunIncompleteError("Parent export interrupted; no implicit rewrite")
        return RunStore(run, None, request, fingerprint)


def _live_scope(spec, profile, bundle, prompts, calls):
    live = spec.live
    if live is None or not prompts or not 1 <= calls <= spec.max_calls:
        raise ValueError("Invalid bounded live stage")
    return LiveRunSpec(**live.model_dump(), sources=({"path":bundle.source_id,"sha256":bundle.source_sha256},),
        split=spec.split, profile_sha256=profile.profile_id.removeprefix("profile:"),
        approved_prompt_sha256=tuple(dict.fromkeys(sha256(p.encode("utf-8")) for p in prompts)),
        max_calls=calls, max_total_output_tokens=calls*live.max_output_tokens)


def _stage(root, spec, parent_request, name, prompt):
    request = {"version":"publication-stage-1", "parent_sha256":digest(parent_request),
               "stage":name, "prompt_sha256":sha256(prompt.encode("utf-8"))}
    return RunStore.begin(root, _child(spec,name), request)


def _provider(store, spec, profile, bundle, prompts, responses, offset, *, calls=1, load_dotenv_file=False):
    if spec.mode == "live":
        return create_live_provider(store.run, _live_scope(spec,profile,bundle,prompts,calls),
                                    load_dotenv_file=load_dotenv_file)
    return MockProvider(responses[offset:offset+calls], model_name="publication-mock")


def _observation(response: ProviderResult, model: str) -> ProviderObservation:
    return ProviderObservation(**response.model_dump(exclude={"content"}), model=model,
        response_sha256=sha256(response.content.encode("utf-8")) if response.content is not None else None)


def _generate(provider, prompt):
    try:
        response = ProviderResult.model_validate(provider.generate(prompt).model_dump())
        if response.mode != provider.mode:
            return ProviderResult(status="failed", mode=provider.mode, error_code="provider_mode_mismatch")
        return response
    except TimeoutError:
        return ProviderResult(status="timeout", mode=provider.mode, error_code="provider_timeout")
    except Exception:
        return ProviderResult(status="failed", mode=provider.mode, error_code="adapter_exception")


def _pins(root, spec):
    pins = {}
    for stage in STAGES:
        run = RunDirectory(root, _child(spec,stage))
        for name in FILES:
            path = checked_path(root, f"runs/agent/{run.run_id}/{name}", allowed_roots=("runs/agent",))
            if path.is_file():
                pins[f"runs/agent/{run.run_id}/{name}"] = sha256(path.read_bytes())
    return pins


def _verify_pins(root, spec, expected):
    if not isinstance(expected, dict) or _pins(root,spec) != expected:
        raise RunConflictError("Publication child artifacts changed")


def _summary(spec, status, *, execution, calls, live, observations, state=None, completed_stage=None):
    return {"version":VERSION, "run_id":spec.run_id, "spec_sha256":digest(spec.model_dump(mode="json")),
            "workflow_run_id":_child(spec,"workflow"), "status":status,
            "workflow_status":state.status if state else None,
            "execution":execution, "new_model_calls":calls, "new_live_requests":live,
            "total_model_calls":len(observations)+(state.uncertain_model_calls if state else 0),
            "uncertain_model_calls":state.uncertain_model_calls if state else 0,
            "total_live_requests":sum(p.mode=="live" and p.request_id is not None for p in observations),
            "live_dispatch_unknown":any(p.mode=="live" and p.request_id is None and p.status!="blocked" for p in observations)
                or bool(state and state.uncertain_model_calls and spec.mode=="live"),
            "usage_unknown":any(p.input_tokens is None or p.output_tokens is None for p in observations)
                or bool(state and state.uncertain_model_calls),
            "known_input_tokens":sum(p.input_tokens or 0 for p in observations),
            "known_output_tokens":sum(p.output_tokens or 0 for p in observations),
            "cost":None, "cost_status":"unknown" if spec.mode=="live" else "not_applicable_mock",
            "record_revision":state.record.revision_id if state else None,
            "canonical_relative":f"runs/agent/{spec.run_id}/canonical.json",
            "sidecar_relative":f"runs/agent/{spec.run_id}/sidecar.json",
            "completed_stage":completed_stage, "coverage_status":"selected_assertions_only",
            "entity_recovery_status":"not_assessed", "empirical_status":"not_run"}


def _workflow_prompts(context, initial):
    """Pre-render only deterministic retrieval/candidate operations, never dispatch."""
    prompts = []
    for issue in initial.issues:
        if not issue.proposal_ids:
            continue
        retrieval = execute_tool(context, RetrieveSource(issue_id=issue.id,query=issue.query), initial,
                                 model_budget=0, unknown_usage="allow_bounded_calls")
        refs = {digest(ref.model_dump(mode="json")):ref for ref in (*issue.evidence,*retrieval.evidence)}
        evidence = tuple(refs.values())[-8:]
        if not evidence:
            continue
        proposal = context.proposals[issue.proposal_ids[0]]
        built = build_candidate(context.profile, context.bundle, initial.record, proposal)
        if built.prepared is None:
            continue
        prepared = built.prepared
        for record, paths in ((prepared.original,prepared.changed_paths), (prepared.candidate,prepared.required_support_paths)):
            prompts.append(render_support_prompt(context.profile,record,context.bundle,
                          tuple(SupportTarget(path=path,evidence=evidence) for path in paths)))
    return tuple(dict.fromkeys(prompts))


def _matches_field(pattern: str, path: str) -> bool:
    """Match a registry JSON pointer; wildcards address array indices only."""
    expected, actual = pattern.split("/"), path.split("/")
    return len(expected) == len(actual) and all(
        left == right or (left == "*" and right.isascii() and right.isdecimal())
        for left, right in zip(expected, actual, strict=True)
    )


def _record_checker(root, spec, parent_request, checker):
    """Persist each workflow checker dispatch in addition to M07 transition claims."""
    original = checker.generate
    counter = [0]
    def generate(prompt):
        # Only one combined proposal is attempted, so at most two check calls exist.
        counter[0] += 1
        if counter[0] > 2:
            raise RunConflictError("Workflow checker call bound exceeded")
        store = _stage(root,spec,parent_request,"wc"+str(counter[0]),prompt)
        if store.completed is not None:
            raise RunIncompleteError("A recorded checker dispatch cannot be repeated")
        response = _generate_callable(original, checker.mode, prompt)
        observation = _observation(response,checker.model_name)
        store.complete({"observation":observation.model_dump(mode="json")}, canonical=None,
                       sidecar={"prompt_sha256":sha256(prompt.encode("utf-8")),
                                "observation":observation.model_dump(mode="json")},
                       raw_response=response.content, retain_raw=spec.raw_retention=="retain")
        return response
    checker.generate = generate
    return checker


def _generate_callable(generate, mode, prompt):
    try:
        response = ProviderResult.model_validate(generate(prompt).model_dump())
        if response.mode != mode:
            return ProviderResult(status="failed", mode=mode, error_code="provider_mode_mismatch")
        return response
    except TimeoutError:
        return ProviderResult(status="timeout", mode=mode, error_code="provider_timeout")
    except Exception:
        return ProviderResult(status="failed", mode=mode, error_code="adapter_exception")


def run_publication(root: Path | str, spec: PublicationSpec, *, max_steps: int | None = None,
                    stop_after: str | None = None, configured_chatai_endpoint: str | None = None,
                    load_dotenv_file: bool = False) -> dict:
    """Extract actual supplied source, check it, then run the existing controller.

    stop_after/max_steps pause only after durable completion; use the exact same
    specification to resume. A failed/uncertain stage is never implicitly retried.
    """
    root = workspace_root(root)
    spec = PublicationSpec.model_validate(spec.model_dump(mode="json"))
    if stop_after not in {None,"extraction","check","proposal"}:
        raise ValueError("Unknown publication pause point")
    if max_steps is not None and (type(max_steps) is not int or not 1 <= max_steps <= 32):
        raise ValueError("max_steps must be in 1..32")
    if type(load_dotenv_file) is not bool:
        raise ValueError("Explicit credential-loading choice required")
    if spec.mode == "live" and configured_chatai_endpoint != spec.live.endpoint:
        raise ValueError("Live publication endpoint must match trusted configured ChatAI endpoint")
    bundle, profile, responses, request = _prepare(root,spec)
    store = _parent(root,spec,request)
    observations = []
    new_calls = 0
    new_live = 0
    with _lease(store.run):
        if store.completed is not None:
            sidecar = _read(store.run.path/"sidecar.json")
            _verify_pins(root,spec,sidecar["child_pins"])
            return {**store.completed,"execution":"cached_replay","new_model_calls":0,"new_live_requests":0}

        def checkpoint(stage, state=None):
            return _summary(spec,"checkpointed",execution="checkpointed",calls=new_calls,live=new_live,
                            observations=observations,state=state,completed_stage=stage)

        def complete(status, canonical, details, state=None):
            summary = _summary(spec,status,execution=spec.mode,calls=new_calls,live=new_live,
                               observations=observations,state=state)
            if canonical is None:
                summary["canonical_relative"] = None
            sidecar = {"source":bundle.model_dump(mode="json"),"profile":profile.model_dump(mode="json"),
                       **details,"workflow_run_id":_child(spec,"workflow"),"child_pins":_pins(root,spec),
                       "observations":[p.model_dump(mode="json") for p in observations],
                       "coverage_status":"selected_assertions_only","expert_approval":"not_supplied"}
            store.complete(summary, canonical=canonical, sidecar=sidecar,raw_response=None,retain_raw=False)
            return summary

        prompt = render_prompt(bundle,profile,"evidence")
        extracted = run_fixed(root,run_id=_child(spec,"extract"),source=spec.source.path,schemas=spec.schemas,
            variant="evidence", mock=MockProvider(responses[:1],model_name="publication-mock") if spec.mode=="mock" else None,
            live_spec=_live_scope(spec,profile,bundle,(prompt,),1) if spec.mode=="live" else None,
            configured_chatai_endpoint=configured_chatai_endpoint,load_dotenv_file=load_dotenv_file,
            retain_raw=spec.raw_retention=="retain")
        extraction = extracted.extraction
        observations.append(extraction.provider)
        if extracted.execution != "cached_replay":
            new_calls += 1
            new_live += extracted.new_live_requests
        details = {"extraction":extraction.model_dump(mode="json")}
        if extraction.snapshot is None or extraction.status == "failed":
            return complete("extraction_failed",None,details)
        record = extraction.snapshot
        if stop_after == "extraction":
            return checkpoint("extraction")
        if spec.unknown_usage == "stop" and (extraction.provider.input_tokens is None or extraction.provider.output_tokens is None):
            return complete("unknown_usage",record.canonical,details)

        try:
            plan = plan_support(profile,record,bundle,assertions=extraction.assertions,
                                max_targets=spec.max_targets,evidence_chars=spec.evidence_chars,
                                retrieval_strategy=spec.retrieval_strategy)
        except PublicationCheckError as error:
            return complete("check_unavailable",record.canonical,{**details,"check_error":error.code})
        details["support_plan"] = plan.model_dump(mode="json")
        check_prompt = render_support_prompt(profile,record,bundle,plan.targets)
        check_store = _stage(root,spec,request,"check",check_prompt)
        if check_store.completed is not None:
            support = SupportReport.model_validate(check_store.completed["report"])
        else:
            provider = _provider(check_store,spec,profile,bundle,(check_prompt,),responses,len(observations),
                                 load_dotenv_file=load_dotenv_file)
            support = check_support(root,profile,record,bundle,plan.targets,provider,raw_retention=spec.raw_retention)
            check_store.complete({"report":support.model_dump(mode="json")},canonical=None,
                sidecar={"report":support.model_dump(mode="json")},raw_response=support.raw_output,
                retain_raw=spec.raw_retention=="retain")
            if support.provider:
                new_calls += 1
                new_live += int(support.provider.mode=="live" and support.provider.request_id is not None)
        verify_report(support,profile,record,bundle,root=root)
        if support.provider:
            observations.append(support.provider)
        details["support_report"] = support.model_dump(mode="json")
        if stop_after == "check":
            return checkpoint("check")

        generation = None
        proposal = None
        automatic_capable = spec.allow_automatic_patches and not plan.source_incomplete
        reserve = 2 if automatic_capable else 0
        unknown = any(p.input_tokens is None or p.output_tokens is None for p in observations)
        can_propose = spec.generate_proposal and not (unknown and spec.unknown_usage=="stop") and (
            len(observations)+1+reserve <= spec.max_calls) and bool(eligible_patch_paths(profile,record,bundle,support))
        if can_propose:
            proposal_prompt = render_patch_prompt(profile,record,bundle,support)
            proposal_store = _stage(root,spec,request,"proposal",proposal_prompt)
            if proposal_store.completed is not None:
                saved = proposal_store.completed
            else:
                provider = _provider(proposal_store,spec,profile,bundle,(proposal_prompt,),responses,len(observations),
                                     load_dotenv_file=load_dotenv_file)
                response = _generate(provider,proposal_prompt)
                observation = _observation(response,provider.model_name)
                saved = {"observation":observation.model_dump(mode="json"),"generation":None,"error":None}
                if response.status == "ok" and response.finish_reason == "stop" and response.content:
                    try:
                        saved["generation"] = parse_patch_response(response.content,profile,record,bundle,support).model_dump(mode="json")
                    except PublicationCheckError as error:
                        saved["error"] = error.code
                else:
                    saved["error"] = "proposal_response_unusable"
                proposal_store.complete(saved,canonical=None,sidecar=saved,raw_response=response.content,
                                        retain_raw=spec.raw_retention=="retain")
                new_calls += 1
                new_live += int(response.mode=="live" and response.request_id is not None)
            observations.append(ProviderObservation.model_validate(saved["observation"]))
            generation = saved["generation"]
            if generation:
                from .publication_checks import PatchGeneration
                parsed = PatchGeneration.model_validate(generation)
                if parsed.record_revision != record.revision_id or parsed.support_report_id != support.report_id:
                    raise RunConflictError("Proposal belongs to different checking inputs")
                proposal = parsed.proposal
            details["proposal_error"] = saved["error"]
        details["patch_generation"] = generation
        if stop_after == "proposal":
            return checkpoint("proposal")

        nexus = load_reviewed_nexus(root,profile,spec.nexus) if spec.nexus else None
        proposals = {proposal.proposal_id:proposal} if proposal else {}
        issues = issues_from_support(profile,record,bundle,plan,support,proposals=tuple(proposals.values()))
        # One combined correction may be attempted once. Other cases remain reviewable
        # if its checks fail, rather than repeating the same model proposal.
        leaders = [i for i in issues if i.proposal_ids]
        if leaders:
            leader = leaders[0]
            issues = (leader, *(i.model_copy(update={"proposal_ids":()}) for i in issues if i.id!=leader.id))
        if nexus and nexus.definition_requests:
            updated = []
            for issue in issues:
                keys = tuple(k for k,(field,role) in nexus.definition_requests.items()
                             if _matches_field(field, issue.path))
                updated.append(issue.model_copy(update={"definition_keys":keys}))
            issues = tuple(updated)
        initial = WorkflowState(record=record,initial_revision=record.revision_id,issues=issues)
        available = min(2,max(0,spec.max_calls-len(observations)))
        usage_stopped = spec.unknown_usage == "stop" and any(
            p.input_tokens is None or p.output_tokens is None for p in observations)
        details["automatic_patch_status"] = (
            "blocked_unknown_usage" if usage_stopped else
            "blocked_source_incomplete" if spec.allow_automatic_patches and plan.source_incomplete else
            "enabled" if automatic_capable else "disabled")
        policy = PatchPolicy(accept_model_judgments=automatic_capable and not usage_stopped)
        # Stub is used only to compute deterministic tool input; it never dispatches.
        checker = WorkflowMockProvider(responses[len(observations):len(observations)+available],model_name="publication-mock")
        context_args = dict(root=root,profile=profile,bundle=bundle,proposals=proposals,patch_policy=policy,
                            nxdl=nexus.snapshot if nexus else None,registry=nexus.registry if nexus else None,
                            definition_requests=nexus.definition_requests if nexus else {})
        context = ToolContext(checker=checker,**context_args)
        if spec.mode == "live" and policy.accept_model_judgments and proposals and available>=2:
            prompts = _workflow_prompts(context,initial)
            if prompts:
                setup_request = {"version":"publication-provider-1","parent_sha256":digest(request),
                                 "scope":_live_scope(spec,profile,bundle,prompts,available).model_dump(mode="json")}
                setup = RunStore.begin(root,_child(spec,"provider"),setup_request)
                checker = create_live_provider(setup.run,_live_scope(spec,profile,bundle,prompts,available),
                                               load_dotenv_file=load_dotenv_file)
                if setup.completed is None:
                    setup.complete({"status":"scope_frozen"},canonical=None,sidecar=setup_request,
                                   raw_response=None,retain_raw=False)
        checker = _record_checker(root,spec,request,checker)
        context = ToolContext(checker=checker,**context_args)
        budget = WorkflowBudget(total_actions=32,per_issue_actions=5,max_model_calls=available,
                                no_progress_limit=4,timeout_seconds=spec.workflow_timeout_seconds,
                                unknown_usage=spec.unknown_usage)
        workflow = run_workflow_graph(root,_child(spec,"workflow"),
            {"version":"publication-workflow-1","proposals":{k:v.model_dump(mode="json") for k,v in proposals.items()}},
            initial,WorkflowRuntime(tools=context,budget=budget,policy=spec.policy),max_steps=max_steps)
        state = workflow["state"]
        new_calls += workflow["new_model_calls"]
        new_live += workflow["new_live_requests"]
        observations.extend(p for e in state.events for p in e.providers)
        if len(observations)+state.uncertain_model_calls > spec.max_calls:
            raise RunConflictError("Aggregate publication call budget exceeded")
        if state.status == "running":
            return checkpoint("workflow",state)
        standards = {"panet":None,"nexus":nexus.model_dump(mode="json") if nexus else {"status":"not_configured"}}
        if spec.panet:
            standards["panet"] = map_record_techniques(root,bundle,state.record,spec.panet,
                assertions=extraction.assertions if state.record.revision_id==record.revision_id else ()).model_dump(mode="json")
        details.update(standards=standards,workflow_state=state.model_dump(mode="json"),
                       initial_record_revision=record.revision_id,
                       review_available=state.status=="pending_review")
        return complete(state.status,state.record.canonical,details,state)


def read_publication(root: Path | str, run_id: str) -> dict:
    """Verify frozen inputs and committed children; never create a provider or dispatch."""
    root = workspace_root(root)
    run = RunDirectory(root,run_id)
    envelope = _read(run.path/"request.json")
    request = envelope["request"]
    if request.get("version") != VERSION:
        raise RunConflictError("Not a publication parent")
    spec = PublicationSpec.model_validate(request["spec"])
    if spec.run_id != run_id:
        raise RunConflictError("Publication identity differs")
    _,_,_,expected = _prepare(root,spec)
    _check_request(run,expected,_digest(_json_bytes(expected)))
    if (run.path/"result.json").is_file():
        result = _read_completion(run,_digest(_json_bytes(expected)))["result"]
        _verify_pins(root,spec,_read(run.path/"sidecar.json")["child_pins"])
        return {**result,"execution":"status","new_model_calls":0,"new_live_requests":0}
    return {"version":VERSION,"run_id":run_id,"spec_sha256":digest(spec.model_dump(mode="json")),
            "workflow_run_id":_child(spec,"workflow"),"status":"incomplete","workflow_status":None,
            "execution":"status","new_model_calls":0,"new_live_requests":0,"record_revision":None}
