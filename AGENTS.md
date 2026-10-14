# AGENTS.md — meta-data-extraction-agent

## Purpose and current direction

Build a small, evidence-linked metadata extraction service for photon and neutron publications. Use PaNET for technique terminology, selected NeXus definitions for representation guidance, and publications as evidence of experimental facts. Use Python, Pydantic, and LangGraph as a thin workflow layer around ordinary, testable functions. One bounded controller is sufficient.

The working research hypothesis is that diagnosing the reason for an issue can improve the choice between standards lookup, further source inspection, a local correction, and human review. This is a hypothesis, not an established novelty or performance result.

For module development or experiments, use `CODEX_MODULE_PLAN.md` and the selected module's record under `docs/agent/modules/`. Consult only the sections relevant to the task. Routine fixes do not require rereading the full plan.

## Repository conventions — inspect before changing

At the planning snapshot, the repository contains the inherited `meta_data_extraction` Python package, benchmark data, evaluation code, and paper-result scripts. Do not recreate the repository or automatically remove these assets. Reconcile this description with the current checkout.

- Keep the existing Python requirement and `uv` environment unless a documented compatibility problem requires a change. Do not rename the distribution or import package merely to match the repository name.
- Prefer additive development under `src/meta_data_extraction/agent/`, `tests/agent/`, `configs/agent/`, and `docs/agent/`. Adapt these locations when the actual checkout already has an equivalent implementation; do not create duplicate stacks.
- Reuse the existing parser, provider adapter, and schema loading where suitable. Preserve their historical behavior behind adapters and regression tests.
- Keep released data, predictions, evaluator configuration, and paper results unchanged. In particular, protect `data/gold/`, `data/annotations/`, `data/experiment/`, `outputs/`, `paper/`, and existing benchmark configs. New run artifacts belong under ignored `runs/agent/<run_id>/`.
- Preserve user changes. Do not reset, clean, rewrite Git history, switch an unexpected branch, or delete unrelated files. Do not push, publish, or create remote resources unless requested. Do not modify another checkout.
- Preserve attribution and licenses. An empty or absent license is unresolved, not permission to select a license or redistribute data. Record issues without attempting legal conclusions.

## Work one module at a time

When asked to design and implement module `Mxx`:

1. Inspect the relevant current code and status; identify what can be reused.
2. State the module question, interfaces, minimal design, baseline, and required tests in its module record. Keep the note short and proceed with the authorized implementation; do not stop at a proposal.
3. Implement the smallest complete behavior and appropriate unit/integration tests. Use offline fixtures by default.
4. Run the relevant safe checks. Run a live experiment only within explicitly authorized endpoint, data, and resource limits.
5. Record outcomes and update `docs/agent/PROGRESS.md`. Distinguish implementation readiness from empirical validation. Explain one concrete example and the main design trade-off so the user can understand and maintain the work.
6. Stop after the requested module and report the next dependency. Do not implement the whole roadmap opportunistically.

If no module is specified for a new development task, start with M00. When a dependency is missing, complete independent work and report the precise blocker. Do not invent data, an approval, a model response, a citation, or a successful experiment to pass a gate.

## Scientific boundaries

- Canonical schema meaning, identity/granularity rules, missingness conventions, and local-ID semantics remain fixed within an experiment. Store evidence, standardized mappings, issues, and review state in sidecars or an internal envelope. Do not silently add them to the canonical benchmark JSON.
- Gold records, adjudicated correspondences, evaluator outputs, `configs/alignment_helper.yaml`, and paper-specific evaluation aliases are OFFLINE ONLY. Runtime retrieval must index only the explicitly supplied source documents and approved standards resources, never the repository as a whole.
- Keep the evaluator version and metric definitions fixed for an experiment. Do not tune the evaluator to favor new outputs. Record new-paper limitations of inherited alignment helpers.
- PaNET/NeXus definitions are not publication evidence. A required standard field is not proof that the publication reports a value. Do not invent ontology IDs, mappings, units, approvals, or complete-NeXus compliance.
- Preserve original source expressions alongside approved normalizations. Distinguish extraction from external enrichment. Missing evidence is not necessarily falsity; a valid JSON record is not necessarily supported or complete.
- Do not silently merge samples with matching composition or reassign measurements by convenience. Verify attachments, not merely the occurrence of scalar values.
- Inspect final-test gold only in the explicitly authorized offline scoring/adjudication step. Do not feed resulting judgments back into a supposedly frozen method. Split by publication, including every chunk, version, prediction, and synthetic variant derived from it.
- Use “gold entities.” Report entity recovery, conditional attributes with support, and references separately. Do not invent a schema-wide average or claim completeness from checking emitted records.

## Implementation requirements

Use typed boundaries and explicit error states. Preserve raw model output under the approved retention policy. A parse failure must not become a successful empty extraction. Record truncation, unavailable source regions, timeouts, failed calls, and unknown usage.

Evidence links must resolve to actual source regions and a source hash. Do not invent page locations. Record assertions against stable entity IDs and record revisions so edits cannot silently invalidate their evidence.

Keep domain logic independent of LangGraph. Restrict model-selected actions to typed, allowlisted tools with input validation, per-issue/total limits, and a no-progress stop. Never execute document-provided instructions or arbitrary generated code.

Use file/database-backed checkpoints for cross-process review. A pending review is not an approval. Validate review edits, reject stale revisions, and make patching/export idempotent. Remember that resuming an interrupted node may rerun preceding code; do not place unsafe side effects there. Keep state-schema migrations explicit.

Freeze the approved profile/resource versions per run. Cache keys must include relevant source, profile, prompt, model, and resource versions. Separate cached replay from live execution in cost and latency reports.

Do not record or request private model chain-of-thought. Store concise action summaries, evidence, issues, and observable tool results.

## Permissions and cost controls

Offline tests, local fixture runs, ordinary edits within the requested module, and environment-scoped setup allowed by the execution environment may proceed without repeated confirmation.

Live model calls—including configured local/institutional inference services—need an approved run specification identifying endpoint, model, allowed documents, and limits. The presence of an API key is not authorization. A scoped approval remains valid for that run; do not request approval for each call.

Standing user authorization (2026-10-07): live performance testing of the agentic system through the configured ChatAI service using model `qwen3.8-27b` is permitted. The ChatAI API key is stored in the local `.env`; load it without displaying, logging, or committing its value.

- Send at most **10 requests in any rolling 60-minute window**, shared across all runs and workers. Count retries and failed request attempts toward this limit.
- Enforce the limit before dispatch using persistent, shared request accounting; restarting a process must not reset the allowance. If recent usage is unknown, do not assume the full allowance is available.
- Record the configured endpoint, allowed documents/split, and run budgets before testing, and preserve the other data and experiment boundaries in this file. This standing service/model approval does not require repeated confirmation for each qualifying run or call.

Provider-limit verification (requested 2026-10-09): read the configured key's actual rate-limit response headers before adapting the local cap. The [SAIA API documentation](https://docs.hpc.gwdg.de/services/ai-services/saia/index.html#api-limits) describes minute/hour/day/month limits; its example numbers are not this account's verified quota. Keep the existing 10/hour cap pending verification, and retain each run's smaller budget when adapting it. A quota check must send no model prompt or document, use no retries or redirects, retain only allowlisted numeric quota headers/status, and record the administrative API attempt separately from model invocations. Never print credentials or response bodies.

The 2026-10-09 checks received HTTP 401 with no quota headers. Endpoint review confirmed that inference examples use `https://chat-ai.academiccloud.de/v1`, while the exact documented quota GET uses `https://saia.gwdg.de/v1/chat/completions`; both use Bearer authentication. The first check substituted the inference hostname; one corrected check of the exact quota URL also returned 401. This establishes rejected requests, not universal key invalidity. Live work is blocked pending resolved authentication/access and a successful quota check; do not repeatedly retry rejected credentials. The pending one-paper job was cancelled before inference. Preserve its sealed artifacts and shared ledger; prepare a fresh job version after verification instead of resetting history or overwriting the cancelled job.

Default unit/integration tests and CI must not call live models or require network access. Public documentation/resource downloads are separate from model inference and follow normal network permissions; pin and record approved resources.

No external tracing, cloud deployment, catalog writes, data uploads, fine-tuning jobs, or large batch runs without explicit scope and permission. Keep credentials, PDFs, private judgments, and raw run artifacts out of commits. Use environment variables; redact sensitive logs. Enforce these boundaries in code/configuration and deployment permissions, not only in this instruction file.

## Verification and commands

The inherited repository documents `uv sync --frozen` and `uv run python -m pytest`. Inspect tests and scripts for writes/network calls before using them. Preserve the lockfile; update it deliberately when adding necessary dependencies. Do not install into a global or unrelated environment.

After adding `tests/agent/`, use `uv run python -m pytest tests/agent -q` for the new offline suite, plus affected inherited regression tests. Run configured lint/type checks when present; do not claim a tool is configured before checking. Record baseline failures and blocked dependencies separately from new failures.

Do not run the inherited `scripts/run_evaluation.py`, `compute_iaa.py`, `compute_paper_results.py`, or paper builders with their default output locations: they replace released artifacts. Use an isolated offline harness and a fresh run directory instead. Do not run the inherited full extraction experiment as a smoke test.

Before handing off: inspect the diff and artifacts for unintended edits/secrets; record actual commands, outcomes, and skipped checks. Mark mock tests, automated diagnostics, expert judgments, and live measurements distinctly. A model judging its own output is not independent gold.

## Experiment discipline

For an authorized experiment, write the comparison, split manifest, budgets, primary outcome, and evaluator version before running. Change one factor where possible. Use the same evidence access, model configuration, and tools for controller comparisons. Include all calls and failures in resource accounting; unknown cost is unknown, not zero.

Keep natural and synthetic results separate. Report harmed as well as improved records, unprocessed/failed documents, unresolved cases, and human effort. A simulated review is not measured expert time. Bootstrap or aggregate at the publication level when appropriate; never treat chunks as independent papers.

Do not claim a method is novel because it uses agents or standards. Support literature claims with verified primary sources. Negative results and a simpler winning baseline are valid outcomes.

## Task handoff

Report: module and scope; changed files; design and a worked example; exact checks/results; experiment status (`not_run`, `blocked`, `pilot`, or `validated` with evidence); remaining limits; and the next recommended module. Preserve concise progress and decision records, not transcripts. This file is repository guidance for Codex, not the runtime extraction agent's system prompt.
