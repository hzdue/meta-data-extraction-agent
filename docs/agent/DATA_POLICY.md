# Agent data and execution boundary

M00–M04, 2026-10-07. This applies to the additive `meta_data_extraction.agent` path;
the inherited benchmark tools retain their historical behavior.

| Role | Permitted access |
|---|---|
| Publication source | Explicit individual files staged in ignored `data/raw/`; no directory scan, glob, corpus-driven selection, automatic download, or repository index |
| Synthetic test input | Explicit files under `tests/agent/fixtures/`; fixture text must be invented and labeled |
| Canonical schemas | Four named inherited files under `schemas/`; meanings, identity rules and missingness remain unchanged |
| Agent config | Strict offline YAML under `configs/agent/`; duplicate and unknown keys rejected; live M02 entry requires a separate typed scope |
| New artifacts | Fresh `runs/agent/<run_id>/`; ignored by Git; exclusive claim, verified completed replay only, no automatic retry of interrupted runs or overwrite |
| Offline-only assets | `data/gold/`, `data/annotations/`, `data/experiment/`, `data/few_shot/`, `data/corpus.json`, inherited experiment/evaluation configs, `outputs/`, `paper/` |
| Standards | M04 accepts one explicitly pinned local PaNET RDF/XML resource from the official repository; no dynamic imports, network lookup or NeXus loading |

Source/config selection is an allowlist, not a configurable repository root for
retrieval. Absolute paths, traversal, Windows device/alternate-stream names,
symlinks, junctions and hard-linked input files are rejected. `RunDirectory`
rechecks paths before writes and opens artifacts exclusively. It provides no
release-write override. An offline evaluation harness must be separate and use
fresh run storage too; M00/M01 do not create that harness.

Runtime must not import `evaluation`, `alignment`, or evaluation `normalization`,
or consume their aliases/judgments. The tests inspect the transitive local import
closure of the agent package, and reject gold/helper/result paths in source and config fields.
The reused strict YAML reader is generic and does not read evaluator files by
itself. The inherited schema utilities also contain few-shot functions, but M00
calls only schema loading/version functions with validated schema paths.

This is an application boundary, not an OS security sandbox. It cannot classify
a gold file deliberately copied into a source directory, prevent unrelated code
from opening files, or defeat a concurrent adversarial filesystem replacement.
Keep future runtime credentials/deployment mounts restricted to approved sources,
schemas/resources and its run directory; do not mount benchmark judgments.
The current local checkout necessarily contains both runtime and offline assets.

M00/M01 have no model execution path. M02 adds a separately scoped ChatAI provider;
the CLI remains mock-only, and agent tests deny socket connections and DNS.
Standing ChatAI authorization is recorded in `AGENTS.md`: model `qwen3.8-27b`,
shared maximum ten requests per rolling hour including failures/retries. It has
not been exercised. Live callers bind the trusted configured endpoint, exactly
allowed source hashes, profile and rendered prompt hashes, request/output/timeout
budgets and explicit retention before dispatch. Credentials load only after ledger
readiness, with no logging. User/model/document-provided endpoint guesses do not
establish approved ChatAI configuration.

The fixed ledger at `runs/agent/_control/chatai-requests.sqlite3` coordinates this
checkout's workers and runs. A new ledger waits a full hour on wall and monotonic
clocks; completed attempts remain charged for an hour after completion on both
clocks. Unfinished reservations remain charged indefinitely. Clock regression or
an incompatible ledger schema fails closed, requiring explicit reconciliation.
Deleting accounting files does not give immediate fresh allowance. Other clients
and checkouts are outside this local enforcement and must share coordinated
accounting. No external tracing, catalog writes or deployment is enabled.

M01 preparation reads only explicit allowlisted files, preserving raw-byte hashes
and exact UTF-8 text or inherited PDF-to-Markdown output. Retrieval accepts only
the supplied immutable bundles; its source filters can narrow, never expand, that
scope. Text instructions and URLs are data, not actions. The PDF path is local
conversion, with no automatic publication fetch or OCR service. Page/image
evidence is explicitly unavailable; plain text never supplies inferred pages.

Gold contents and private judgments were not inspected for M00 or M01. File counts,
the corpus split manifest and source/config hashes serve the audit only. Existing
predictions are available for a separately authorized offline replay, not runtime
evidence. Paper-specific alignment aliases do not establish new-paper coverage.

Raw text, PDFs, raw responses, checkpoints and private judgments stay out of
commits. M00 stores only synthetic input hashes/byte counts and schema diagnostics.
M01 inspection writes text snapshots, source/provenance hashes, warnings and
evidence quotations under ignored `runs/agent/`; it does not emit canonical
extractions. M02 requires an explicit raw-retention choice: retained responses go
only to `raw-response.json`; otherwise only response hash/status persists. Its
run directory also holds the prepared source, frozen profile, canonical export
when valid, sidecar, request and completion manifest. Invalid/truncated model
output cannot become a successful empty extraction. Unlinked or invalid evidence
requires review; a resolved quotation proves location only. Evidence/status and
internal handles stay in sidecars. Review storage/approval is a later module.

`LICENSE` is empty. Licensing/redistribution is unresolved; M00 neither changes
attribution nor grants permission or makes a legal determination.

## M03 offline scoring exception, separate from runtime

`meta_data_extraction.offline` may read explicitly hash-pinned gold, predictions,
frozen evaluator configuration/helper and reference reports during the authorized
offline scoring step. Runtime imports of that package are forbidden and tested.
Its manifests never widen runtime source access. No directory discovery, model
call, global helper rewrite or result-regeneration script is part of the harness.

Gold comes only from `data/gold/` or explicitly synthetic `tests/offline/fixtures/`.
Predictions may come from released `data/experiment/`, isolated `runs/agent/` or
synthetic fixtures. Reference reports are read-only under `outputs/evaluation/`,
isolated runs or fixtures. Manifest role allowlists reject paths to unrelated
inputs. Copies and new reports stay in a fresh ignored run directory. Both originals
and staged copies are checked when a completed run is replayed.

Publication IDs, normalized DOI/arXiv aliases, source hashes and declared
derivatives fix the analysis group. All derivatives share its split; one record
per group/variant is scored. Shared gold/prediction/sidecar/report paths cannot
belong to unrelated groups. The manifest cannot discover unstated publication
relationships. Natural and synthetic data are reported separately; future-held-out
gold is gated before read and is not authorized merely by appearing in a manifest.

M03 E03 froze **1804.07163**, `pan_gpt4o_full_0shot` and
`pan_gpt4o_no_inst_0shot` before scoring. Only that historical gold record and the
two corresponding predictions/reports were used in the natural replay. Gold
contents were consumed by the offline scorer, not displayed in conversation or
used to alter runtime extraction. This is historical reproducibility, not an
untouched confirmatory set. Future split approval and new-paper adjudication are
separate work; unavailable identity judgments block affected comparisons.

The scorer does not assess evidence support or completeness. Entity recovery,
conditional attributes, references and document fields keep their inherited
definitions, with support shown separately. Unknown accounting remains null, and
live, mock and cached execution are separated. No private gold/annotation contents,
new metric reports or staged natural inputs belong in commits.

## M04 terminology boundary

The PaNET resource pin is separate from publication-source selection. It identifies
an official commit URL, local ignored run path, exact SHA256, namespace, release
label and verified resource-license annotation. Raw OWL and attribution/license
text stay under `runs/agent/m04-20261007-panet-v1/`. Runtime performs no download or
XML import resolution; DTD/entity declarations are rejected. The chosen v1.1.0
historical snapshot is documented separately from newer upstream releases.

Synthetic ontologies use `urn:synthetic:` names under `tests/agent/fixtures/` only;
the inspection CLI requires an authentic resource declaration. The resource pin
is trusted configuration whose upstream provenance was checked when acquired,
not an independent certificate. Runtime accepts only candidates from the loaded,
versioned resource and explicit family scope. Synonym/alternative-label kinds are
preserved, and incomplete named-parent projection does not imply OWL entailment.

Technique mappings are sidecars, never canonical replacements. Source mentions
and context must resolve to actual hashed source regions. Optional canonical
bindings address an exact dataset measurement-technique value and immutable record
revision. Context roles are caller-supplied and explicitly unreviewed. Exact lexical
matches can produce proposals; related-work, uncertain use, hidden exact ties,
obsolete/unknown IDs and unsupported specific selections remain unresolved.

Every decision remains `approval_status: unreviewed` and `support_status:
not_assessed`. Neither a standards definition nor a supplied context label proves
an experimental fact. No expert terminology judgments or mapping approvals were
invented for M04; its demonstrations use invented publication statements. The
existing ChatAI authorization/limit remains unchanged and was not exercised.

## M05 definitions and profile reviews

NeXus NXDL is representation guidance, never publication evidence. The selected
authentic files and upstream license/attribution remain under ignored
`runs/agent/m05-20261007-nexus-v1/`; the versioned manifest pins exact bytes and an
official repository commit. Runtime loads only named local files and performs no
network, include download or schema generation. Class/field definition addresses
are not asserted to be universal HDF5 paths. Unresolved dependency/default/shape
limitations remain visible; this projection does not validate NeXus compliance.

M05 mapping entries preserve the inherited profile/field instructions, the NXDL
context, semantic role, relation, scope and content version. New entries are
unreviewed. Default guidance retrieval returns only entries with a supplied review
decision bound to the exact entry content; unavailable or stale reviews cannot
approve anything. The local review artifact also has an explicit byte hash and
remains under ignored run storage. These hashes verify content, not reviewer
identity or authority. Production approval requires an actual authorized human
review; only synthetic test cases simulate those records. No such production
review is supplied or created during M05. A full review workflow remains M08.

Inspection writes only request/result/sidecar artifacts and leaves the canonical
schema, extraction prompts and records unchanged. Approved guidance would still
carry publication support and application compliance as not assessed. Required
experimental values are not synthesized from standard fields. Missing values and
units remain unavailable. Preparation and acquisition temperatures retain separate
roles, including when they use identical units.

The optional deterministic temperature utility accepts explicit bounded decimal
strings and units; it does not extract them from text. It preserves the original
expression and reporting resolution beside the exact arithmetic result. This is
neither an evidence check nor automatic canonical normalization. All quantities
in the M05 inspection config are invented engineering inputs, not measurements.

No real .env, model endpoint, publication gold, evaluator judgment or release
regeneration participates in M05. E05 requires a reviewed mapping subset and an
independently labelled comparison scope before a performance claim.

## M06 checks and bounded corrections

Structural diagnostics keep invalid submitted canonical data, including unexpected
fields and dangling references, in a separate candidate sidecar. They do not repair
records silently or turn invalid records into successful empty extractions. The
original profile, source hash, representation and candidate revision bind each report.

Source-support checks consume only selected evidence from the supplied source and
the frozen canonical profile/record. Prompts include linked entity context; standard
definitions and offline evaluator/gold resources are excluded. A verified quotation
establishes location, while supported/contradicted/insufficient-evidence verdicts
remain explicitly unreviewed model judgments. Mock outcomes are synthetic diagnostics;
they are not measured natural-paper performance or independent labels. The coverage
layer is unsupported: checking emitted assertions cannot establish entity completeness.

The synthetic inspection CLI accepts only explicitly hash-pinned test-fixture inputs.
It cannot construct a live provider. The separate checker API reuses the existing
ChatAI exact-scope/provider/accounting boundary; the real .env is not read in M06.
Each checker call handles a bounded assertion set without automatic retry. Failures,
truncation, unavailable evidence and unknown usage remain explicit. Raw responses
are retained under ignored run storage only when the config says retain; hash_only
exports no raw output. No private chain-of-thought is requested or stored.

Patch application needs an explicit local policy, expected revision and old values,
source-linked contradictions for changed assertions and supportive candidate judgments.
The supported M06 policy permits only dataset descriptions, techniques and individual
sample references. All sample/facility attachments of touched datasets are rechecked.
Conflicting source context is left for review. No entity IDs, sample identity fields,
array topology, merges or splits can be edited by this policy. Model support is not
expert approval or a guarantee that a correction is scientifically right.

Canonical corrections are emitted only to a fresh ignored run, never written back
to input records or released results. All operations apply together or none do.
A receipt binds original/candidate revisions, policy and support reports; completed
replay checks exact artifacts and makes no new provider calls. Evidence from the
old revision is invalidated and new assertions get fresh bindings, with unchecked
assertions explicitly unlinked. Old support cannot silently survive record edits.

E06 remains blocked pending independently labelled real outputs and a frozen
comparison manifest. No natural gold, adjudication, live inference or released
result-regeneration step was used for this module. M05's approved-guidance gate
remains blocked and was not bypassed.


## M07 workflow, checkpoint and review boundaries

The single local workflow accepts only typed, allowlisted actions and frozen issue,
source, profile, provider and correction catalogs. Policy choice cannot enlarge
resource budgets or source access. No repository-wide retrieval, evaluator input,
external tracing or document-provided executable instruction is permitted. Generic
selection and source-support checks use the existing provider boundary; actual
synthetic CLI runs only construct explicit mock providers. ChatAI standing limits
remain enforced by the existing shared request ledger for any separately scoped
live use; M07 performed no live experiment.

Publication excerpts and approved standards guidance occupy separate typed fields.
Source references resolve against the actual supplied bundle. Definitions cannot
be used as publication evidence. A record revision invalidates earlier issue
resolutions unless a fresh support report binds that exact assertion to the new
revision. Other open/resolved issue states are explicitly marked stale for a fresh
check/review. Existing pending cases retain their original revision and are never
approvals. M08 must validate later human responses and handle stale cases explicitly.

New state uses version m07-state-1, real SQLite LangGraph checkpoints, a separate
content-bound dispatch journal and an OS-held per-run lease. Completed steps can
replay without dispatch. An incomplete dispatch claim stops with uncertain possible
attempts; automatic resumption cannot reset the call allowance or repeat a request.
Persisted deadlines survive process restarts. Source/provider failures, unknown
usage and unprocessed issues remain visible. Known token subtotals with an unknown
flag are not complete totals. Provider invocations, reserved live attempts and
unknown dispatch outcomes are reported separately.

The M07 fixture runner uses explicit hash-only raw-response retention. It writes
only ignored runs/agent/<run_id>/ artifacts: checkpoints, journal, immutable
request/result/canonical/sidecar exports and a local lock. No expert approval or
completeness claim is inferred from finished control flow. File hashes and duplicate
state cross-checks provide integrity checks, not authentication of an untrusted
filesystem. No silent state-schema migration or interrupted-call retry is supported.

## M08 review continuation

A completed pending M07 run may gain a versioned review/review.sqlite and immutable
review/exports/<state-digest>/ artifacts within the same ignored run directory.
Original top-level canonical data, request, completion, dispatch journal and
LangGraph checkpoints are frozen and checked against their recorded hashes. A
review export contains the current canonical record plus a sidecar retaining the
original workflow, explicit responses, revisions, source-bound attestations and
original usage accounting. No review stage constructs or invokes a model provider.

Review responses are individual explicit JSON inputs under that run's
review-responses/ directory; tests may use tests/agent/fixtures/. They must declare
simulated or user_supplied provenance. Reviewer labels are self-declared and never
authenticated expert identities. A pending case, generated response or model
judgment is not human approval. All checked-in M08 policy fixtures and demonstrated
review responses are synthetic; private real reviews/raw artifacts stay ignored.

Approved cases cover only selected assertions and required attachment context.
Canonical schema/local IDs remain unchanged. A new revision invalidates old
assertions, and explicit unresolved dependencies cannot retain conflicting active
approvals. Exact response replay is idempotent. Exports recover from committed
SQLite state without redispatch or replacing completed artifacts; only private
unfinished staging bytes may be rebuilt. Changed schema/code/runtime/resource
bindings fail closed, without automatic migration. No reviewer-time or residual-
error result exists until a separately frozen real-review experiment is run.

## M09 study and offline collection boundaries

The runtime study specification lives under configs/agent and contains only an
opaque digest of its evaluation protocol. It pins the explicitly supplied source,
record, precomputed issue judgments, profile, prompts, policy fixtures, tools and
implementation before creating any child. The CLI accepts synthetic repair-policy
execution only; it cannot construct a live provider or read the offline protocol,
gold, alignment helper or credentials. Study manifests, children and reports use
fresh ignored runs/agent directories. Completed replay is read-only and dispatches
zero calls. Failed cells remain explicit, with uncertain attempts reserved against
their cap; unknown tokens are not reported as zero total usage.

The separately frozen protocol under configs/offline binds explicit synthetic gold
and an evaluator snapshot. Only the offline collector opens these inputs and calls
the unchanged M03 scorer. It checks each child's exact policy, cap, source, profile,
revision, artifact inventory and resource observations before scoring raw canonical
predictions. Missing cells remain unprocessed; budget-exhausted cells with a valid
record retain that record and their unresolved workflow status. There is no silent
selection of M08 review exports or authenticated review evidence.

Quality remains per field with gold support; entity recovery, conditional
attributes and references are reported separately. Matched repeats are averaged
within each publication before paired publication-level aggregation. One publication
has no uncertainty interval. Mapping correctness, source-support errors, harmful
corrections and actual review time remain unavailable without independent labels
or observations. The synthetic fixture's expected record was authored and frozen
before execution; it is an engineering expectation, not independent natural gold.
This module does not establish end-to-end extraction, standards benefit, live
latency/cost or the controller hypothesis. E09 remains blocked by its recorded
empirical prerequisites; the ChatAI permission and shared ten-request/hour limit
in AGENTS.md remain unchanged and were not exercised.

## M10 local service and packaging

The pilot runs a fixed invented source and deterministic mock review workflow in
an isolated, hash-bound bundle. Only 64 explicit assets are copied; evaluation
modules/configuration, gold, benchmark outputs, paper files, credentials, test
code and previous raw runs are absent. Production startup verifies that loaded
project modules and their package search paths belong to the pinned bundle.
The immutable assets are rechecked for every backend operation; only that
bundle's runs/agent subtree may change. A local filesystem owner remains outside
this application boundary. The unbuilt container recipe adds a non-root user and
read-only application assets, and must not receive a repository mount.

The API binds only 127.0.0.1, requires an independent pilot token for every data
route, and rejects foreign Host/Origin, ambiguous framing/JSON, arbitrary paths
and oversized input. It has no document upload, provider/model selection, resource
download, evaluation, external tracing or catalog-export route. Its token is read
from META_AGENT_PILOT_TOKEN; the real .env and ChatAI credentials are not used.
The public static page has no private data or external assets; its access token
is retained only in page memory. Structured request logs omit tokens, run IDs,
source/review text and raw paths/bodies. Explicit review content is retained in
the private SQLite review store as required for audit and idempotency.

One operator and a shared OS-held lease are supported. Run/storage/input/path
limits apply before mutation. M07 checkpoints, uncertain-dispatch stops and
persistent deadlines are reused; M08 exact-state review, stale-revision rejection,
transactions and immutable exports are unchanged. Download returns canonical JSON
with its review sidecar/manifest and requires the displayed state ID. Exporting
pending, unresolved or simulated decisions cannot confer expert approval.

All demonstrated review responses are explicitly simulated. M10 introduces no
new canonical fields, expert authentication, natural-paper evidence, user-time
measurement or live inference approval. M09 offline collection still requires
its separate frozen inputs and does not silently select pilot review exports.
E10 usability/effort evidence remains not_run pending an actual operator pilot;
the earlier empirical gates remain unresolved. Optional M11 training is not
authorized by this implementation.
