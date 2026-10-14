# M04 — PaNET lookup and conservative mapping proposals

Date: 2026-10-07. Scope: M04 only; no NeXus profile, controller or live inference.

## Question and minimal design

Can versioned terminology lookup suggest a justified technique concept without
turning a broad mention, related-work citation or valid ontology ID into a claim
that the publication used that technique?

M00–M03 already supply checked paths, immutable source/evidence/record snapshots,
isolated run storage and a separate offline evaluator. There is no standards
lookup implementation to reuse. Reuse these boundaries and leave inherited
extraction, canonical schemas and evaluator semantics unchanged.

Inputs: an authentic pinned local PaNET resource; an exact source mention and
context; optional canonical assertion/revision binding; one or two declared
technique-family roots. Outputs: ranked lexical candidates and an unapproved
mapping proposal, explicit unresolved/no-match states and resource/evidence
provenance in a sidecar. Preserve the original source expression. Ontology labels
and hierarchy describe terminology, not publication evidence.

Load the official RDF/XML using the standard library; do not add a dependency or
download imports at runtime. Preserve labels, available synonyms/alternative
labels, named parents, definitions and deprecation metadata. Named-parent lookup
is not OWL reasoning. Do not pretend an alternative label is an asserted exact
synonym or promote a broad parent mention to a more specific child.

Deterministic lexical lookup is the baseline. An exact unambiguous label may
produce a proposal only when supplied context describes current-study use;
context annotations remain unreviewed. Related-work/uncertain contexts and fuzzy
matches stay unresolved. A proposed mapping is never an expert approval and does
not alter canonical benchmark JSON. The optional LLM selector is deferred; no
model call is necessary for this module's implementation checks.

## Resource and verification plan

Use the official `pan-ontologies/PaNET` repository, an immutable commit and a local
SHA256-checked artifact under ignored run storage. The accessible v1.1.0 generated
OWL is pinned to commit `6d23ae3ea9ab62f4e8be97cc38c2ebdf0367c142`. A newer v1.3.0
release exists, but its generated release asset was not obtainable through the
available download route. This selected historical snapshot is not called latest.
Record release identity separately from the ontology's own version annotations,
and retain upstream attribution and the stated CC-BY 4.0 resource terms.

Offline checks: actual resource integrity/loading, synthetic labels and synonyms,
named hierarchy and ambiguity, unknown/obsolete IDs, related-work-only mention,
no suitable concept, corrupt/unavailable/version-mismatched resource, bounded
lookup, stale evidence/record bindings and cache invalidation. Default tests do
not access the network. Synthetic test namespaces must remain clearly synthetic.

E04 comparison (not yet run): no normalization versus lexical lookup versus lookup
plus contextual selection, with extraction fixed. Candidate recall, exact justified
mapping accuracy, over-specific errors, unresolved cases and cost require an
expert-reviewed mention-to-concept dataset. No such judgments are supplied here;
do not invent them or repurpose extraction gold as terminology truth. Synthetic
engineering checks are reported separately from empirical mapping performance.

## Implementation and worked example

Added four modules under `src/meta_data_extraction/agent/`:

- `panet_resources.py`: a typed, SHA256-pinned RDF/XML projection with a bounded
  parser, immutable snapshot identity, no entity expansion or fetched imports.
- `panet_lookup.py`: bounded deterministic label/alias matching inside explicit
  named-parent family scopes. Lexical scores are not confidence or evidence scores.
- `panet_mapping.py`: exact source/context references, optional stable record/
  revision binding, and proposed/unresolved/no-match states. Selected IDs must be
  active retrieved candidates; they cannot override ambiguity or context gates.
- `inspect_panet.py`: strict offline configuration, authentic-resource-only CLI,
  isolated sidecars and verified replay. It fingerprints resources, source, record,
  configuration, code and dependencies. It never exports a changed canonical record.

Added four test modules, one invented source fixture, three agent configs, the
resource provenance note and updated progress/command/data-policy records. The
downloaded OWL and license stay in ignored run storage. No existing extractor,
provider, scorer, canonical schema, lockfile or package identity was changed.

The selected resource loads **374 named concepts**. Its byte SHA256 is
`ef4dbea93c9e16883114fb1fc6132fc79797382351d8796d0daf827a1b413ddb`; its Git blob hash
matches the official commit tree. See [resource provenance](../PANET_RESOURCE.md)
for exact upstream links, license/attribution, parser limits and the reason for
using this historical version. Snapshot ID:
`panet:b827b6f3d8e4affca1f3609a77b1aa056fc0cb51706057d139bb434eabe8d041`.

The demonstration scopes powder diffraction (`PaNET01030`) and small angle
scattering (`PaNET01124`). In the invented source, current-study x-ray powder
diffraction yields an unapproved `PaNET01101` proposal. The same concept's `XPD`
alternative label in related work remains unresolved. A generic powder-diffraction
mention retains the broader `PaNET01030` proposal rather than acquiring an x-ray
or neutron qualifier. This tests the policy, not the correctness of real-paper
mapping judgments.

The separate M02-bound example points to `/datasets/0/measurement_technique` and
its existing stable entity handle/record revision. The source expression remains
`X-ray diffraction`; within the chosen two-family scope there is no exact match,
so overlapping powder techniques do not become an assignment. No canonical bytes
are changed.

Main trade-off: exact lexical matching plus supplied context is intentionally
conservative and does not resolve semantic synonyms or missing hierarchy links.
All context annotations are marked `supplied_unreviewed`; all mapping decisions
retain `approval_status: unreviewed` and `support_status: not_assessed`. No approval
workflow is implied. If a result limit could hide another exact match, the mapping
stays unresolved. If verified ranking proves every omitted candidate is only an
overlap, a unique exact candidate can still be proposed.

## Checks and outcomes

- **168 M04 tests passed in 5.37s**: resource 34, lexical lookup 44, mapping 37 and
  runner 53. Cases include wrong/stale hashes, forbidden paths, DTD/entities,
  synonym kinds, hierarchy scope, obsolete/unknown IDs, ambiguity, context gates,
  exact evidence resolution, stale bindings and cache tampering even with recomputed
  storage checksums.
- Final combined safe suite: **798 passed in 50.64s** (572 agent, 151 offline and
  75 inherited regressions). The first full invocation exceeded the tool's 30-second
  output window, so no result was claimed; the longer captured invocation passed.
- Authentic-resource CLI: six invented mentions produced **3 proposals, 2
  unresolved contexts and 1 no-match**. Repeat execution returned `cached_replay`.
- Record-bound CLI: one explicit `lexical_only` unresolved decision, retaining the
  M02 entity/revision binding and source wording.
- All runs used **zero model calls**, and the real `.env` was not read. Unit tests
  deny sockets/DNS. Resource/lookup/mapping unit cases use synthetic namespaces;
  runner unit tests use an explicitly mocked loader, while the CLI integrations
  use the actual downloaded ontology.

Exact commands and blocked attempts are in [COMMANDS.md](../COMMANDS.md). Final
artifacts are `request.json`, `result.json` and `sidecar.json` in:

- `runs/agent/m04-20261007-lookup-v1/`
- `runs/agent/m04-20261007-bound-v1/`

No canonical output is emitted. Completed replay validates the whole report and
sidecar against fresh source/resource/record inputs, including metadata such as
source incompleteness and call counts. Resource bytes remain separately pinned
under `runs/agent/m04-20261007-panet-v1/`.

Implementation: **ready**. Offline/integration checks: **pass**. E04 empirical
status: **blocked** pending an expert-reviewed mention-to-concept dataset. No
natural-paper mapping accuracy, candidate recall, cost advantage, over-specific
error rate or controller improvement is claimed. The optional LLM contextual
selector is not implemented or exercised; deterministic context gating is tested.

## Remaining limits and next dependency

The historical ontology projection is not a complete OWL reasoner or the latest
release. Only explicit named relationships and English/unqualified annotations
are used; hierarchy gaps and missing resources produce explicit limitations or
failures. Existing extraction gold is not terminology adjudication. Public resource
download permission and a valid concept ID do not establish mapping approval.

Full environment synchronization remains blocked by missing cached `hatchling`;
the existing checkout-bound environment passed without dependency changes. No
lint/type checker is configured. Gold-reading inherited suites, release regeneration,
live inference and NeXus work were not run. Protected baseline hashes and M02 runtime
fingerprints remain unchanged; final diff, secret-pattern and artifact checks are
recorded in the commands file.

Stop after M04. Next dependency: **M05 — NeXus definitions and a reviewed profile
registry**, beginning with a small pinned definition set and explicit mapping
proposals; reviewed status still requires actual review.
