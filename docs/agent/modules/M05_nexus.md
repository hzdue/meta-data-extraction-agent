# M05 — NeXus definitions and a reviewed profile registry

Date: 2026-10-07. Scope: M05 only; stop before M06. Implementation and offline checks complete; reviewed-profile gate blocked.

## Question and starting point

Can a small pinned set of NeXus definitions clarify existing field meanings without
encouraging unsupported values or changing the inherited canonical schema?

M00–M04 provide safe run paths, source/evidence and record snapshots, frozen inherited
profiles, fixed extraction, isolated offline evaluation, and conservative PaNET
mapping proposals. Reuse those contracts. There is no NXDL loader or reviewed NeXus
profile registry to reuse. Keep extractor/evaluator behavior and package identity fixed.

## Interfaces, design and baseline

Inputs: inherited profile; a small authentic commit/byte-pinned local NXDL snapshot;
explicit mapping requests; optional supplied content-bound review records. Outputs:
class/field-context definitions, type/units as stated, unresolved dependency notices,
and proposed mappings or reviewed guidance in separate sidecars. All new proposals
remain unreviewed; no expert approval is supplied in this checkout.

Use a bounded standard-library XML projection without network/entity resolution.
Base-class possibilities and application requirements stay distinct. Required values
are never filled from standards. Class/field references are definition addresses,
not asserted universal HDF5 paths. Unhandled includes, inheritance and group types
remain explicitly unresolved. Do not generate schemas or export a NeXus file.

The registry binds inherited field instructions, NXDL context/version, mapping scope,
relation and semantic role. Keep preparation and acquisition temperatures distinct.
Default guidance retrieval excludes unreviewed proposals. Review records bind an
exact entry revision and reviewer/version; their real-world authority is an external
input, not something Codex can create. Freeze all inputs and code in isolated runs.

Explicit Decimal conversion utilities preserve raw expressions and precision metadata;
conversion never edits canonical values and does not infer absent units or evidence.
The baseline is unchanged inherited instructions. NeXus guidance is opt-in and is not
wired into extraction before mapping review and an E05 comparison specification.

## Required checks and experiment boundary

Offline fixtures: preparation versus acquisition temperature, same quantity/different
roles, incompatible units, missing required experimental data, definition/version/context
changes, no match, malicious or unavailable XML, stale/forged review bindings, exact
conversion and precision/raw preservation, safe run storage and verified replay.
Use the authentic local resource for a separate deterministic integration inspection;
keep synthetic diagnostics distinct from natural-paper outcomes.

E05 compares existing instructions with the same instructions plus selected verified
definitions, holding extractor/evidence/PaNET configuration fixed. Outcomes require
independent interpretation/support/harm labels and a frozen comparison manifest. No
reviewed mapping profile or these labels are supplied: E05 is blocked, not validated.
No live inference, real .env read, gold inspection or released-result regeneration is
needed for M05 engineering checks. Standing ChatAI authorization remains in AGENTS.md.

## Outcomes

Implementation: **ready within the declared projection scope**. Offline checks and
authentic-resource integration: **pass**. Reviewed-profile gate and E05: **blocked**
by unresolved relevant NXDL dependencies/defaults and absent actual mapping reviews;
E05 additionally needs a frozen comparison scope and independent interpretation,
source-support and harmful-correction labels. No performance gain is claimed.

## Implemented behavior and worked example

Four new modules reuse existing path/profile/model/storage contracts:

- `nexus_resources.py`: explicit local NXDL file allowlist, hash/commit/license
  pins, bounded XML declaration projection and exact-context definition retrieval.
- `nexus_registry.py`: content-bound field mapping entries, partial/exact/equivalent/
  no-match distinctions, typed semantic roles and supplied review records.
- `nexus_units.py`: explicit versioned Celsius/Kelvin Decimal arithmetic, separate
  absolute/difference roles, original expression and reporting-resolution retention.
- `inspect_nexus.py`: strict offline configuration, authentic-resource-only
  inspection, immutable sidecars and full recomputation when verifying replay.

Added four focused test modules, resource/inspection configs and resource provenance
documentation. The existing parser, provider, profile, PaNET, extractor and evaluator
implementations remain unchanged. No new dependency or canonical field was added.

The authentic selected resource is NeXus v2026.01 at commit
`0ad0f784950eefea5675e725d2ecb10a5f83e8ad`: NXsample (39 fields), NXbeam (32) and
NXmonopd (14). Upstream LGPL-3.0-or-later statements, attribution and license text
are retained with the unmodified downloaded files. See [NEXUS_RESOURCE.md](../NEXUS_RESOURCE.md)
for exact source URLs and byte/Git-blob hashes; this is a subset, not a full release
installation or standards-compliance validator.

A concrete example: preparation temperature belongs only to the preparation aspect
of the inherited sample description; acquisition temperature belongs only to an
acquisition aspect of dataset description. Both are free-text fields, so a
quantity definition can only supply partial context. NXsample temperature cannot
be relabelled as preparation temperature. The actual config records no match for
preparation within the selected subset, and an unresolved partial acquisition
proposal. Both stay unreviewed; identical units do not merge their roles.

For an invented arithmetic example, `25.0 °C` becomes the exact arithmetic value
`298.15 K` with the source expression and reporting resolution `0.1 K` retained.
The calculation does not claim improved measurement precision, source support or
a canonical correction. Missing values/units remain unavailable; incompatible
units and unsupported expressions are rejected explicitly.

The main trade-off is a small declared-field projection instead of a full NXDL
processor. It keeps local definitions inspectable but leaves inheritance, schema
defaults and unhandled constraints explicit. The selected real targets remain
unresolved, so their approval gate also needs the relevant dependency/default
resolution, followed by actual mapping review. Synthetic tests use completely
explicit invented declarations to exercise approved/rejected/stale review paths.

Review artifacts are supplied local inputs, pinned by bytes and entry content.
The code checks freshness and consistency; it does not authenticate human identity.
No production review is created here. Default guidance remains empty while entries
are unreviewed or unresolved. Neither reviewed definitions nor required application
fields would establish a publication fact or authorize filling absent values.

## Checks and final outcome

- **286 M05 offline tests passed in 14.10s**: resources 68, registry 64, units 94,
  runner 60. They cover explicit semantic roles, representation/unit incompatibility,
  missing required experimental data, version/context/review changes, XML and
  projection bounds, stale/forged artifact rejection, safe output paths and replay.
- The full selected safe suite passed **1,084 tests in 52.70s**: 858 agent,
  151 offline and 75 affected inherited regressions. Default tests deny network;
  synthetic review records test mechanics and are not expert approvals.
- Actual-resource inspection under `runs/agent/m05-20261007-inspection-v1/`
  completed, then returned `cached_replay` with all three artifact byte hashes
  unchanged. Files: `request.json`, `result.json`, `sidecar.json`; no canonical
  or raw model-output artifact was generated.
- Six mapping entries: four unresolved partial proposals, two scoped no-matches,
  all unreviewed; default approved guidance count is zero. Three exact definition
  lookups retained base/application context. Four invented quantity checks yielded
  two exact conversions, one unavailable value and one incompatible-unit rejection.
- All resource byte/Git-blob hashes and baseline protected assets were checked.
  Existing source modules, dependencies, lockfile, canonical schemas, evaluator
  configuration and released results remain unchanged. Credentials were not read.

Final resource snapshot:
`nxdl:8a779d11868e2f94d45b0cc6d24285c0cf157c1d035ee1bf58b1bcee7e54bc3b`.
Registry snapshot:
`nexus-registry:7a0c233b7f1ecc650c5f310372a62f785bedd7f702a54c20d10c3c2fd686307a`.
Exact commands, host-tool workarounds and final audit are in [COMMANDS.md](../COMMANDS.md).

There were **zero live model calls** and no real `.env` read, natural-publication
experiment, gold scoring or result-regeneration command. No lint/type checker is
configured; full environment synchronization still lacks cached `hatchling`. The
existing checkout-bound environment was used without dependency changes.

Stop after M05. Next coding module: **M06 — record checks, issue diagnosis and
supported patches**, starting with its independent structural/evidence checks.
NeXus-informed runtime guidance remains disabled until the selected definitions'
relevant dependencies/defaults are resolved and a real content-bound mapping review
is supplied. Full NXDL processing, schema generation, NeXus export, M08 review
workflow and empirical E05 performance validation remain outside this completed
engineering scope.
