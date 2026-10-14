# M10 operator trial: fixed synthetic example

Protocol version: `m10-operator-trial-1`, prepared 2026-10-08. **Status: not_run.**
This document is a procedure and blank record, not a completed human study. It
uses the M10 pilot described in [PILOT.md](PILOT.md). A person may measure their
own operation of this invented example; an automated browser/API run remains an
engineering diagnostic and cannot fill the human observation fields.

## Frozen scope and outcome

Question: can one operator inspect evidence, record uncertainty, recover a stale
view and restart, and retrieve the correct saved record without a developer
repairing files? The baseline is the existing documented CLI/API capability; this
session has no randomized comparator and cannot establish a speed improvement.

Use one fresh verified bundle, its fixed `workflow_review.json`, invented
`m06_source.md` and `m06_record.json`, and the bundled mock responses. No other
publication, gold, evaluator or alignment helper is permitted. Record the bundle
manifest SHA256 before starting. The exact source/record/configuration hashes are
already in its manifest. No runtime resource, prompt, policy or schema changes
are allowed during the session. No accuracy scorer is used: this is a task
completion checklist, not independent scientific gold.

Reserve at most two child runs (`m10-u1` for core tasks and `m10-e1` for the
optional edit), with the existing limit of eight mock policy calls per run,
12 actions and a 120-second workflow deadline. A normal complete start uses four
mock policy calls; refresh, review and export do not make new model calls. This
trial allows **zero live requests**, zero network inference and no uploads. Set a
45-minute session cap and a 10-minute optional edit cap before starting; these
are budgets, not observed times. Preserve incomplete or failed runs at the cap.

Primary outcome: completion of each core task below, classified as `unassisted`,
`assisted`, `failed` or `not_attempted`. Reading this protocol and PILOT.md counts
as the documented baseline. Any additional instruction or developer action is
assistance and must be recorded. Secondary observations are actual elapsed time,
pauses, errors/retries, recovery success and the optional correction task outcome.
Record every attempted task, including failures. Never substitute an automated
run duration, estimate, or memory of a previous session for observed human time.

## Setup and observation record

The user selected preparation for a later trial. The already prepared bundle is
`runs/agent/m10-trial-ready-v1/runtime`, with 64 pinned assets and manifest SHA256
`9cb0d2a6b213cc6d1d23417915ecaf3d8d5a03c433d7761eb98cfe87bca48ff5`.
Preparation did not perform a human session or browser interaction. When ready,
run this command from the checkout root; it verifies the existing bundle and
starts the loopback service using the existing environment:

```powershell
.\.venv\Scripts\python.exe scripts/agent/start_pilot.py --bundle-id m10-trial-ready-v1 --resume-bundle
```

For a separate fresh trial instead, choose a new short ID such as `m10-t1` and
run `.\.venv\Scripts\python.exe scripts/agent/start_pilot.py --bundle-id m10-t1 --port 8765`. Omit
`--resume-bundle` only when creating a new bundle; existing directories are never
overwritten.

The default token mode copies a temporary local access token to the clipboard.
It is unrelated to ChatAI and must not be included in notes, screenshots or
exports. `--prepare-only` performs preparation without starting the listener;
`--resume-bundle` explicitly reuses a verified existing bundle. The alternative
`--token-mode environment` uses an already supplied private local pilot token in
`META_AGENT_PILOT_TOKEN`; never substitute the ChatAI key. See the launcher's
`--help` or [PILOT.md](PILOT.md) for the full manual setup and recovery procedure. Do
not modify the pinned guide or add observation files inside the immutable
`runtime` directory.

Before opening the page, copy the blank record below to
`runs/agent/<bundle-id>/observations.md`, beside `runtime/`. Store downloads in
that same ignored outer directory, with distinct names such as `u1-before.json`,
`u1-after.json` and `u1-restarted.json`. Configure the browser's save dialog or
move each explicitly selected download there; do not rely on overwriting its
default filename. Keep notes and review artifacts out of commits.

An operator may self-record, or an observer may take notes without coaching.
Record which arrangement was used, prior familiarity, browser/version and the
actual local time zone. Start and stop a clock at the stated task boundaries;
record pauses separately and leave missing times `unknown`. The scripted review
uses **Simulated test response**, even when a real person operates the page.
Human interaction observations can be real while the review remains simulated.

## Core operator tasks

1. **Connect and start.** Begin timing before opening `http://127.0.0.1:8765`.
   Enter the temporary token in **Access token**, click **Connect**, enter
   `m10-u1` in **Run name**, and click **Start / resume**. Stop timing when
   **3. Inspect the current record** is visible. Record the displayed run name,
   two pending cases and zero decisions. If preparation or access fails, record
   the error and assistance; do not report a completed UI task from API tests.

2. **Inspect and explain the evidence.** Time from beginning inspection until
   the operator has explained what they would change and why. Use **Current
   value**, **Publication evidence**, **Linked specimen and facility records**,
   **Source region and hash**, and **Current entity, assertion and relation
   bindings**. Explain the difference between Sample A's preparation at 500 K
   and Dataset A's acquisition at 300 K, and why two CaCO3 specimens remain
   distinct. Identify Dataset A's sample attachment and source hash. Save the
   pending export with **Save current record + review** as `u1-before.json`.
   Record whether the operator understands that pending is not approval and
   that a standards definition, if shown, is not publication evidence.

3. **Record uncertainty and recover a stale view.** Open a second tab, connect
   with the same token, enter `m10-u1` and click **Refresh** so both tabs show the
   same pending state. In each tab select the same first pending case, enter a
   pseudonymous **Your reviewer name or identifier**, choose **Simulated test
   response**, and enter an explicit scripted **Reason for your decision**.
   In the first tab click **Mark selected cases unresolved**. In the second tab
   attempt the same action without refreshing. Record the stale-record message;
   it must not create another decision. Click **Refresh** in the second tab and
   verify one decision and the resolved case's `unresolved` status. Then select
   the remaining pending case and mark it unresolved. Time the stale rejection
   and recovery separately from the successful decisions. Save `u1-after.json`:
   expect two decisions, two unresolved cases and no expert approval. The
   canonical JSON must still equal the pending export's canonical JSON.

4. **Recover after a process restart.** Record the current state ID, revision
   and decision count from **Run bindings, warnings and review history**. Stop
   only this service with Ctrl+C in its shell. Restart the same bundle with
   the launcher and `--resume-bundle`; paste the new temporary token when
   connecting again. Enter `m10-u1` and click **Refresh**. Time from initiating
   restart until the saved view is restored. Confirm the same state, revision,
   decisions and unresolved cases. Save `u1-restarted.json`; its canonical
   record, review sidecar and export manifest should match `u1-after.json`.
   No database repair, fresh child run or extra review submission is needed.

5. **Close and assess.** Click **Disconnect** and close both tabs, stop this
   service with Ctrl+C, and clear the temporary clipboard token. Record any
   confusing text, inaccessible controls, lost work, additional instructions
   or unanswered uncertainty. Describe one concrete improvement tied to an
   observed task. Do not convert the observer's opinion into an accuracy or
   commercial validation claim.

## Optional task: author an explicit correction

Decide whether to attempt this task before starting; report a skipped task as
`not_attempted`. Perform it before task 5 in a fresh `m10-e1` run. Save its pending
export, select both cases, fill the reviewer/provenance/rationale fields, and
open **Advanced: approve or edit with an explicit evidence response**. Click
**Prepare revision and case fields**. The editor supplies bindings only.

Time from reading the two issues to a successfully saved, inspected export;
include JSON authoring and validation retries. Set `outcome` to `edit`. Inspect
**Candidate corrections — not applied or approved** and copy the applicable
`proposal` object, without changing its expected revision. Author attestations
for the following three scalar paths, using the actual complete evidence object
displayed under **Source region and hash** for the source passage supporting
the claim. Preserve the source hash, region, offsets and quotation exactly.

| Path | `value_json` string content | Role |
|---|---|---|
| `/datasets/0/dataset_description` | `"Measured at 300 K."` | `measurement_acquisition` |
| `/datasets/0/sample_refs/0` | `"sample_1"` | `attachment` |
| `/datasets/0/facility_refs/0` | `"facility_1"` | `attachment` |

Each attestation has `path`, `value_json`, `role`, and `evidence` (an array of
displayed evidence objects). `value_json` is itself a JSON string containing
the quoted scalar; for example, the response JSON represents it as
`"value_json": "\"sample_1\""`. This deliberate manual step is a usability
constraint to observe. If it is unclear, stop or request help and record that
outcome; do not generate an apparently human response using an assistant.

After inspecting the entire response, check **I have inspected the evidence
and intend to submit this exact response.** and click **Submit explicit JSON
response**. Save the export. Success requires a completed review with two approved
cases, exactly the two intended canonical scalar changes, unchanged specimen IDs
and other values, three current evidence assertions, and simulated provenance
with expert identity still unauthenticated. A valid JSON submission alone is not
success. If rejected,
record the exact safe error, refresh before reconsidering stale bindings, and
keep all retries in the elapsed time. Never invent evidence or alter a database
to force completion. The supplied target values make this a guided synthetic
task; even actual correction time does not establish natural-paper performance.

## Recovery and stopping rules

On `pilot_busy`, wait for the active operation, then refresh. On stale state,
refresh and inspect rather than repeatedly submitting an old response. If a
review was saved but export failed, refresh and use **Save current record +
review**; do not submit another decision. For an ambiguous acknowledgement of
an advanced response, only exact replay of the original JSON/idempotency key
is safe; retain that response privately if this recovery is attempted.

On a verification, checksum, migration, unknown-dispatch, storage or path error,
preserve the run and record a blocked task. Use the manual guide for diagnosis;
do not delete files, reset the journal, change inputs, disable guards or open a
different port/interface to bypass an error. Stop at the session cap, on missing
evidence or if the operator cannot distinguish the active run. No scientific
approval, live inference, new document or training work follows automatically.

## Blank observation record

```markdown
# M10 synthetic operator trial observation
Protocol: m10-operator-trial-1
Status: not_run
Actual session date/time zone: not_observed
Operator pseudonym / observer or self-recording: not_observed
Prior familiarity / assistance agreed before start: not_observed
OS / browser/version / viewport or accessibility setup: not_observed
Bundle ID / manifest SHA256: not_recorded
Child run IDs: not_started
Source/configuration manifest references: not_recorded
Comparison: documented operation only; no comparative performance estimate
Scorer: none; completion checklist only
Budget: 45 minutes total; optional edit 10 minutes; two runs; zero live requests
Optional edit selected before session: undecided

| Task | Start | End | Pauses | Outcome | Assistance | Errors/retries | Artifact/evidence |
|---|---|---|---|---|---|---|---|
| 1 Connect/start | unknown | unknown | unknown | not_attempted | not_observed | not_observed | none |
| 2 Evidence explanation | unknown | unknown | unknown | not_attempted | not_observed | not_observed | none |
| 3 Decision/stale recovery | unknown | unknown | unknown | not_attempted | not_observed | not_observed | none |
| 4 Process restart/export | unknown | unknown | unknown | not_attempted | not_observed | not_observed | none |
| Optional explicit edit | unknown | unknown | unknown | not_attempted | not_observed | not_observed | none |
| 5 Close/assessment | unknown | unknown | unknown | not_attempted | not_observed | not_observed | none |

Observed mock dispatches / live attempts / cached replay: not_observed
Pending, unresolved and final export paths / state IDs: not_recorded
Canonical equality or exact changed paths checked: not_observed
Stale attempt decision counts before/after: not_observed
State/revision/export equality across restart: not_observed
Operator's evidence explanation and uncertainty: not_observed
Observed confusion / usability issue / concrete improvement: not_observed
Lost work, unresolved tasks, failures and limits: not_observed
Actual correction elapsed time / assistance / pauses: not_observed
No token or private judgment included in committed files: not_checked
Conclusion limited to this operator and invented example: no observations yet
```

After an actual person performs tasks, mark only that recorded session `pilot`
and distinguish completed, assisted, failed and unattempted tasks. Automated
diagnostics or this prepared document leave human E10 `not_run`. This small
synthetic rehearsal does not satisfy real-document usability, extraction
accuracy, independent-label or E09 comparison gates. It does not justify opening
M11 without a measured bottleneck and explicit training/data/compute scope.
