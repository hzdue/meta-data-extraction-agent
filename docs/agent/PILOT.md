# Local metadata review pilot

This is a single-user, loopback-only demonstration using one invented publication,
precomputed initial issues and mock model responses. It lets an operator start or
resume a workflow, inspect source evidence, record an explicit review response and
download canonical JSON with its review sidecar. It does not accept publications,
run live extraction, measure scientific accuracy or authenticate expert identity.

## Prepare an isolated runtime

Use the repository's existing Python 3.13+ and uv environment. From the repository
root in PowerShell, choose a new bundle ID; existing directories are never replaced:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$pilotPython = (Resolve-Path '.venv/Scripts/python.exe').Path
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.pilot_bundle build --run-id m10-local-v1
$pilotRoot = (Resolve-Path 'runs/agent/m10-local-v1/runtime').Path
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python -m meta_data_extraction.agent.pilot_bundle verify --root $pilotRoot
```

The bundle contains an exact allowlist of runtime code, schemas, prompt, mock
configuration and invented fixtures. It excludes evaluation/gold, benchmark
outputs, paper assets, credentials, test code and earlier raw runs. The original
README and LICENSE are preserved for attribution; this PILOT.md governs operation
of the isolated bundle. An empty license remains unresolved. No redistribution or
public deployment is approved by creating this local bundle.

The current environment can run the bundle through PYTHONPATH. For a separate
installation, place its environment outside the immutable runtime tree:

```powershell
$env:UV_PROJECT_ENVIRONMENT = (Join-Path (Split-Path $pilotRoot -Parent) 'venv')
uv --cache-dir tmp/uv-cache sync --project $pilotRoot --frozen --offline --no-dev
$pilotPython = Join-Path $env:UV_PROJECT_ENVIRONMENT 'Scripts/python.exe'
```

An offline installation needs the locked packages and build requirements in the
local uv cache. A missing cache is a setup blocker, not permission to alter the
lockfile. Do not install into a global environment. Do not add .venv, caches or
other files inside the runtime; verification permits mutations only in runs/agent.

## Start the local service

Generate a separate temporary pilot access token. This is not the ChatAI API key;
the service never reads the repository .env. Keep the same shell open:

```powershell
$pilotTokenBytes = New-Object byte[] 32
$pilotRng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
$pilotRng.GetBytes($pilotTokenBytes)
$env:META_AGENT_PILOT_TOKEN = [Convert]::ToBase64String($pilotTokenBytes).TrimEnd('=').Replace('+','-').Replace('/','_')
$pilotRng.Dispose()
Set-Clipboard -Value $env:META_AGENT_PILOT_TOKEN
$env:PYTHONPATH = (Join-Path $pilotRoot 'src')
& $pilotPython -m meta_data_extraction.agent.pilot_http serve --root $pilotRoot --host 127.0.0.1 --port 8765
```

Open http://127.0.0.1:8765 and paste the token into the access field. The page keeps
it only in memory; reload requires entering it again. The server prints a listening
event and redacted request summaries, never the token. If using the original
repository environment instead of a separate installation, keep the original
pilotPython path. The command invokes that selected interpreter directly.

Use an ID such as m10-example-1. Start/resume runs the fixed mock review scenario.
Inspect each case's value, linked entities, source passages and uncertainty before
responding. Source passages are publication evidence; standards definitions, when
present, are representation guidance. A supported location does not establish
complete scientific support or entity completeness.

The page provides a guided unresolved response and an explicit JSON editor for
approve/edit responses. Approval/edit requires source-linked attestations satisfying
the existing M08 rules; it is never filled or submitted automatically. Reviewer
identity is self-declared. Select simulated provenance for scripted demonstrations.
The advanced editor is a current usability limitation, not a finished curator UI.

Download returns canonical JSON, its review sidecar and an immutable export
manifest together. Pending/unresolved or simulated review remains labelled; an
export does not imply expert approval. Canonical schema, entity IDs and attachment
semantics remain unchanged. No catalog or external destination is contacted.

## API and recovery

All API requests require `Authorization: Bearer <pilot token>`. POST requests use
application/json. The static landing page contains no private data and is public
only on the loopback listener. Nonlocal Host/Origin, cross-origin API access,
arbitrary URLs/paths, duplicate JSON keys and oversized requests are rejected.

| Method and route | Operation |
|---|---|
| GET /health | Verified bundle mode, capabilities and limits |
| POST /runs | Start/resume with run_id and optional bounded max_steps |
| GET /runs/{id} | Inspect completion/incomplete status without dispatch |
| GET /runs/{id}/review | Current review state, cases, values and evidence |
| POST /runs/{id}/review | Explicit M08 ReviewResponse with expected state/revision and idempotency key |
| POST /runs/{id}/export | Download current export with expected_state_id |

Only m10-* IDs in this bundle are accepted; the service-lock ID is reserved. The
pilot admits eight runs and bounds stored bytes/files. Full-workspace retrieval,
uploaded documents, configurable model endpoints and arbitrary export paths are
not exposed. Source/configuration changes require a fresh bundle and fresh runs.
On Windows, run_path_too_long means the complete immutable export path would exceed
the supported path bound. Use a shorter bundle location/name and run ID before
starting; the service rejects this before dispatch rather than changing OS settings.

Stop with Ctrl+C. Restart with the same bundle and token (or a new token entered
in the page), then inspect/resume the same run ID. Existing SQLite checkpoints and
the dispatch journal survive. Resuming an interrupted HTTP request is safe only
through this explicit operation: an ambiguous dispatched model action stays
blocked and is not silently repeated. Workflow deadlines persist across restart.

Submit the exact same ReviewResponse/idempotency key to recover an uncertain
response acknowledgement. It cannot create a second decision. A changed response
needs a new key and current state/revision. A stale response is rejected; refresh
the view and reconsider it. A committed review whose export was interrupted can
be exported again with its current state ID, without another decision or model call.

Concurrent operations can return pilot_busy; wait for the active request and retry
the explicit operation. The shared lease and per-run transaction protect one local
operator, not a distributed multi-user deployment. Do not reset/delete databases
to bypass revision, checksum, unknown-dispatch or migration errors. Inspect status,
preserve the failed run and use a fresh ID when inputs must change.

The API layer rejects stale state and unsafe input; it is not an operating-system
sandbox against the local filesystem owner. Keep runtime code/configuration
immutable, and restrict writes to the bundle's runs/agent directory. Credentials,
source/review text and raw request bodies are excluded from server logs. Review
artifacts themselves retain the explicit responses under the private run directory.

## Container and evaluation handover

The bundle includes a Dockerfile and deny-by-default .dockerignore. Build only with
the isolated runtime as context, never the repository:

```powershell
docker build -t metadata-agent-pilot:local $pilotRoot
```

The recipe preserves the locked environment, uses a non-root runtime user and
makes application assets read-only, leaving runs/agent writable. It needs package
and base-image downloads. Docker was unavailable during this implementation, so
the image build, container recovery and container networking remain untested.
The listener stays on 127.0.0.1; host-browser access would require an appropriate
local container networking arrangement (the recipe notes Linux host networking).
Do not expose a public listener, publish the image or mount the repository into it.
The tested deployment target is the local Python service.

Offline evaluation is deliberately absent from the runtime. Keep study collection
in the development/evaluation checkout using the separately frozen M09 protocol;
see docs/agent/modules/M09_study.md and docs/agent/COMMANDS.md there. M09 currently
collects M07 predictions, not arbitrary pilot review exports. A future reviewed-
export comparison needs an explicit frozen selection and independent labels.

The worked example begins with Dataset A at 500 K and attached to sample_2 while
the invented source supports 300 K and sample_1. A simulated unresolved response
leaves the record unchanged with no active approval. An explicit valid two-scalar
edit may correct it while retaining both specimen identities and source bindings.
Neither example measures expert effort or validates natural-paper accuracy.

E10 user usability and actual correction time remain unmeasured until a real user
pilot is run on approved documents. E09's independent-label and reviewed-guidance
gates remain unresolved. The next practical work is that scoped operator pilot;
M11 fine-tuning is optional and requires a demonstrated bottleneck plus explicit
data/compute authorization.
