# Opper 30-paper evaluation batch

This launcher prepares and runs the frozen set of 30 publications whose split is
`evaluation` in `data/corpus.json`, using Opper model
`openai/gpt-4o-2024-08-06`. It extracts records only: it does not open final-test
gold, score results, or treat pending review as approval.

The prepared local batch is `m10-e30-gpt4o-v2`. Its exact offline preflight
parsed all 30 papers and found every first request eligible. The preflight read no
credentials and made no network or model requests.

## Inspect the prepared batch

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python scripts/agent/run_opper_evaluation_30.py status --batch-id m10-e30-gpt4o-v2
~~~

To reproduce preparation and preflight under a new batch ID:

~~~powershell
$batch="m10-e30-gpt4o-v3"; uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python scripts/agent/run_opper_evaluation_30.py prepare --batch-id $batch; if ($LASTEXITCODE -eq 0) { uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python scripts/agent/run_opper_evaluation_30.py preflight --batch-id $batch }
~~~

## Catalog and live execution

First make the separate administrative catalog request. It sends no paper or
prompt and records only allowlisted model metadata and numeric limit headers:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --no-sync python scripts/agent/run_opper_evaluation_30.py catalog --batch-id m10-e30-gpt4o-v2 --load-dotenv
~~~

After reviewing the returned catalog-derived maximum cost, run the available
papers. This example accepts no catalog-derived ceiling above USD 35:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --no-sync python scripts/agent/run_opper_evaluation_30.py run-available --batch-id m10-e30-gpt4o-v2 --load-dotenv --execute-live --retention-reviewed --accept-cost-ceiling-usd 35 --max-papers 30
~~~

The command is intentionally resumable. It stops before starting another paper
when fewer than three of the shared ten rolling-hour request slots are available.
Run the same command again after old attempts age out; completed and terminal
paper runs are not repeated. It never waits in the background, resets the shared
ledger, or bypasses unknown request history.

At the catalog prices observed in the one-paper pilot, the deliberately
conservative hard maximum is USD 34.1478. This assumes all 90 possible requests
use 119,000 input and 8,192 output tokens; it is a planning ceiling, not expected
usage or a bill. Actual provider-reported usage remains authoritative, and
missing usage remains unknown.

Each paper keeps hash-only raw-response retention, allows at most three model
requests, reserves at most 8,192 output tokens per request, and disables automatic
record patches. Results stay in ignored `runs/agent/` directories. Do not run an
evaluator or inspect final-test gold as part of this extraction batch.
