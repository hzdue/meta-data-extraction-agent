# Run one guarded paper through Opper

This launcher is pinned to one calibration paper and
`openai/gpt-4o-2024-08-06`. It allows at most three model requests and does not
apply model-proposed corrections automatically.

Use a fresh run ID. The example below is PowerShell:

~~~powershell
$run = "m10-opper-4o-181203956-v1"

uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python scripts/agent/test_opper_one_paper.py prepare --run-id $run

Get-Content "runs/agent/$run-opper-prep/token-plan.json"
Get-Content "runs/agent/$run-opper-prep/protocol.json"
~~~

Preparation is offline. It sends no request, reads no credential, and creates an
ignored, immutable prep directory. Put the key only in the ignored root `.env`:

~~~text
OPPER_API_KEY=replace-with-the-real-key
~~~

Do not commit `.env` or pass the key on the command line. Review Opper's tracing,
retention, and data-residency settings for this paper before continuing.

Run the single authenticated model-catalog check:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --no-sync python scripts/agent/test_opper_one_paper.py catalog --run-id $run --load-dotenv
~~~

This makes one administrative `GET /models`; it sends no prompt or document and
does not invoke a model. It records only allowlisted model metadata, numeric quota
headers, and status. There is no automatic retry. If it fails, inspect the safe
error and prepare a fresh run ID rather than overwriting the attempt.

After the catalog succeeds, retention has been reviewed, and the persistent
ledger reports ready, execute the bounded run:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --no-sync python scripts/agent/test_opper_one_paper.py run --run-id $run --load-dotenv --execute-live --retention-reviewed
~~~

A new Opper ledger intentionally begins with unknown recent history. If the
command reports `request_history_unknown`, wait until one hour after `prepare`
and rerun the same command. It has not created the publication run or sent a
model request at that point. Do not reset or edit the ledger.

If the operator explicitly accepts that external request history is untracked,
and the shared local Opper ledger still contains zero attempts, the initial wait
can be overridden for this run only:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --no-sync python scripts/agent/test_opper_one_paper.py run --run-id $run --load-dotenv --execute-live --retention-reviewed --acknowledge-untracked-history
~~~

This writes an immutable run-scoped acknowledgement. It does not claim that
external history was verified, and it does not disable the rolling-hour limit,
three-call limit, output budget, or request accounting. Do not use the flag if
another application may have attempted Opper inference with the same key during
the preceding hour.

Read the final usage summary without dispatching anything:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python scripts/agent/test_opper_one_paper.py report --run-id $run
~~~

`known_input_tokens` and `known_output_tokens` come from provider usage. If Opper
does not return usage, `usage_unknown` remains true and zero known tokens must not
be interpreted as zero consumption. The workflow may finish as `pending_review`;
that is expected, and no proposed patch is automatically accepted.

The pinned preflight is 68,033 characters / 68,176 UTF-8 bytes for the first
prompt, about 17,009 input tokens by the rough characters/4 heuristic. The
conservative first-request bound is 68,304. The hard reservation is at most
100,128 input-token equivalents and 8,192 output tokens per request, for up to
three requests (300,384 and 24,576 respectively). Only returned provider usage is
an authoritative token measurement.
