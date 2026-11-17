# Run one guarded paper through Opper

This launcher is pinned to one calibration paper. Its historical default remains
`openai/gpt-4o-2024-08-06`; `--model` can instead freeze another exact,
provider-prefixed Opper route for a fresh run. Repeat that same model argument at
every stage so a prepared scope cannot silently change. The launcher allows at
most three model requests and does not apply model-proposed corrections
automatically.

Use a fresh run ID. The example below is PowerShell:

~~~powershell
$run = "m10-opper-fable52-181203956-v2"
$model = "anthropic/claude-fable-5-2"
$format = "json_object"
$effort = "low"
$retention = "retain"
$output = 16384

uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python scripts/agent/test_opper_one_paper.py prepare --run-id $run --model $model --response-format $format --reasoning-effort $effort --raw-retention $retention --max-output-tokens $output

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
uv --cache-dir tmp/uv-cache run --frozen --no-sync python scripts/agent/test_opper_one_paper.py catalog --run-id $run --model $model --response-format $format --reasoning-effort $effort --raw-retention $retention --max-output-tokens $output --load-dotenv
~~~

This makes one administrative `GET /models`; it sends no prompt or document and
does not invoke a model. It records only allowlisted model metadata, numeric quota
headers, and status. There is no automatic retry. If it fails, inspect the safe
error and prepare a fresh run ID rather than overwriting the attempt.

After the catalog succeeds, retention has been reviewed, and the persistent
ledger reports ready, execute the bounded run:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --no-sync python scripts/agent/test_opper_one_paper.py run --run-id $run --model $model --response-format $format --reasoning-effort $effort --raw-retention $retention --max-output-tokens $output --load-dotenv --execute-live --retention-reviewed
~~~

At the operator's explicit request, Opper does not impose an initial one-hour
unknown-history wait. The persistent ledger still counts every locally attempted
request, enforces ten attempts in a rolling hour, and preserves the per-run call
and output budgets. Calls made by unrelated applications remain unobservable, so
the operator must account for them separately. The older
`--acknowledge-untracked-history` mechanism remains readable for historical runs
but is not needed for new Opper execution.

Read the final usage summary without dispatching anything:

~~~powershell
uv --cache-dir tmp/uv-cache run --frozen --offline --no-sync python scripts/agent/test_opper_one_paper.py report --run-id $run --model $model --response-format $format --reasoning-effort $effort --raw-retention $retention --max-output-tokens $output
~~~

`known_input_tokens` and `known_output_tokens` come from provider usage. If Opper
does not return usage, `usage_unknown` remains true and zero known tokens must not
be interpreted as zero consumption. The workflow may finish as `pending_review`;
that is expected, and no proposed patch is automatically accepted.

The pinned preflight is 68,033 characters / 68,176 UTF-8 bytes for the first
prompt, about 17,009 input tokens by the rough characters/4 heuristic. The
conservative first-request bound is 68,304. The hard reservation is at most
100,128 input-token equivalents and 16,384 output tokens per request, for up to
three requests (300,384 and 49,152 respectively). Only returned provider usage is
an authoritative token measurement.

For Fable 5.2, adaptive thinking is always enabled. Opper's live model entry
advertises `low|medium|high|xhigh|max` and defaults to `high`; the example freezes
`low` because the first high-effort attempt spent its complete 16,384-token output
allowance on hidden thinking and returned an empty visible response. The effort
value is part of the immutable scope and must be repeated at every stage.
