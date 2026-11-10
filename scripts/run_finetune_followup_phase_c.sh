#!/usr/bin/env bash
# Run the six authorized Phase C order controls sequentially and fail closed.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ $# -ne 1 || ( "$1" != "--status" && "$1" != "--run" ) ]]; then
  echo "Usage: bash scripts/run_finetune_followup_phase_c.sh --status|--run" >&2
  exit 2
fi
MODE="$1"
source "$ROOT/outputs/finetune/local_env.sh"
PYTHON="$ROOT/outputs/finetune/venv-gpu/bin/python"
RUNNER="$ROOT/scripts/run_finetune_followup_training.py"
OUTPUT_ROOT="$ROOT/outputs/finetune/followup/phase-c"
LOG_ROOT="$OUTPUT_ROOT/logs"
EXPECTED_STEP=8
conditions=(r_then_d d_then_r)
seeds=(42 123 456)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/outputs/finetune/triton-cache}"
mkdir -p "$LOG_ROOT" "$TRITON_CACHE_DIR"
exec 9>"$LOG_ROOT/phase-c.lock"
flock -n 9 || { echo "Another Phase C runner holds the lock" >&2; exit 1; }

log() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG_ROOT/phase-c.log"; }

verify_complete() {
  local output="$1" condition="$2" seed="$3"
  "$PYTHON" - "$output" "$condition" "$seed" <<'PY'
import json
from pathlib import Path
import sys
from mlflow import MlflowClient

output, condition, seed = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
boundary = 2 if condition == "r_then_d" else 6
def read(path):
    if not path.is_file(): raise ValueError(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))
summary, record = read(output / "training_summary.json"), read(output / "mlflow_run.json")
experiment = summary.get("experiment") or {}
if (summary.get("global_step"), experiment.get("phase"), experiment.get("condition"),
        experiment.get("seed"), experiment.get("boundary_update")) != (
        8, "followup_phase_c", condition, seed, boundary):
    raise ValueError("endpoint or experiment identity differs")
if (summary.get("exposure") or {}).get("examples") != {"D": 24, "R": 8}:
    raise ValueError("endpoint exposure differs")
expected_boundary = {"R": 8} if condition == "r_then_d" else {"D": 24}
required = ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
            "optimizer.pt", "scheduler.pt", "rng_state.pth", "exposure.json",
            "validation_state.json", "inference_precision.json")
for step, exposure in ((boundary, expected_boundary), (8, {"D": 24, "R": 8})):
    checkpoint = output / f"checkpoint-{step}"
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing: raise ValueError(f"checkpoint-{step} incomplete: {missing}")
    saved = read(checkpoint / "exposure.json")
    if (saved.get("global_step"), saved.get("examples")) != (step, exposure):
        raise ValueError(f"checkpoint-{step} exposure differs")
run = MlflowClient(tracking_uri=record["settings"]["tracking_uri"]).get_run(record["run_id"])
if run.info.status != "FINISHED": raise ValueError(f"MLflow status is {run.info.status}")
if (run.data.tags.get("finetune.phase"), run.data.tags.get("finetune.experiment_condition")) != (
        "followup_phase_c", condition):
    raise ValueError("MLflow tags differ")
print(f"verified step=8 boundary={boundary} exposure=D24+R8 mlflow={record['run_id']}")
PY
}

latest_resume() {
  local output="$1" path step latest="" name complete
  shopt -s nullglob
  for path in "$output"/checkpoint-*; do
    step="${path##*-}"; [[ "$step" =~ ^[0-9]+$ ]] || continue
    (( step < EXPECTED_STEP )) || continue
    complete=true
    for name in adapter_config.json adapter_model.safetensors trainer_state.json optimizer.pt \
      scheduler.pt rng_state.pth exposure.json validation_state.json inference_precision.json; do
      [[ -f "$path/$name" ]] || complete=false
    done
    [[ "$complete" == true ]] || continue
    [[ -z "$latest" || "$step" -gt "${latest##*-}" ]] && latest="$path"
  done
  shopt -u nullglob
  [[ -n "$latest" ]] && printf '%s\n' "$latest"
}

state() {
  local output="$1" condition="$2" seed="$3" resume
  if verify_complete "$output" "$condition" "$seed" >/dev/null 2>&1; then echo complete
  elif [[ ! -d "$output" ]] || [[ -z "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then echo pending
  elif resume="$(latest_resume "$output")"; then echo "resumable:$resume"
  else echo blocked
  fi
}

if [[ "$MODE" == "--status" ]]; then
  for condition in "${conditions[@]}"; do
    for seed in "${seeds[@]}"; do
      output="$OUTPUT_ROOT/${condition}_seed${seed}"
      printf '%-24s %s\n' "${condition}_seed${seed}" "$(state "$output" "$condition" "$seed")"
    done
  done
  exit 0
fi

if existing="$(pgrep -af '[/]run_finetune.*--train' || true)" && [[ -n "$existing" ]]; then
  echo "A fine-tuning process is already active:" >&2; echo "$existing" >&2; exit 1
fi
log "study=post_hoc_exploratory phase=followup_phase_c action=start runs=6"
for condition in "${conditions[@]}"; do
  for seed in "${seeds[@]}"; do
    output="$OUTPUT_ROOT/${condition}_seed${seed}"
    name="${condition}_seed${seed}"
    current="$(state "$output" "$condition" "$seed")"
    case "$current" in
      complete) log "run=$name action=skip reason=verified_complete"; continue ;;
      pending) command=("$PYTHON" "$RUNNER" --phase c --train --condition "$condition" --seed "$seed" --offline) ;;
      resumable:*) resume="${current#resumable:}"; command=("$PYTHON" "$RUNNER" --phase c --train --condition "$condition" --seed "$seed" --offline --resume "$resume") ;;
      *) log "run=$name action=stop reason=ambiguous_nonempty_output"; exit 1 ;;
    esac
    log "run=$name action=start_or_resume state=$current endpoint=8"
    run_log="$LOG_ROOT/${name}.log"
    if ! PYTHONUNBUFFERED=1 "${command[@]}" 2>&1 | tee -a "$run_log"; then
      status="${PIPESTATUS[0]}"; log "run=$name action=stop exit=$status"; exit "$status"
    fi
    verification="$(verify_complete "$output" "$condition" "$seed")"
    log "run=$name action=complete $verification"
  done
done
log "study=post_hoc_exploratory phase=followup_phase_c action=complete runs=6"
