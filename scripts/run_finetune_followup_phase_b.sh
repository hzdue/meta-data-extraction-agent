#!/usr/bin/env bash
# Run the three frozen Phase B D32 seeds sequentially and fail closed.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  echo "Usage: bash scripts/run_finetune_followup_phase_b.sh --status|--run" >&2
}
if [[ $# -ne 1 || ( "$1" != "--status" && "$1" != "--run" ) ]]; then
  usage
  exit 2
fi
MODE="$1"

ENVIRONMENT="$ROOT/outputs/finetune/local_env.sh"
PYTHON="$ROOT/outputs/finetune/venv-gpu/bin/python"
RUNNER="$ROOT/scripts/run_finetune_followup_training.py"
OUTPUT_ROOT="$ROOT/outputs/finetune/followup/phase-b"
LOG_ROOT="$OUTPUT_ROOT/logs"
EXPECTED_STEP=8
seeds=(42 123 456)

[[ -f "$ENVIRONMENT" ]] || { echo "Missing environment: $ENVIRONMENT" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "Missing GPU Python: $PYTHON" >&2; exit 1; }
[[ -f "$RUNNER" ]] || { echo "Missing runner: $RUNNER" >&2; exit 1; }

# shellcheck disable=SC1090
source "$ENVIRONMENT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/outputs/finetune/triton-cache}"

mkdir -p "$LOG_ROOT" "$TRITON_CACHE_DIR"
exec 9>"$LOG_ROOT/phase-b.lock"
if ! flock -n 9; then
  echo "Another Phase B runner holds $LOG_ROOT/phase-b.lock" >&2
  exit 1
fi

MASTER_LOG="$LOG_ROOT/phase-b.log"
log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$MASTER_LOG"
}

verify_complete() {
  local output="$1" seed="$2"
  "$PYTHON" - "$output" "$seed" "$EXPECTED_STEP" <<'PY'
import json
from pathlib import Path
import sys

from mlflow import MlflowClient

output = Path(sys.argv[1])
seed, endpoint = int(sys.argv[2]), int(sys.argv[3])

def read(name):
    path = output / name
    if not path.is_file():
        raise ValueError(f"missing {name}")
    return json.loads(path.read_text(encoding="utf-8"))

summary = read("training_summary.json")
selection = read("selection.json")
record = read("mlflow_run.json")
experiment = summary.get("experiment") or {}
if (summary.get("global_step"), experiment.get("phase"), experiment.get("condition"),
        experiment.get("seed")) != (endpoint, "followup_phase_b", "d32", seed):
    raise ValueError("endpoint, phase, condition or seed differs from the frozen design")
if (summary.get("exposure") or {}).get("examples") != {"D": 32}:
    raise ValueError("actual exposure is not D32")
if experiment.get("planned_exposure") != {"D": 32}:
    raise ValueError("planned exposure is not D32")
if set((experiment.get("per_paper_exposure") or {}).values()) != {4}:
    raise ValueError("each training paper must appear four times")
if selection.get("status") != "complete" or selection.get("step") not in {0, endpoint}:
    raise ValueError("selection is incomplete or not a declared endpoint")
checkpoint = output / f"checkpoint-{endpoint}"
required = (
    "adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
    "optimizer.pt", "scheduler.pt", "rng_state.pth", "exposure.json",
    "validation_state.json", "inference_precision.json",
)
missing = [name for name in required if not (checkpoint / name).is_file()]
if missing:
    raise ValueError(f"endpoint checkpoint is incomplete: {missing}")
run = MlflowClient(tracking_uri=record["settings"]["tracking_uri"]).get_run(record["run_id"])
if run.info.status != "FINISHED":
    raise ValueError(f"MLflow status is {run.info.status}, not FINISHED")
if (run.data.tags.get("finetune.phase"),
        run.data.tags.get("finetune.experiment_condition")) != ("followup_phase_b", "d32"):
    raise ValueError("MLflow phase or condition tag differs")
print(f"verified complete: step={endpoint}, exposure=D32, selected={selection['step']}, "
      f"mlflow={record['run_id']}")
PY
}

latest_resume_checkpoint() {
  local output="$1" path step latest="" complete name
  shopt -s nullglob
  for path in "$output"/checkpoint-*; do
    [[ -d "$path" ]] || continue
    step="${path##*-}"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    (( step < EXPECTED_STEP )) || continue
    complete=true
    for name in adapter_config.json adapter_model.safetensors trainer_state.json optimizer.pt \
      scheduler.pt rng_state.pth exposure.json validation_state.json inference_precision.json; do
      [[ -f "$path/$name" ]] || complete=false
    done
    [[ "$complete" == true ]] || continue
    if [[ -z "$latest" || "$step" -gt "${latest##*-}" ]]; then
      latest="$path"
    fi
  done
  shopt -u nullglob
  [[ -n "$latest" ]] && printf '%s\n' "$latest"
}

run_state() {
  local output="$1" seed="$2" resume
  if verify_complete "$output" "$seed" >/dev/null 2>&1; then
    echo complete
  elif [[ ! -d "$output" ]] || [[ -z "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo pending
  elif resume="$(latest_resume_checkpoint "$output")"; then
    echo "resumable:$resume"
  else
    echo blocked
  fi
}

if [[ "$MODE" == "--status" ]]; then
  complete_count=0
  pending_count=0
  resumable_count=0
  blocked_count=0
  for seed in "${seeds[@]}"; do
    output="$OUTPUT_ROOT/d32_seed${seed}"
    state="$(run_state "$output" "$seed")"
    printf '%-20s %s\n' "d32_seed${seed}" "$state"
    case "$state" in
      complete) ((complete_count += 1)) ;;
      pending) ((pending_count += 1)) ;;
      resumable:*) ((resumable_count += 1)) ;;
      blocked) ((blocked_count += 1)) ;;
    esac
  done
  printf 'summary: complete=%d pending=%d resumable=%d blocked=%d\n' \
    "$complete_count" "$pending_count" "$resumable_count" "$blocked_count"
  exit $(( blocked_count > 0 ? 1 : 0 ))
fi

if existing="$(pgrep -af '[/]run_finetune.*--train' || true)" && [[ -n "$existing" ]]; then
  echo "A fine-tuning process is already active; refusing parallel execution:" >&2
  echo "$existing" >&2
  exit 1
fi
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is unavailable" >&2; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

log "study=exploratory phase=followup_phase_b action=start runs=3"
for seed in "${seeds[@]}"; do
  output="$OUTPUT_ROOT/d32_seed${seed}"
  name="d32_seed${seed}"
  state="$(run_state "$output" "$seed")"
  case "$state" in
    complete)
      log "run=$name action=skip reason=verified_complete"
      continue
      ;;
    pending)
      command=("$PYTHON" "$RUNNER" --phase b --train --seed "$seed" --offline)
      log "run=$name action=start endpoint=$EXPECTED_STEP"
      ;;
    resumable:*)
      resume="${state#resumable:}"
      command=("$PYTHON" "$RUNNER" --phase b --train --seed "$seed" --offline \
               --resume "$resume")
      log "run=$name action=resume endpoint=$EXPECTED_STEP checkpoint=$resume"
      ;;
    *)
      log "run=$name action=stop reason=nonempty_output_without_resumable_state output=$output"
      echo "Inspect $output manually; it is neither verified complete nor safely resumable." >&2
      exit 1
      ;;
  esac

  run_log="$LOG_ROOT/${name}.log"
  printf '\n===== %s %s =====\n' "$(date --iso-8601=seconds)" "$name" | tee -a "$run_log"
  if ! PYTHONUNBUFFERED=1 "${command[@]}" 2>&1 | tee -a "$run_log"; then
    status="${PIPESTATUS[0]}"
    log "run=$name action=stop reason=training_failed exit=$status log=$run_log"
    exit "$status"
  fi
  if ! verification="$(verify_complete "$output" "$seed" 2>&1)"; then
    log "run=$name action=stop reason=post_run_verification_failed"
    echo "$verification" >&2
    exit 1
  fi
  log "run=$name action=complete $verification"
done
log "study=exploratory phase=followup_phase_b action=complete runs=3"
