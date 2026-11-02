#!/usr/bin/env bash
# Run the frozen exploratory ablation matrix sequentially and fail closed.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  echo "Usage: bash scripts/run_finetune_ablation_matrix.sh --status|--run" >&2
}

if [[ $# -ne 1 || ( "$1" != "--status" && "$1" != "--run" ) ]]; then
  usage
  exit 2
fi
MODE="$1"

ENVIRONMENT="$ROOT/outputs/finetune/local_env.sh"
PYTHON="$ROOT/outputs/finetune/venv-gpu/bin/python"
RUNNER="$ROOT/scripts/run_finetune_ablation.py"
OUTPUT_ROOT="$ROOT/outputs/finetune/ablations"
LOG_ROOT="$ROOT/outputs/finetune/ablation-matrix-logs"

[[ -f "$ENVIRONMENT" ]] || { echo "Missing environment: $ENVIRONMENT" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "Missing GPU Python: $PYTHON" >&2; exit 1; }
[[ -f "$RUNNER" ]] || { echo "Missing runner: $RUNNER" >&2; exit 1; }

# shellcheck disable=SC1090
source "$ENVIRONMENT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/outputs/finetune/triton-cache}"

mkdir -p "$LOG_ROOT" "$TRITON_CACHE_DIR"
exec 9>"$LOG_ROOT/matrix.lock"
if ! flock -n 9; then
  echo "Another matrix runner holds $LOG_ROOT/matrix.lock" >&2
  exit 1
fi

MATRIX_LOG="$LOG_ROOT/matrix.log"
log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$MATRIX_LOG"
}

conditions=(
  d_only
  matched_d_de
  matched_d_dr
  matched_d_der
  matched_compute_de
  matched_compute_dr
  matched_compute_der
  matched_d_drp
)
seeds=(42 123 456)
declare -A endpoint=(
  [d_only]=6
  [matched_d_de]=8
  [matched_d_dr]=8
  [matched_d_der]=10
  [matched_compute_de]=6
  [matched_compute_dr]=6
  [matched_compute_der]=6
  [matched_d_drp]=8
)

verify_complete() {
  local output="$1" condition="$2" seed="$3" expected_step="$4"
  "$PYTHON" - "$output" "$condition" "$seed" "$expected_step" <<'PY'
import json
from pathlib import Path
import sys

from mlflow import MlflowClient

output = Path(sys.argv[1])
condition, seed, endpoint = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

def read(name):
    path = output / name
    if not path.is_file():
        raise ValueError(f"missing {name}")
    return json.loads(path.read_text(encoding="utf-8"))

summary = read("training_summary.json")
selection = read("selection.json")
record = read("mlflow_run.json")
ablation = summary.get("ablation") or {}
if (summary.get("global_step"), ablation.get("condition"), ablation.get("seed")) != (
        endpoint, condition, seed):
    raise ValueError("endpoint, condition or seed does not match the frozen matrix")
planned = ablation.get("planned_exposure")
actual = (summary.get("exposure") or {}).get("examples")
if not planned or actual != planned:
    raise ValueError(f"exposure mismatch: planned={planned!r}, actual={actual!r}")
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
        run.data.tags.get("finetune.ablation_condition")) != ("ablation", condition):
    raise ValueError("MLflow phase or condition tag differs")
print(f"verified complete: step={endpoint}, exposure={actual}, selected={selection['step']}, "
      f"mlflow={record['run_id']}")
PY
}

latest_resume_checkpoint() {
  local output="$1" expected_step="$2" path step latest=""
  shopt -s nullglob
  for path in "$output"/checkpoint-*; do
    [[ -d "$path" ]] || continue
    step="${path##*-}"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    (( step < expected_step )) || continue
    local complete=true name
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
  local output="$1" condition="$2" seed="$3" expected_step="$4" resume
  if verify_complete "$output" "$condition" "$seed" "$expected_step" >/dev/null 2>&1; then
    echo complete
  elif [[ ! -d "$output" ]] || [[ -z "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo pending
  elif resume="$(latest_resume_checkpoint "$output" "$expected_step")"; then
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
    for condition in "${conditions[@]}"; do
      output="$OUTPUT_ROOT/${condition}_seed${seed}"
      state="$(run_state "$output" "$condition" "$seed" "${endpoint[$condition]}")"
      printf '%-28s %s\n' "${condition}_seed${seed}" "$state"
      case "$state" in
        complete) ((complete_count += 1)) ;;
        pending) ((pending_count += 1)) ;;
        resumable:*) ((resumable_count += 1)) ;;
        blocked) ((blocked_count += 1)) ;;
      esac
    done
  done
  printf 'summary: complete=%d pending=%d resumable=%d blocked=%d\n' \
    "$complete_count" "$pending_count" "$resumable_count" "$blocked_count"
  exit $(( blocked_count > 0 ? 1 : 0 ))
fi

if existing="$(pgrep -af '[/]run_finetune_ablation.py --train' || true)" && [[ -n "$existing" ]]; then
  echo "A fine-tuning process is already active; refusing parallel execution:" >&2
  echo "$existing" >&2
  exit 1
fi
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is unavailable" >&2; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

log "study=exploratory action=matrix_start"
for seed in "${seeds[@]}"; do
  for condition in "${conditions[@]}"; do
    expected_step="${endpoint[$condition]}"
    output="$OUTPUT_ROOT/${condition}_seed${seed}"
    name="${condition}_seed${seed}"
    state="$(run_state "$output" "$condition" "$seed" "$expected_step")"
    case "$state" in
      complete)
        log "run=$name action=skip reason=verified_complete"
        continue
        ;;
      pending)
        command=("$PYTHON" "$RUNNER" --train --condition "$condition" --seed "$seed" --offline)
        log "run=$name action=start endpoint=$expected_step"
        ;;
      resumable:*)
        resume="${state#resumable:}"
        command=("$PYTHON" "$RUNNER" --train --condition "$condition" --seed "$seed" \
                 --offline --resume "$resume")
        log "run=$name action=resume endpoint=$expected_step checkpoint=$resume"
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
    if ! verification="$(verify_complete "$output" "$condition" "$seed" "$expected_step" 2>&1)"; then
      log "run=$name action=stop reason=post_run_verification_failed"
      echo "$verification" >&2
      exit 1
    fi
    log "run=$name action=complete $verification"
  done
done
log "study=exploratory action=matrix_complete runs=24"
