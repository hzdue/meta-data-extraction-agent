#!/usr/bin/env bash
# Run the three D32 development evaluations sequentially and apply Gate B.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -ne 1 || ( "$1" != "--status" && "$1" != "--run" ) ]]; then
  echo "Usage: bash scripts/run_finetune_followup_phase_b_evaluation.sh --status|--run" >&2
  exit 2
fi
MODE="$1"
ENVIRONMENT="$ROOT/outputs/finetune/local_env.sh"
PYTHON="$ROOT/outputs/finetune/venv-gpu/bin/python"
RUNNER="$ROOT/scripts/evaluate_finetune_followup_phase_b.py"
OUTPUT="$ROOT/outputs/finetune/followup/phase-b/evaluation"
LOG="$OUTPUT/run.log"
LOCK="$OUTPUT/evaluation.lock"

[[ -f "$ENVIRONMENT" ]] || { echo "Missing environment: $ENVIRONMENT" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "Missing GPU Python: $PYTHON" >&2; exit 1; }
[[ -f "$RUNNER" ]] || { echo "Missing evaluator: $RUNNER" >&2; exit 1; }

status() {
  local seed directory count state complete=0
  for seed in 42 123 456; do
    directory="$OUTPUT/d32_seed${seed}"
    count=0
    [[ -f "$directory/generation.jsonl" ]] && count="$(wc -l < "$directory/generation.jsonl")"
    if [[ "$count" -eq 30 && -f "$directory/metrics.json" ]]; then
      state=complete
      ((complete += 1))
    elif [[ "$count" -gt 0 ]]; then
      state="in_progress:${count}/30"
    else
      state=pending
    fi
    printf '%-20s %s\n' "d32_seed${seed}" "$state"
  done
  if [[ "$complete" -eq 3 && -f "$OUTPUT/comparison.json" && -f "$OUTPUT/comparison.md" ]]; then
    echo "gate_b complete"
    return 0
  fi
  echo "gate_b pending"
  return 1
}

if [[ "$MODE" == "--status" ]]; then
  status
  exit $?
fi

# shellcheck disable=SC1090
source "$ENVIRONMENT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/outputs/finetune/triton-cache}"
mkdir -p "$OUTPUT" "$TRITON_CACHE_DIR"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Another Phase B evaluation holds $LOCK" >&2
  exit 1
fi
if existing="$(pgrep -af '[/]run_finetune.*--train|[/]evaluate_finetune_followup_phase_b.py' || true)" && [[ -n "$existing" ]]; then
  echo "A fine-tuning or Phase B evaluation process is already active:" >&2
  echo "$existing" >&2
  exit 1
fi
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is unavailable" >&2; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
printf '\n===== %s Phase B evaluation =====\n' "$(date --iso-8601=seconds)" | tee -a "$LOG"
if ! PYTHONUNBUFFERED=1 "$PYTHON" "$RUNNER" --offline 2>&1 | tee -a "$LOG"; then
  code="${PIPESTATUS[0]}"
  printf '%s evaluation_failed exit=%s\n' "$(date --iso-8601=seconds)" "$code" | tee -a "$LOG"
  exit "$code"
fi
printf '%s evaluation_complete\n' "$(date --iso-8601=seconds)" | tee -a "$LOG"
status
