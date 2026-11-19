#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="$ROOT/outputs/finetune/pseudo_label_supervision/pseudo_length_filtered_seed42"
LOG_DIR="$ROOT/outputs/finetune/pseudo_label_supervision/logs"
PID_FILE="$LOG_DIR/pseudo_stage_a.pid"

if [[ -f "$PID_FILE" ]]; then
  previous_pid="$(<"$PID_FILE")"
  if kill -0 "$previous_pid" 2>/dev/null; then
    echo "Stage A is already running as PID $previous_pid" >&2
    exit 1
  fi
fi
if [[ -d "$OUTPUT" ]] && [[ -n "$(find "$OUTPUT" -mindepth 1 -print -quit)" ]]; then
  echo "Stage-A output is not empty; archive it or resume a valid checkpoint." >&2
  exit 1
fi
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Missing $ROOT/.venv; run: uv sync --group finetune-cuda --offline" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$ROOT/outputs/finetune/triton-cache"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log="$LOG_DIR/pseudo_stage_a_${stamp}.log"

cd "$ROOT"
nohup env \
  CUDA_VISIBLE_DEVICES=0 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  MLFLOW_DISABLE_AGENT_HINT=1 \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  TRITON_CACHE_DIR="$ROOT/outputs/finetune/triton-cache" \
  MLFLOW_TRACKING_URI="sqlite:///$ROOT/outputs/finetune/mlflow.db" \
  FINETUNE_MLFLOW_ARTIFACT_LOCATION="file://$ROOT/outputs/finetune/mlflow-artifacts" \
  "$ROOT/.venv/bin/python" scripts/run_pseudo_finetune.py \
    --stage a --train --offline \
  >"$log" 2>&1 </dev/null &

pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "Stage A exited during startup; inspect $log" >&2
  exit 1
fi

echo "Stage A started as PID $pid"
echo "Log: $log"
echo "Status: ps -p $pid -o pid,etime,stat,cmd"
echo "Follow: tail -f $log"
