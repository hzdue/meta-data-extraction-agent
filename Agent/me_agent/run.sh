#!/usr/bin/env bash
set -Eeuo pipefail
# One-GPU Gemma baseline. See README.md for full and cached-repair commands.
# This runner owns only services it starts. Per-paper outputs include traces and costs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -z "${VIRTUAL_ENV:-}" && -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
fi

PYTHON="${PYTHON:-python}"
GPU_ID="${GPU_ID:-0}"
export SINGLE_GPU_PIPELINE="${SINGLE_GPU_PIPELINE:-1}"

# Model selection. MODEL_NAME is accepted as an alias because it is easier to
# remember from experiment notes; MODEL keeps backward compatibility.
MODEL_PROFILE="${MODEL_PROFILE:-gemma31b}"
if [[ -n "${MODEL_NAME:-}" && -z "${MODEL:-}" ]]; then
    MODEL="${MODEL_NAME}"
else
    MODEL="${MODEL:-}"
fi

case "${MODEL_PROFILE}" in
    gemma31b)
        MODEL="${MODEL:-google/gemma-4-31B-it}"
        DEFAULT_VLLM_MAX_MODEL_LEN=65536
        DEFAULT_MAX_CONTEXT_CHARS=90000
        DEFAULT_MAX_OUTPUT_TOKENS=6500
        DEFAULT_VLLM_GPU_MEMORY_UTILIZATION=0.90
        ;;
    qwen35b|large)
        MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
        DEFAULT_VLLM_MAX_MODEL_LEN=65536
        DEFAULT_MAX_CONTEXT_CHARS=120000
        DEFAULT_MAX_OUTPUT_TOKENS=10000
        DEFAULT_VLLM_GPU_MEMORY_UTILIZATION=0.95
        ;;
    default|qwen9b|small)
        MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
        DEFAULT_VLLM_MAX_MODEL_LEN=65536
        DEFAULT_MAX_CONTEXT_CHARS=90000
        DEFAULT_MAX_OUTPUT_TOKENS=6500
        DEFAULT_VLLM_GPU_MEMORY_UTILIZATION=0.90
        ;;
    *)
        echo "ERROR: MODEL_PROFILE must be gemma31b, default, qwen35b, large, qwen9b, or small."
        exit 2
        ;;
esac

PAPER_LIST="${PAPER_LIST:-qwen30_papers.txt}"
BATCH_SPLIT="${BATCH_SPLIT:-all}"
BATCH_RUN_ID="${BATCH_RUN_ID:-run_${BATCH_SPLIT}}"
if [[ "${MODEL_PROFILE}" == "gemma31b" ]]; then
    export ME_OUTPUT_ROOT="${ME_OUTPUT_ROOT:-outputs/experiments}"
    export ME_RUN_NAME="${ME_RUN_NAME:-${BATCH_RUN_ID}}"
    VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
fi
export ME_OUTPUT_ROOT="${ME_OUTPUT_ROOT:-outputs}"
export ME_RUN_NAME="${ME_RUN_NAME:-run}"
if [[ ! "${ME_RUN_NAME}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "ERROR: ME_RUN_NAME must be a simple experiment name."
    exit 2
fi
# Replace repair artifacts in the existing run; preserve all frozen upstream inputs.
if [[ "${REPAIR_IN_PLACE:-0}" == "1" ]]; then
    if [[ -n "${REPAIR_UPSTREAM_ROOT:-}" ]]; then
        echo "ERROR: REPAIR_IN_PLACE uses this output directory; unset REPAIR_UPSTREAM_ROOT."
        exit 2
    fi
    export FRESH_RUN=0 REPAIR_FROM_CACHE=1 SKIP_COMPLETED=0 SINGLE_GPU_PIPELINE=0
    echo "[run] In-place repair: replacing repair/final outputs in ${ME_OUTPUT_ROOT}/${ME_RUN_NAME}; reusing seed/anchors/detector."
fi
RUN_OUTPUT_DIR="${ME_OUTPUT_ROOT}/${ME_RUN_NAME}"
MAX_PAPERS="${MAX_PAPERS:-30}"
MAX_CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-${DEFAULT_MAX_CONTEXT_CHARS}}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-${DEFAULT_MAX_OUTPUT_TOKENS}}"
START_AT="${START_AT:-}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
FRESH_RUN="${FRESH_RUN:-1}"
REPAIR_FROM_CACHE="${REPAIR_FROM_CACHE:-0}"
if [[ -n "${REPAIR_UPSTREAM_ROOT:-}" ]]; then
    export FRESH_RUN=0 REPAIR_FROM_CACHE=1 SINGLE_GPU_PIPELINE=1
fi
export REPAIR_MODEL="${REPAIR_MODEL:-${MODEL}}"
if [[ "${MODEL_PROFILE}" == "gemma31b" && "${REPAIR_MODEL}" != "${MODEL}" ]]; then
    echo "ERROR: gemma31b experiment requires the same model for upstream and repair."
    exit 2
fi
export REPAIR_BASE_URL="${REPAIR_BASE_URL:-}"
export REPAIR_MAX_OUTPUT_TOKENS="${REPAIR_MAX_OUTPUT_TOKENS:-${MAX_OUTPUT_TOKENS}}"
REPAIR_GPU_ID="${REPAIR_GPU_ID:-}"
REPAIR_VLLM_PORT="${REPAIR_VLLM_PORT:-8001}"
REPAIR_VLLM_MAX_MODEL_LEN="${REPAIR_VLLM_MAX_MODEL_LEN:-${VLLM_MAX_MODEL_LEN:-65536}}"
REPAIR_VLLM_GPU_MEMORY_UTILIZATION="${REPAIR_VLLM_GPU_MEMORY_UTILIZATION:-${VLLM_GPU_MEMORY_UTILIZATION:-0.90}}"
REPAIR_TENSOR_PARALLEL_SIZE="${REPAIR_TENSOR_PARALLEL_SIZE:-1}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
REPAIR_VLLM_MAX_NUM_SEQS="${REPAIR_VLLM_MAX_NUM_SEQS:-${VLLM_MAX_NUM_SEQS}}"
for limit_name in VLLM_MAX_NUM_SEQS REPAIR_VLLM_MAX_NUM_SEQS; do
    if [[ ! "${!limit_name}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: ${limit_name} must be a positive integer."
        exit 2
    fi
done
export REPAIR_THINKING="${REPAIR_THINKING:-1}"
export REPAIR_THINKING_TOKENS="${REPAIR_THINKING_TOKENS:-4096}"
export REPAIR_MAX_TOTAL_TOKENS="${REPAIR_MAX_TOTAL_TOKENS:-0}"
export REPAIR_MAX_ROUNDS="${REPAIR_MAX_ROUNDS:-2}"
export REPAIR_MEASUREMENT_RECONCILIATION="${REPAIR_MEASUREMENT_RECONCILIATION:-1}"
TOOL_TRANSPORT="${TOOL_TRANSPORT:-local}"
ENABLE_TRACING="${ENABLE_TRACING:-0}"

if [[ "${REPAIR_FROM_CACHE}" == "1" && "${FRESH_RUN}" == "1" ]]; then
    echo "ERROR: REPAIR_FROM_CACHE=1 requires FRESH_RUN=0."
    exit 2
fi

PHOENIX_PORT="${PHOENIX_PORT:-6006}"
VLLM_PORT="${VLLM_PORT:-8000}"
MCP_PORT="${MCP_PORT:-8765}"
MCP_PATH="${MCP_PATH:-/mcp}"

VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-${DEFAULT_VLLM_MAX_MODEL_LEN}}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-${DEFAULT_VLLM_GPU_MEMORY_UTILIZATION}}"

PHOENIX_START_TIMEOUT="${PHOENIX_START_TIMEOUT:-120}"
MCP_START_TIMEOUT="${MCP_START_TIMEOUT:-120}"
VLLM_START_TIMEOUT="${VLLM_START_TIMEOUT:-1800}"

if [[ "${BATCH_SPLIT}" != "all" && "${BATCH_SPLIT}" != "dev" && "${BATCH_SPLIT}" != "validation" ]]; then
    echo "ERROR: BATCH_SPLIT must be all, dev, or validation."
    exit 2
fi

if [[ "${TOOL_TRANSPORT}" != "mcp" && "${TOOL_TRANSPORT}" != "local" ]]; then
    echo "ERROR: TOOL_TRANSPORT must be mcp or local."
    exit 2
fi

if [[ "${ENABLE_TRACING}" != "0" && "${ENABLE_TRACING}" != "1" ]]; then
    echo "ERROR: ENABLE_TRACING must be 0 or 1."
    exit 2
fi

if [[ ! -f "extract.py" ]]; then
    echo "ERROR: extract.py not found in ${SCRIPT_DIR}"
    exit 1
fi

if [[ ! -f "${PAPER_LIST}" ]]; then
    echo "ERROR: paper list not found: ${PAPER_LIST}"
    exit 1
fi

if [[ ! -f "SOURCES/schemas/generic.json" || ! -f "SOURCES/schemas/photon_and_neutron.json" ]]; then
    echo "ERROR: required schema files are missing under SOURCES/schemas/."
    exit 1
fi

if [[ "${SINGLE_GPU_PIPELINE:-0}" == "1" && -z "${SINGLE_GPU_PHASE:-}" ]]; then
    exec "${PYTHON}" extract.py single-gpu-run \
        --python "${PYTHON}" --gpu-id "${GPU_ID}" --model "${MODEL}" --repair-model "${REPAIR_MODEL}" \
        --paper-list "${PAPER_LIST}" --split "${BATCH_SPLIT}" --max-papers "${MAX_PAPERS}" \
        --start-at "${START_AT}" --batch-id "${BATCH_RUN_ID}"
fi
UPSTREAM_ONLY="${UPSTREAM_ONLY:-0}"
if [[ "${UPSTREAM_ONLY}" == "1" && ( "${FRESH_RUN}" != "1" || "${REPAIR_FROM_CACHE}" == "1" ) ]]; then
    echo "ERROR: upstream-only requires FRESH_RUN=1 and REPAIR_FROM_CACHE=0."
    exit 2
fi

"${PYTHON}" -m py_compile extract.py
"${PYTHON}" -c 'import extract; extract._pan_repair_runtime_config()'

PHOENIX_URL="http://127.0.0.1:${PHOENIX_PORT}"
PHOENIX_TRACE_ENDPOINT="${PHOENIX_URL}/v1/traces"
VLLM_URL="http://127.0.0.1:${VLLM_PORT}"
VLLM_API_URL="${VLLM_URL}/v1"
SERVE_MODEL="${MODEL}"
SERVE_TENSOR_PARALLEL_SIZE=1
# A cache-only run never loads the upstream model. The CLI --model remains the
# upstream provenance; repair has an explicit, verified model route.
if [[ "${REPAIR_FROM_CACHE}" == "1" && -z "${REPAIR_BASE_URL}" ]]; then
    SERVE_MODEL="${REPAIR_MODEL}"
    SERVE_TENSOR_PARALLEL_SIZE="${REPAIR_TENSOR_PARALLEL_SIZE}"
    VLLM_MAX_NUM_SEQS="${REPAIR_VLLM_MAX_NUM_SEQS}"
    VLLM_MAX_MODEL_LEN="${REPAIR_VLLM_MAX_MODEL_LEN}"
    VLLM_GPU_MEMORY_UTILIZATION="${REPAIR_VLLM_GPU_MEMORY_UTILIZATION}"
    export REPAIR_BASE_URL="${VLLM_API_URL}"
fi
MCP_URL="http://127.0.0.1:${MCP_PORT}${MCP_PATH}"

SERVICE_LOG_DIR="${RUN_OUTPUT_DIR}/service_logs/${BATCH_RUN_ID}"
BATCH_DIR="${RUN_OUTPUT_DIR}/error_analysis/${BATCH_RUN_ID}"
PAPER_LOG_DIR="${BATCH_DIR}/paper_logs"
STATUS_LOG="${BATCH_DIR}/paper_status.jsonl"

mkdir -p "${SERVICE_LOG_DIR}" "${BATCH_DIR}" "${PAPER_LOG_DIR}"

PHOENIX_LOG="${SERVICE_LOG_DIR}/phoenix.log"
VLLM_LOG="${SERVICE_LOG_DIR}/vllm.log"
MCP_LOG="${SERVICE_LOG_DIR}/mcp_server.log"

PHOENIX_PID=""
VLLM_PID=""
MCP_PID=""
OWNS_PHOENIX=0
OWNS_VLLM=0
OWNS_MCP=0
REPAIR_VLLM_PID=""
OWNS_REPAIR_VLLM=0

http_ready() {
    local url="$1"
    "${PYTHON}" - "${url}" <<'PY' >/dev/null 2>&1
import sys
import urllib.request
url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=2) as response:
        status = getattr(response, "status", 200)
        raise SystemExit(0 if status < 500 else 1)
except Exception:
    raise SystemExit(1)
PY
}

tcp_ready() {
    local host="$1"
    local port="$2"
    "${PYTHON}" - "${host}" "${port}" <<'PY' >/dev/null 2>&1
import socket
import sys
host = sys.argv[1]
port = int(sys.argv[2])
try:
    with socket.create_connection((host, port), timeout=2):
        raise SystemExit(0)
except OSError:
    raise SystemExit(1)
PY
}

wait_for_http() {
    local url="$1"
    local pid="$2"
    local timeout="$3"
    local name="$4"
    local elapsed=0
    while ! http_ready "${url}"; do
        if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
            echo "ERROR: ${name} exited during startup."
            return 1
        fi
        if (( elapsed >= timeout )); then
            echo "ERROR: timeout waiting for ${name}: ${url}"
            return 1
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
}

wait_for_tcp() {
    local host="$1"
    local port="$2"
    local pid="$3"
    local timeout="$4"
    local name="$5"
    local elapsed=0
    while ! tcp_ready "${host}" "${port}"; do
        if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
            echo "ERROR: ${name} exited during startup."
            return 1
        fi
        if (( elapsed >= timeout )); then
            echo "ERROR: timeout waiting for ${name} on ${host}:${port}"
            return 1
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
}

terminate_process_group() {
    local pid="$1"
    local name="$2"
    if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
        return
    fi

    echo "[run] Stopping ${name}..."
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true

    for _ in {1..30}; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "[run] ${name} stopped."
            return
        fi
        sleep 1
    done

    kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
}

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM

    echo
    echo "[run] Cleaning up services started by this batch runner..."
    if [[ "${OWNS_REPAIR_VLLM}" -eq 1 ]]; then
        terminate_process_group "${REPAIR_VLLM_PID}" "repair vLLM"
    fi

    if [[ "${OWNS_VLLM}" -eq 1 ]]; then
        terminate_process_group "${VLLM_PID}" "vLLM"
    else
        echo "[run] vLLM was reused; leaving it running."
    fi

    if [[ "${OWNS_MCP}" -eq 1 ]]; then
        terminate_process_group "${MCP_PID}" "MCP metadata-tool server"
    elif [[ "${TOOL_TRANSPORT}" == "mcp" ]]; then
        echo "[run] MCP server was reused; leaving it running."
    fi

    if [[ "${OWNS_PHOENIX}" -eq 1 ]]; then
        terminate_process_group "${PHOENIX_PID}" "Phoenix"
    elif [[ "${ENABLE_TRACING}" == "1" ]]; then
        echo "[run] Phoenix was reused; leaving it running."
    else
        echo "[run] Phoenix tracing disabled."
    fi

    exit "${exit_code}"
}

trap cleanup EXIT INT TERM

record_status() {
    local paper_id="$1"
    local split="$2"
    local complexity="$3"
    local status="$4"
    local message="${5:-}"

    "${PYTHON}" - "${STATUS_LOG}" "${paper_id}" "${split}" "${complexity}" "${status}" "${message}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "paper_id": sys.argv[2],
    "split": sys.argv[3],
    "complexity": sys.argv[4],
    "status": sys.argv[5],
    "message": sys.argv[6],
}
with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
PY
}

# ------------------------------------------------------------
# Start/reuse Phoenix when tracing is enabled.
# Tracing is opt-in; set ENABLE_TRACING=1
# to run extraction without Phoenix/OpenTelemetry.
# ------------------------------------------------------------
if [[ "${ENABLE_TRACING}" == "1" ]]; then
    if http_ready "${PHOENIX_URL}"; then
        echo "[run] Reusing Phoenix at ${PHOENIX_URL}"
    else
        command -v phoenix >/dev/null 2>&1 || {
            echo "ERROR: phoenix command not found."
            exit 1
        }
        echo "[run] Starting Phoenix..."
        setsid phoenix serve \
            --host 127.0.0.1 \
            --port "${PHOENIX_PORT}" \
            >"${PHOENIX_LOG}" 2>&1 &
        PHOENIX_PID=$!
        OWNS_PHOENIX=1
        if ! wait_for_http "${PHOENIX_URL}" "${PHOENIX_PID}" "${PHOENIX_START_TIMEOUT}" "Phoenix"; then
            tail -n 120 "${PHOENIX_LOG}" || true
            exit 1
        fi
    fi
else
    echo "[run] Phoenix tracing disabled by ENABLE_TRACING=0."
fi

# ------------------------------------------------------------
# Start/reuse MCP ONCE for the entire batch.
# ------------------------------------------------------------
mcp_inventory_ready() {
    local url="$1"
    "${PYTHON}" extract.py mcp-check --mcp-url "${url}" >/dev/null 2>&1
}

if [[ "${TOOL_TRANSPORT}" == "mcp" ]]; then
    if tcp_ready 127.0.0.1 "${MCP_PORT}" && mcp_inventory_ready "${MCP_URL}"; then
        echo "[run] Reusing compatible MCP server on port ${MCP_PORT}"
    elif tcp_ready 127.0.0.1 "${MCP_PORT}"; then
        echo "ERROR: MCP port ${MCP_PORT} is occupied but tool inventory is incompatible."
        echo "       Stop the old MCP server or choose a different MCP_PORT."
        exit 1
    else
        echo "[run] Starting MCP metadata-tool server..."
        setsid "${PYTHON}" extract.py mcp-server \
            --host 127.0.0.1 \
            --port "${MCP_PORT}" \
            --path "${MCP_PATH}" \
            >"${MCP_LOG}" 2>&1 &
        MCP_PID=$!
        OWNS_MCP=1
        if ! wait_for_tcp 127.0.0.1 "${MCP_PORT}" "${MCP_PID}" "${MCP_START_TIMEOUT}" "MCP server"; then
            tail -n 160 "${MCP_LOG}" || true
            exit 1
        fi
    fi
else
    echo "[run] TOOL_TRANSPORT=local; MCP server will not be started."
fi

# ------------------------------------------------------------
# Start/reuse vLLM ONCE for the entire batch.
# ------------------------------------------------------------
reasoning_parser_for_model() {
    case "${1,,}" in
        *gemma-4*) echo gemma4 ;;
        *qwen*) echo qwen3 ;;
        *) echo "ERROR: set an explicit reasoning parser for model $1" >&2; return 2 ;;
    esac
}
SERVE_REASONING_PARSER="${VLLM_REASONING_PARSER:-$(reasoning_parser_for_model "${SERVE_MODEL}")}"
REPAIR_REASONING_PARSER="${REPAIR_VLLM_REASONING_PARSER:-$(reasoning_parser_for_model "${REPAIR_MODEL}")}"
if [[ "${REPAIR_FROM_CACHE}" == "1" && "${REPAIR_BASE_URL}" != "${VLLM_API_URL}" ]]; then
    echo "[run] Cache-only run uses external repair endpoint; upstream server is not needed."
elif http_ready "${VLLM_URL}/v1/models"; then
    if [[ "${SINGLE_GPU_PIPELINE:-0}" == "1" ]]; then
        echo "ERROR: single-GPU switching requires an unused vLLM port; refusing to reuse/stop another server."
        exit 2
    fi
    if ! "${PYTHON}" - "${VLLM_URL}/v1/models" "${SERVE_MODEL}" <<'PY'
import json
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1], timeout=15) as response:
    served = [item.get("id") for item in json.load(response).get("data", [])]
if sys.argv[2] not in served:
    raise SystemExit(f"ERROR: requested model {sys.argv[2]!r}, but existing vLLM serves {served!r}. Stop it or use a different VLLM_PORT; no inference was started.")
PY
    then
        exit 1
    fi
    echo "[run] Reusing vLLM at ${VLLM_URL}"
else
    command -v vllm >/dev/null 2>&1 || {
        echo "ERROR: vllm command not found."
        exit 1
    }
    echo "[run] Starting vLLM on physical GPU ${GPU_ID}..."
    echo "[run] Model: ${SERVE_MODEL}"
    setsid env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        vllm serve "${SERVE_MODEL}" \
        --port "${VLLM_PORT}" \
        --max-model-len "${VLLM_MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
        --tensor-parallel-size "${SERVE_TENSOR_PARALLEL_SIZE}" \
        --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
        --reasoning-parser "${SERVE_REASONING_PARSER}" \
        --language-model-only \
        >"${VLLM_LOG}" 2>&1 &
    VLLM_PID=$!
    OWNS_VLLM=1
    if ! wait_for_http "${VLLM_URL}/v1/models" "${VLLM_PID}" "${VLLM_START_TIMEOUT}" "vLLM"; then
        tail -n 180 "${VLLM_LOG}" || true
        exit 1
    fi
fi

# Full runs may use a second owned server or an explicitly supplied endpoint.
if [[ -z "${REPAIR_BASE_URL}" && "${REPAIR_MODEL}" != "${MODEL}" ]]; then
    if [[ -z "${REPAIR_GPU_ID}" || "${REPAIR_VLLM_PORT}" == "${VLLM_PORT}" ]]; then
        echo "ERROR: mixed-model full runs require REPAIR_GPU_ID and a distinct REPAIR_VLLM_PORT, or REPAIR_BASE_URL."
        exit 2
    fi
    "${PYTHON}" - "${GPU_ID}" "${REPAIR_GPU_ID}" <<'PY'
import sys
if set(sys.argv[1].split(',')) & set(sys.argv[2].split(',')):
    raise SystemExit('ERROR: owned upstream and repair servers require disjoint GPUs; alternatively supply REPAIR_BASE_URL.')
PY
    export REPAIR_BASE_URL="http://127.0.0.1:${REPAIR_VLLM_PORT}/v1"
    if ! http_ready "${REPAIR_BASE_URL}/models"; then
        setsid env CUDA_VISIBLE_DEVICES="${REPAIR_GPU_ID}" \
            vllm serve "${REPAIR_MODEL}" --port "${REPAIR_VLLM_PORT}" \
            --max-model-len "${REPAIR_VLLM_MAX_MODEL_LEN}" \
            --gpu-memory-utilization "${REPAIR_VLLM_GPU_MEMORY_UTILIZATION}" \
            --tensor-parallel-size "${REPAIR_TENSOR_PARALLEL_SIZE}" \
            --max-num-seqs "${REPAIR_VLLM_MAX_NUM_SEQS}" \
            --reasoning-parser "${REPAIR_REASONING_PARSER}" --language-model-only \
            >"${SERVICE_LOG_DIR}/vllm_repair.log" 2>&1 &
        REPAIR_VLLM_PID=$!
        OWNS_REPAIR_VLLM=1
        wait_for_http "${REPAIR_BASE_URL}/models" "${REPAIR_VLLM_PID}" "${VLLM_START_TIMEOUT}" "repair vLLM"
    fi
fi
export REPAIR_BASE_URL="${REPAIR_BASE_URL:-${VLLM_API_URL}}"
"${PYTHON}" - "${REPAIR_BASE_URL}/models" "${REPAIR_MODEL}" <<'PY'
import json, sys, urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=15) as response:
    served = [item.get('id') for item in json.load(response).get('data', [])]
if sys.argv[2] not in served:
    raise SystemExit(f'ERROR: requested repair model {sys.argv[2]!r}, endpoint serves {served!r}; no inference started.')
PY

# ------------------------------------------------------------
# Read selected papers.
# Lines use: paper_id split complexity
# ------------------------------------------------------------
if [[ "${REPAIR_THINKING}" == "1" ]]; then
    # Inspect capabilities without an inference request. Do not silently ignore a
    # reasoning budget on an older server and pay for unbounded thinking instead.
    "${PYTHON}" - "${REPAIR_BASE_URL%/v1}/openapi.json" <<'PY'
import json
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1], timeout=15) as response:
    schemas = json.load(response).get("components", {}).get("schemas", {})
supported = any("thinking_token_budget" in schema.get("properties", {}) for schema in schemas.values())
if not supported:
    raise SystemExit("ERROR: this vLLM does not advertise thinking_token_budget. Upgrade vLLM before this run, or explicitly set REPAIR_THINKING=0; no inference was started.")
PY
fi

mapfile -t PAPER_ROWS < <(
    awk -v wanted="${BATCH_SPLIT}" '
        /^[[:space:]]*#/ || /^[[:space:]]*$/ {next}
        {
            split_name = ($2 == "" ? "unspecified" : $2)
            complexity = ($3 == "" ? "unspecified" : $3)
            if (wanted == "all" || split_name == wanted) {
                print $1 "\t" split_name "\t" complexity
            }
        }
    ' "${PAPER_LIST}"
)

if [[ "${#PAPER_ROWS[@]}" -eq 0 ]]; then
    echo "ERROR: no papers selected from ${PAPER_LIST} for split=${BATCH_SPLIT}"
    exit 1
fi

echo
echo "============================================================"
echo " ME AGENT VERIFICATION-REPAIR ERROR-ANALYSIS BATCH"
echo "============================================================"
echo "Paper list:       ${PAPER_LIST}"
echo "Split:            ${BATCH_SPLIT}"
echo "Selected papers:  ${#PAPER_ROWS[@]}"
echo "Batch run id:     ${BATCH_RUN_ID}"
echo "Tool transport:   ${TOOL_TRANSPORT}"
echo "Model profile:    ${MODEL_PROFILE}"
echo "Model:            ${MODEL}"
echo "Repair model:     ${REPAIR_MODEL}"
echo "Repair endpoint:  ${REPAIR_BASE_URL}"
echo "vLLM max len:     ${VLLM_MAX_MODEL_LEN}"
echo "vLLM max seqs:    ${VLLM_MAX_NUM_SEQS}"
echo "Repair max seqs:  ${REPAIR_VLLM_MAX_NUM_SEQS} (applies to runner-owned server)"
echo "GPU memory util:  ${VLLM_GPU_MEMORY_UTILIZATION}"
echo "Fresh run:        ${FRESH_RUN}"
echo "Skip completed:   ${SKIP_COMPLETED}"
echo "Max context chars:${MAX_CONTEXT_CHARS}"
echo "Max output tokens:${MAX_OUTPUT_TOKENS}"
echo "Repair cache:     ${REPAIR_FROM_CACHE}"
echo "Repair thinking:  ${REPAIR_THINKING} (budget=${REPAIR_THINKING_TOKENS})"
echo "Repair rounds:    ${REPAIR_MAX_ROUNDS}; measurement reconciliation=${REPAIR_MEASUREMENT_RECONCILIATION}"
echo "Repair limit: ${REPAIR_MAX_ROUNDS} paper-wide rounds; cumulative tokens are reported, not capped"
"${PYTHON}" -c 'import extract; print("Repair versions: ", extract.PAN_REPAIR_VERSION, extract.PAN_REPAIR_APPLY_VERSION); extract._pan_repair_runtime_config()'
echo "Batch output:     ${BATCH_DIR}"
echo "Paper output:     ${RUN_OUTPUT_DIR} (isolated by ME_OUTPUT_ROOT / ME_RUN_NAME)"
echo "============================================================"
echo

selected_count=0
processed_count=0
start_reached=0
if [[ -z "${START_AT}" ]]; then
    start_reached=1
fi

for row in "${PAPER_ROWS[@]}"; do
    IFS=$'\t' read -r PAPER_ID PAPER_SPLIT PAPER_COMPLEXITY <<<"${row}"

    if [[ "${start_reached}" -eq 0 ]]; then
        if [[ "${PAPER_ID}" == "${START_AT}" ]]; then
            start_reached=1
        else
            continue
        fi
    fi

    if [[ "${MAX_PAPERS}" -gt 0 && "${processed_count}" -ge "${MAX_PAPERS}" ]]; then
        break
    fi

    selected_count=$((selected_count + 1))
    processed_count=$((processed_count + 1))

    PAPER_DIR="${RUN_OUTPUT_DIR}/${PAPER_ID}"
    PREDICTION="${PAPER_DIR}/prediction.json"
    AUDIT="${PAPER_DIR}/audit.json"
    EVALUATION="${PAPER_DIR}/evaluation.json"
    GLOBAL_SEED_PREDICTION="${PAPER_DIR}/global_seed_prediction.json"
    GLOBAL_SEED_EVALUATION="${PAPER_DIR}/global_seed_evaluation.json"
    ERROR_RECORDS="${PAPER_DIR}/error_records.jsonl"
    PAPER_LOG="${PAPER_LOG_DIR}/${PAPER_ID}.log"

    mkdir -p "${PAPER_DIR}"

    echo
    echo "------------------------------------------------------------"
    echo "[run] Paper ${processed_count}: ${PAPER_ID}"
    echo "[run] split=${PAPER_SPLIT} complexity=${PAPER_COMPLEXITY}"
    echo "------------------------------------------------------------"

    if [[ ! -f "SOURCES/input_papers/${PAPER_ID}/paper.pdf" ]]; then
        echo "[run] ERROR: missing paper.pdf; continuing batch."
        record_status "${PAPER_ID}" "${PAPER_SPLIT}" "${PAPER_COMPLEXITY}" "missing_input" "paper.pdf missing"
        continue
    fi

    if [[ "${UPSTREAM_ONLY}" != "1" && ! -f "SOURCES/gold_data/${PAPER_ID}.json" ]]; then
        echo "[run] ERROR: missing gold annotation; continuing batch."
        record_status "${PAPER_ID}" "${PAPER_SPLIT}" "${PAPER_COMPLEXITY}" "missing_gold" "gold_data JSON missing"
        continue
    fi

    if [[ "${SKIP_COMPLETED}" == "1" && "${FRESH_RUN}" != "1" && "${REPAIR_FROM_CACHE}" != "1" \
          && -f "${PREDICTION}" && -f "${EVALUATION}" && -f "${ERROR_RECORDS}" ]]; then
        echo "[run] Complete outputs already exist; skipping inference."
        record_status "${PAPER_ID}" "${PAPER_SPLIT}" "${PAPER_COMPLEXITY}" "skipped_complete" "all expected outputs already exist"
        continue
    fi

    EXTRACT_EXTRA_ARGS=()
    if [[ "${ENABLE_TRACING}" != "1" ]]; then
        EXTRACT_EXTRA_ARGS+=(--no-tracing)
    fi
    if [[ "${FRESH_RUN}" == "1" ]]; then
        EXTRACT_EXTRA_ARGS+=(--fresh)
    fi
    if [[ "${REPAIR_FROM_CACHE}" == "1" ]]; then
        EXTRACT_EXTRA_ARGS+=(--repair-from-cache)
    fi
    if [[ "${UPSTREAM_ONLY}" == "1" ]]; then
        EXTRACT_EXTRA_ARGS+=(--upstream-only)
    fi

    echo "===== EXTRACTION: ${PAPER_ID} =====" | tee -a "${PAPER_LOG}"
    if ! "${PYTHON}" extract.py extract \
        --paper-id "${PAPER_ID}" \
        --model "${MODEL}" \
        --vllm-base-url "${VLLM_API_URL}" \
        --phoenix-endpoint "${PHOENIX_TRACE_ENDPOINT}" \
        --tool-transport "${TOOL_TRANSPORT}" \
        --mcp-url "${MCP_URL}" \
        --max-context-chars "${MAX_CONTEXT_CHARS}" \
        --max-output-tokens "${MAX_OUTPUT_TOKENS}" \
        "${EXTRACT_EXTRA_ARGS[@]}" \
        2>&1 | tee -a "${PAPER_LOG}"; then
        echo "[run] Extraction failed for ${PAPER_ID}; continuing batch."
        record_status "${PAPER_ID}" "${PAPER_SPLIT}" "${PAPER_COMPLEXITY}" "extraction_failed" "see ${PAPER_LOG}"
        continue
    fi

    if [[ "${UPSTREAM_ONLY}" == "1" ]]; then
        record_status "${PAPER_ID}" "${PAPER_SPLIT}" "${PAPER_COMPLEXITY}" "upstream_success" "verified seed/anchor/detector cache; repair not run"
        continue
    fi
    if [[ ! -f "${PREDICTION}" ]]; then
        record_status "${PAPER_ID}" "${PAPER_SPLIT}" "${PAPER_COMPLEXITY}" "extraction_failed" "prediction.json not created"
        continue
    fi

    if [[ -f "${GLOBAL_SEED_PREDICTION}" ]]; then
        echo "===== GLOBAL SEED EVALUATION: ${PAPER_ID} =====" | tee -a "${PAPER_LOG}"
        if ! "${PYTHON}" extract.py evaluate \
            --paper-id "${PAPER_ID}" \
            --prediction "${GLOBAL_SEED_PREDICTION}" \
            --audit "${AUDIT}" \
            --run-tag "run_global_seed_initial" \
            --output "${GLOBAL_SEED_EVALUATION}" \
            2>&1 | tee -a "${PAPER_LOG}"; then
            echo "[run] Global seed evaluation failed for ${PAPER_ID}; continuing to final evaluation."
        fi
    else
        echo "[run] WARNING: global seed snapshot missing for ${PAPER_ID}" | tee -a "${PAPER_LOG}"
    fi

    echo "===== EVALUATION: ${PAPER_ID} =====" | tee -a "${PAPER_LOG}"
    if ! "${PYTHON}" extract.py evaluate \
        --paper-id "${PAPER_ID}" \
        --prediction "${PREDICTION}" \
        --audit "${AUDIT}" \
        --run-tag "run_global_seed_repair" \
        2>&1 | tee -a "${PAPER_LOG}"; then
        echo "[run] Evaluation failed for ${PAPER_ID}; continuing batch."
        record_status "${PAPER_ID}" "${PAPER_SPLIT}" "${PAPER_COMPLEXITY}" "evaluation_failed" "see ${PAPER_LOG}"
        continue
    fi

    echo "===== ERROR RECORDS: ${PAPER_ID} =====" | tee -a "${PAPER_LOG}"
    if ! "${PYTHON}" extract.py error-records \
        --paper-id "${PAPER_ID}" \
        --prediction "${PREDICTION}" \
        --evaluation "${EVALUATION}" \
        --audit "${AUDIT}" \
        --output "${ERROR_RECORDS}" \
        2>&1 | tee -a "${PAPER_LOG}"; then
        echo "[run] Error-record construction failed for ${PAPER_ID}; continuing batch."
        record_status "${PAPER_ID}" "${PAPER_SPLIT}" "${PAPER_COMPLEXITY}" "error_records_failed" "see ${PAPER_LOG}"
        continue
    fi

    record_status "${PAPER_ID}" "${PAPER_SPLIT}" "${PAPER_COMPLEXITY}" "success" "completed extraction, evaluation, and error records"
    echo "[run] SUCCESS: ${PAPER_ID}"
done

if [[ "${UPSTREAM_ONLY}" == "1" ]]; then
    echo "[run] Upstream phase complete; no final evaluation or repair was performed."
    exit 0
fi

# ------------------------------------------------------------
# Aggregate all latest successful/skipped paper results.
# ------------------------------------------------------------
echo
echo "============================================================"
echo " ME AGENT BATCH AGGREGATION"
echo "============================================================"

"${PYTHON}" extract.py aggregate-error-analysis \
    --status "${STATUS_LOG}" \
    --output-dir "${BATCH_DIR}" \
    --paper-list "${PAPER_LIST}" \
    --batch-run-id "${BATCH_RUN_ID}" \
    --split "${BATCH_SPLIT}"

echo
echo "============================================================"
echo " ME AGENT BATCH FINISHED"
echo "============================================================"
echo "Status log:       ${STATUS_LOG}"
echo "Batch summary:    ${BATCH_DIR}/batch_summary.json"
echo "Combined errors:  ${BATCH_DIR}/error_records.jsonl"
echo "Manifest:         ${BATCH_DIR}/manifest.json"
echo "Per-paper logs:   ${PAPER_LOG_DIR}/"
echo
echo "Tip: for a quick smoke test before the full dev set:"
echo "  MAX_PAPERS=2 BATCH_SPLIT=all BATCH_RUN_ID=run_smoke bash run.sh"
echo
