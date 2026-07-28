#!/usr/bin/env bash
#
# Boot llama.cpp server instances for the "llamacpp" family.
#
# Each role gets its own server on a dedicated port.
# Edit MODEL_DIR and model paths to match your setup.
#
# Usage:
#   ./scripts/llamacpp-boot.sh start   # start all servers
#   ./scripts/llamacpp-boot.sh stop    # kill all servers
#   ./scripts/llamacpp-boot.sh status  # check what's running
#
# After starting, export the env:
#   source scripts/llamacpp-env.sh

set -euo pipefail

# --- Configuration -----------------------------------------------------------

LLAMA_SERVER="${LLAMA_SERVER:-./build/bin/llama-server}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
GPU_LAYERS="${GPU_LAYERS:-99}"           # offload all layers to GPU
LOG_DIR="${LOG_DIR:-/tmp/llamacpp-logs}"

# Role -> port mapping
# One server per role. Adjust models/ports as needed.
declare -A PORTS=(
    [coder]=8080
    [oracle]=8081
    [vision]=8082
    [embed]=8083
)

# Role -> model file
# Update these paths to your actual GGUF files.
declare -A MODELS=(
    [coder]="${MODEL_DIR}/qwen3-coder-32b-q4_k_m.gguf"
    [oracle]="${MODEL_DIR}/qwen3-coder-32b-q8_0.gguf"
    [vision]="${MODEL_DIR}/qwen3-vl-30b-q4_k_m.gguf"
    [embed]="${MODEL_DIR}/qwen3-embedding-4b.gguf"
)

# Role -> extra flags (optional)
declare -A EXTRA_FLAGS=(
    [coder]=""
    [oracle]=""
    [vision]=""
    [embed]="--embedding"
)

# --- Functions ---------------------------------------------------------------

start_server() {
    local role=$1
    local port=${PORTS[$role]}
    local model=${MODELS[$role]}
    local extra=${EXTRA_FLAGS[$role]:-}
    local pidfile="/tmp/llamacpp-${role}.pid"
    local logfile="${LOG_DIR}/${role}.log"

    if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        echo "[skip] ${role} already running (pid $(cat "$pidfile")) on port ${port}"
        return
    fi

    if [[ ! -f "$model" ]]; then
        echo "[WARN] ${role}: model not found at ${model} -- skipping"
        return
    fi

    mkdir -p "$LOG_DIR"

    echo "[boot] ${role} -> port ${port} ($(basename "$model"))"
    nohup "$LLAMA_SERVER" \
        -m "$model" \
        -ngl "$GPU_LAYERS" \
        --host 0.0.0.0 \
        --port "$port" \
        $extra \
        > "$logfile" 2>&1 &

    echo $! > "$pidfile"
    echo "       pid=$! log=${logfile}"
}

stop_server() {
    local role=$1
    local pidfile="/tmp/llamacpp-${role}.pid"

    if [[ -f "$pidfile" ]]; then
        local pid
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo "[stop] ${role} (pid ${pid})"
            kill "$pid"
        else
            echo "[skip] ${role} not running"
        fi
        rm -f "$pidfile"
    else
        echo "[skip] ${role} no pidfile"
    fi
}

status_server() {
    local role=$1
    local port=${PORTS[$role]}
    local pidfile="/tmp/llamacpp-${role}.pid"

    if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        local pid
        pid=$(cat "$pidfile")
        # Check health endpoint
        if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "[  OK  ] ${role}  port=${port}  pid=${pid}"
        else
            echo "[ BOOT ] ${role}  port=${port}  pid=${pid}  (still loading)"
        fi
    else
        echo "[ DOWN ] ${role}  port=${port}"
    fi
}

# --- Main --------------------------------------------------------------------

ACTION="${1:-status}"

case "$ACTION" in
    start)
        for role in "${!PORTS[@]}"; do
            start_server "$role"
        done
        echo ""
        echo "Servers booting. Run 'source scripts/llamacpp-env.sh' to configure env."
        ;;
    stop)
        for role in "${!PORTS[@]}"; do
            stop_server "$role"
        done
        ;;
    status)
        for role in "${!PORTS[@]}"; do
            status_server "$role"
        done
        ;;
    *)
        echo "Usage: $0 {start|stop|status}"
        exit 1
        ;;
esac
