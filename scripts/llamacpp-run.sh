#!/bin/sh
#
# Run a llama.cpp server for a model.
#
# Accepts either:
#   - A known model alias
#   - A direct path to a GGUF file
#   - An Ollama model name (resolved from blob store)
#
# Usage:
#   ./scripts/llamacpp-run.sh qwen3-coder-next
#   ./scripts/llamacpp-run.sh qwen3.5 --port 8081
#   ./scripts/llamacpp-run.sh ~/models/some-model.gguf
#   ./scripts/llamacpp-run.sh qwen3-coder-next --port 8081
#
# Environment:
#   LLAMA_SERVER  - path to llama-server binary (default: ~/Code/llama.cpp/build/bin/llama-server)
#   MODEL_DIR     - path to models directory (default: ~/models)
#   OLLAMA_DIR    - path to Ollama data dir (default: ~/.ollama)

set -eu

LLAMA_SERVER="${LLAMA_SERVER:-$HOME/Code/llama.cpp/build/bin/llama-server}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
OLLAMA_DIR="${OLLAMA_DIR:-$HOME/.ollama}"
DEFAULT_PORT=8080

# --- Model aliases -----------------------------------------------------------

resolve_alias() {
    case "$1" in
        qwen3-coder-next) echo "${MODEL_DIR}/Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf" ;;
        qwen3.5|qwen35)   echo "${MODEL_DIR}/Qwen3.5-122B-A10B-Q4_K_M/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf" ;;
        qwen3-vl)          echo "${MODEL_DIR}/Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf" ;;
        qwen3-embed)       echo "${MODEL_DIR}/Qwen3-Embedding-4B/Qwen3-Embedding-4B-Q4_K_M.gguf" ;;
        *)                 return 1 ;;
    esac
}

# Some models need extra flags (e.g., vision models need --mmproj)
resolve_alias_extra() {
    case "$1" in
        qwen3-vl)    echo "--mmproj ${MODEL_DIR}/mmproj-Qwen3VL-30B-A3B-Instruct-F16.gguf" ;;
        qwen3-embed) echo "--embedding" ;;
        *)           return 1 ;;
    esac
}

list_aliases() {
    echo "  qwen3-coder-next    Qwen3-Coder-Next 80B Q4_K_M (48GB)"
    echo "  qwen3.5             Qwen3.5-122B-A10B Q4_K_M (76GB)"
    echo "  qwen3-vl            Qwen3-VL-30B vision Q4_K_M (19GB)"
    echo "  qwen3-embed         Qwen3-Embedding-4B Q4_K_M (2.5GB)"
}

# --- Resolve Ollama model name -----------------------------------------------

resolve_ollama() {
    spec="$1"
    case "$spec" in
        *:*) model="${spec%%:*}"; tag="${spec#*:}" ;;
        *)   model="$spec"; tag="latest" ;;
    esac

    manifest="${OLLAMA_DIR}/models/manifests/registry.ollama.ai/library/${model}/${tag}"
    if [ ! -f "$manifest" ]; then
        return 1
    fi

    digest=$(python3 -c "
import json, sys
with open('${manifest}') as f:
    d = json.load(f)
for layer in d.get('layers', []):
    if 'model' in layer.get('mediaType', ''):
        print(layer['digest'])
        sys.exit(0)
sys.exit(1)
" 2>/dev/null) || return 1

    blobpath="${OLLAMA_DIR}/models/blobs/$(echo "$digest" | sed 's/:/-/')"
    if [ -f "$blobpath" ]; then
        echo "$blobpath"
        return 0
    fi
    return 1
}

# --- Resolve model to a file path -------------------------------------------

resolve_model() {
    spec="$1"

    # 1. Direct path to a .gguf file
    case "$spec" in
        *.gguf)
            if [ ! -f "$spec" ]; then
                echo "ERROR: File not found: ${spec}" >&2
                exit 1
            fi
            echo "$spec"
            return
            ;;
    esac

    # 2. Known alias
    resolved=$(resolve_alias "$spec" 2>/dev/null) && {
        if [ -f "$resolved" ]; then
            echo "$resolved"
            return
        else
            echo "ERROR: Alias '${spec}' -> ${resolved} but file not found" >&2
            exit 1
        fi
    }

    # 3. Ollama model name
    resolved=$(resolve_ollama "$spec" 2>/dev/null) && {
        echo "$resolved"
        return
    }

    echo "ERROR: Cannot resolve '${spec}'" >&2
    echo "" >&2
    echo "Aliases:" >&2
    list_aliases >&2
    echo "" >&2
    echo "Ollama models:" >&2
    find "${OLLAMA_DIR}/models/manifests/registry.ollama.ai/library" -type f 2>/dev/null | while read -r f; do
        echo "  $(echo "$f" | sed "s|.*/library/||" | tr '/' ':')" >&2
    done
    exit 1
}

# --- Parse args --------------------------------------------------------------

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model> [--port PORT] [extra llama-server flags...]"
    echo ""
    echo "Aliases:"
    list_aliases
    echo ""
    echo "Examples:"
    echo "  $0 qwen3-coder-next"
    echo "  $0 qwen3.5 --port 8081"
    echo "  $0 ~/models/some-model.gguf"
    exit 1
fi

MODEL_SPEC="$1"
shift

PORT="$DEFAULT_PORT"
EXTRA_FLAGS=""

while [ $# -gt 0 ]; do
    case "$1" in
        --port)
            PORT="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_FLAGS="$*"
            break
            ;;
        *)
            EXTRA_FLAGS="${EXTRA_FLAGS} $1"
            shift
            ;;
    esac
done

# --- Resolve and launch ------------------------------------------------------

MODEL_PATH=$(resolve_model "$MODEL_SPEC")

# Pick up alias-specific flags (e.g., --mmproj for vision models)
ALIAS_EXTRA=$(resolve_alias_extra "$MODEL_SPEC" 2>/dev/null) || true
if [ -n "$ALIAS_EXTRA" ]; then
    EXTRA_FLAGS="${ALIAS_EXTRA} ${EXTRA_FLAGS}"
fi

echo "model:  ${MODEL_SPEC}"
echo "file:   ${MODEL_PATH}"
echo "port:   ${PORT}"
if [ -n "$EXTRA_FLAGS" ]; then
    echo "extra:  ${EXTRA_FLAGS}"
fi
echo ""

exec "$LLAMA_SERVER" \
    -m "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    $EXTRA_FLAGS
