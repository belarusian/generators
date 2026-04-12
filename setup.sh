#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="${PYTHON:-python3}"

cd "$ROOT_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "[setup] creating virtual environment at .venv"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "[setup] using existing virtual environment at .venv"
fi

if ! "$VENV_DIR/bin/python" -c "import setuptools" >/dev/null 2>&1; then
  echo "[setup] installing packaging prerequisites"
  "$VENV_DIR/bin/python" -m pip install setuptools wheel
fi

echo "[setup] installing Compass in editable dev mode"
"$VENV_DIR/bin/python" -m pip install --no-build-isolation -e ".[dev]"

cat <<'EOF'

[setup] done

Next steps:
  source .venv/bin/activate
  generate --list
  pytest

Configuration:
  Compass optionally loads shared environment variables from ~/.compass/.env
EOF
