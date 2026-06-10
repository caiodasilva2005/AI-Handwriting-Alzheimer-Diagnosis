#!/usr/bin/env bash
# Run with your virtual environment already activated.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/../src"

PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 "$SRC_DIR/testing/test_cnn.py" "$@"
