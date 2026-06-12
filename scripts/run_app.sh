#!/usr/bin/env bash
# Run with your virtual environment already activated.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."

cd "$ROOT_DIR"
python3 src/frontend/app.py "$@"
