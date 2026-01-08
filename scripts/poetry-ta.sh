#!/usr/bin/env bash
set -euo pipefail

# Helper to run Poetry with TA-Lib paths preconfigured on macOS (arm64)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export CPPFLAGS="-I/opt/homebrew/opt/ta-lib/include"
export LDFLAGS="-L${ROOT}/.talib/lib"
export TA_INCLUDE_PATH="/opt/homebrew/opt/ta-lib/include"
export TA_LIBRARY_PATH="${ROOT}/.talib/lib"

cd "$ROOT"
poetry install --no-root "$@"
