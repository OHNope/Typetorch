#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$script_dir/.." && pwd)}"

if [ -n "${GCC_ROOT:-}" ]; then
  export PATH="$GCC_ROOT/bin:$PATH"
  export CC="${CC:-$GCC_ROOT/bin/gcc}"
  export CXX="${CXX:-$GCC_ROOT/bin/g++}"
  if [ -d "$GCC_ROOT/lib64" ]; then
    export LD_LIBRARY_PATH="$GCC_ROOT/lib64:${LD_LIBRARY_PATH:-}"
  fi
fi

if [ -n "${GCC_RUNTIME_LIB:-}" ]; then
  export LD_LIBRARY_PATH="$GCC_RUNTIME_LIB:${LD_LIBRARY_PATH:-}"
fi

if [ -n "${XMAKE_ROOT:-}" ]; then
  export PATH="$XMAKE_ROOT/bin:$PATH"
fi

export PYTHON="${PYTHON:-python3}"
cd "$PROJECT_ROOT"
