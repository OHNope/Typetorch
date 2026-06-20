#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

check_cmd() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    printf '[ok]      %s: %s\n' "$name" "$(command -v "$name")"
  else
    printf '[missing] %s: not found in PATH\n' "$name"
  fi
}

check_path() {
  local name="$1"
  local path="$2"
  if [ -e "$path" ]; then
    printf '[ok]      %s: %s\n' "$name" "$path"
  else
    printf '[missing] %s: %s\n' "$name" "$path"
  fi
}

check_cmd "${CC:-gcc}"
check_cmd "${CXX:-g++}"
check_cmd xmake
check_cmd "$PYTHON"
check_path project "$PROJECT_ROOT"

"${CXX:-g++}" --version | head -n 1 || true
xmake --version | head -n 1 || true
"$PYTHON" --version || true
"$PYTHON" - <<'PYINFO' || true
import sys
import sysconfig
print(f"python executable: {sys.executable}")
print(f"python include: {sysconfig.get_path('include')}")
print(f"python extension suffix: {sysconfig.get_config_var('EXT_SUFFIX') or '.so'}")
PYINFO
