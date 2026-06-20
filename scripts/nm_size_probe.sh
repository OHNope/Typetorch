#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

mode="${1:-debug}"
max_lines="${MAX_LINES:-80}"

xmake f -m "$mode" --yes --python="$PYTHON"
xmake build binary_size_tensor_probe

probe="$(find build -type f -name binary_size_tensor_probe -perm -111 | sort | tail -n 1)"
if [ -z "$probe" ]; then
  echo "failed to locate binary_size_tensor_probe under build/" >&2
  exit 1
fi

ls -lh "$probe"
size "$probe" || true

echo
echo "-- largest project symbols --"
nm -S --size-sort --demangle "$probe" |
  sed -n '/tenspec/p' |
  tail -n "$max_lines"
