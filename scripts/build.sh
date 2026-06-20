#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

mode="${1:-debug}"
target="${2:-tenspec_tensor_arithmetic_test}"

xmake f -m "$mode" --yes --python="$PYTHON"
xmake build "$target"
