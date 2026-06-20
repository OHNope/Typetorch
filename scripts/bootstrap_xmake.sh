#!/usr/bin/env bash
set -euo pipefail

if command -v xmake >/dev/null 2>&1; then
  xmake --version
  exit 0
fi

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
curl -fsSL https://xmake.io/shget.text -o "$workdir/install.sh"
bash "$workdir/install.sh"

cat <<'MSG'
xmake installer finished. If xmake is not on PATH, add the installed bin
directory to PATH or set XMAKE_ROOT before sourcing scripts/env.sh.
MSG
