#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$script_dir/.." && pwd)}"

local_env="$script_dir/env.local.sh"
if [ -f "$local_env" ]; then
  # shellcheck source=/dev/null
  source "$local_env"
fi

if [ -n "${GCC_ROOT:-}" ]; then
  export PATH="$GCC_ROOT/bin:$PATH"
  export CC="${CC:-$GCC_ROOT/bin/gcc}"
  export CXX="${CXX:-$GCC_ROOT/bin/g++}"
  if [ -d "$GCC_ROOT/lib64" ]; then
    export LD_LIBRARY_PATH="$GCC_ROOT/lib64:${LD_LIBRARY_PATH:-}"
  fi

  gcc_triplet="${GCC_TRIPLET:-x86_64-focal-linux-gnu}"
  gcc_version="${GCC_VERSION:-16.1.0}"
  gcc_toolchain_root="${GCC_TOOLCHAIN_ROOT:-$(cd "$GCC_ROOT/.." 2>/dev/null && pwd)/x-tools/$gcc_triplet}"
  gcc_frontend_dir="$gcc_toolchain_root/libexec/gcc/$gcc_triplet/$gcc_version"
  gcc_lib_dir="$gcc_toolchain_root/lib/gcc/$gcc_triplet/$gcc_version"
  gcc_sysroot="${GCC_SYSROOT:-$gcc_toolchain_root/$gcc_triplet/sysroot}"
  gcc_runtime_lib="$gcc_toolchain_root/$gcc_triplet/lib64"
  if [ -d "$gcc_runtime_lib" ]; then
    export LD_LIBRARY_PATH="$gcc_runtime_lib:${LD_LIBRARY_PATH:-}"
  fi
  if [ -x "$gcc_frontend_dir/cc1plus" ]; then
    export COMPILER_PATH="$gcc_frontend_dir:${COMPILER_PATH:-}"
    gcc_relocate_flags="-B$gcc_frontend_dir/ -B$gcc_lib_dir/ -B$gcc_sysroot/usr/lib/ -B$gcc_sysroot/lib/ --sysroot=$gcc_sysroot"
    export TYPETORCH_GCC_RELOCATE_LDFLAGS="$gcc_relocate_flags"
    export LDFLAGS="$gcc_relocate_flags ${LDFLAGS:-}"
  fi
fi

if [ -n "${GCC_RUNTIME_LIB:-}" ]; then
  export LD_LIBRARY_PATH="$GCC_RUNTIME_LIB:${LD_LIBRARY_PATH:-}"
fi

for libtorch_lib in "$HOME"/.xmake/packages/l/libtorch-bin/*/*/lib; do
  if [ -d "$libtorch_lib" ]; then
    export LD_LIBRARY_PATH="$libtorch_lib:${LD_LIBRARY_PATH:-}"
    break
  fi
done

if [ -n "${XMAKE_ROOT:-}" ] && [ -d "$XMAKE_ROOT/bin" ]; then
  export PATH="$XMAKE_ROOT/bin:$PATH"
fi

export PYTHON="${PYTHON:-python3}"
if [ -z "${PYTHON_INCLUDE_DIR:-}" ]; then
  python_include="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("include") or "")' 2>/dev/null || true)"
  if [ -n "$python_include" ] && [ -d "$python_include" ]; then
    export PYTHON_INCLUDE_DIR="$python_include"
  fi
fi
cd "$PROJECT_ROOT"
