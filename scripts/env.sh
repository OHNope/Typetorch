#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$script_dir/.." && pwd)}"

if [ -z "${GCC_ROOT:-}" ]; then
  for candidate in \
    /work/7/uw07387/opt/gcc-17 \
    /work/7/uw07387/opt/gcc-17.0.0 \
    /work/7/uw07387/opt/gcc-17-snapshot \
    /work/7/uw07387/opt/gcc-16.1.0; do
    if [ -x "$candidate/bin/g++" ]; then
      export GCC_ROOT="$candidate"
      break
    fi
  done
fi

if [ -n "${GCC_ROOT:-}" ]; then
  export PATH="$GCC_ROOT/bin:$PATH"
  export CC="${CC:-$GCC_ROOT/bin/gcc}"
  export CXX="${CXX:-$GCC_ROOT/bin/g++}"
  if [ -d "$GCC_ROOT/lib64" ]; then
    export LD_LIBRARY_PATH="$GCC_ROOT/lib64:${LD_LIBRARY_PATH:-}"
  fi

  gcc_toolchain_root="$(cd "$GCC_ROOT/.." 2>/dev/null && pwd)/x-tools/x86_64-focal-linux-gnu"
  gcc_frontend_dir="$gcc_toolchain_root/libexec/gcc/x86_64-focal-linux-gnu/16.1.0"
  gcc_lib_dir="$gcc_toolchain_root/lib/gcc/x86_64-focal-linux-gnu/16.1.0"
  gcc_sysroot="$gcc_toolchain_root/x86_64-focal-linux-gnu/sysroot"
  gcc_runtime_lib="$gcc_toolchain_root/x86_64-focal-linux-gnu/lib64"
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
cd "$PROJECT_ROOT"
