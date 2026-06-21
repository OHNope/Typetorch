# Copy this file to scripts/env.local.sh and edit it for your machine.
# scripts/env.local.sh is ignored by git.

# Optional compiler prefix. If set, scripts/env.sh prepends "$GCC_ROOT/bin" to
# PATH and sets CC/CXX when they are not already set.
export GCC_ROOT="/path/to/gcc-prefix"

# Optional relocated GCC toolchain metadata. These are only needed when your GCC
# driver cannot find crtbeginS.o/crti.o without explicit -B/--sysroot flags.
export GCC_TOOLCHAIN_ROOT="/path/to/x-tools/x86_64-focal-linux-gnu"
export GCC_TRIPLET="x86_64-focal-linux-gnu"
export GCC_VERSION="16.1.0"
export GCC_SYSROOT="$GCC_TOOLCHAIN_ROOT/$GCC_TRIPLET/sysroot"

# Optional runtime/library paths.
export GCC_RUNTIME_LIB="$GCC_TOOLCHAIN_ROOT/$GCC_TRIPLET/lib64"
export XMAKE_ROOT="/path/to/xmake-prefix"

# Optional Python selection. PYTHON_INCLUDE_DIR can usually be omitted because
# scripts/env.sh derives it from Python automatically.
export PYTHON="python3"
export PYTHON_INCLUDE_DIR="/path/to/python/include"
