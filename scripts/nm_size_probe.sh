#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

mode="${1:-debug}"
max_lines="${MAX_LINES:-80}"
native_target="binary_size_libtorch_probe"
checked_target="binary_size_tensor_checked_probe"
unsafe_target="binary_size_tensor_unsafe_probe"

targets=("$native_target" "$checked_target" "$unsafe_target")

for target in "${targets[@]}"; do
  xmake build -j "${BUILD_JOBS:-2}" "$target"
done

bin_path() {
  printf 'build/linux/x86_64/%s/%s' "$mode" "$1"
}

for target in "${targets[@]}"; do
  if [ ! -x "$(bin_path "$target")" ]; then
    echo "failed to locate size probe binary: $(bin_path "$target")" >&2
    exit 1
  fi
done

native_probe="$(bin_path "$native_target")"
checked_probe="$(bin_path "$checked_target")"
unsafe_probe="$(bin_path "$unsafe_target")"

echo "-- binaries --"
ls -lh "$native_probe" "$checked_probe" "$unsafe_probe"

echo
echo "-- size --"
size "$native_probe" "$checked_probe" "$unsafe_probe" || true

echo
echo "-- native project symbols --"
nm -S --size-sort --demangle "$native_probe" 2>/dev/null |
  sed -n '/typetorch_binary_size_libtorch_probe/p' |
  tail -n "$max_lines"

for entry in \
  "$checked_probe typetorch_binary_size_tensor_checked_probe checked" \
  "$unsafe_probe typetorch_binary_size_tensor_unsafe_probe unsafe"; do
  set -- $entry
  probe="$1"
  namespace="$2"
  label="$3"

  echo
  echo "-- typetorch $label project symbols --"
  nm -S --size-sort --demangle "$probe" 2>/dev/null |
    sed -n '/typetorch/p' |
    tail -n "$max_lines"
done

symbol_count() {
  local probe="$1"
  local pattern="$2"
  nm -S --size-sort --demangle "$probe" 2>/dev/null |
    awk -v pattern="$pattern" '$0 ~ pattern { count += 1 } END { print count + 0 }'
}

symbol_bytes() {
	local probe="$1"
	local pattern="$2"

	nm -S --size-sort --demangle "$probe" 2>/dev/null |
		"${PYTHON:-python3}" -c '
import re
import sys

pattern = re.compile(sys.argv[1])
total = 0

for line in sys.stdin:
    if not pattern.search(line):
        continue

    parts = line.split()
    if len(parts) < 2:
        continue

    try:
        total += int(parts[1], 16)
    except ValueError:
        pass

print(total)
' "$pattern"
}

tensor_bytes() {
	local probe="$1"
	local namespace="$2"

	nm -S --size-sort --demangle "$probe" 2>/dev/null |
		"${PYTHON:-python3}" -c '
import sys

namespace = sys.argv[1]
needle = "typetorchW9typetorch6Tensor"
total = 0

for line in sys.stdin:
    if namespace in line:
        continue
    if needle not in line:
        continue

    parts = line.split()
    if len(parts) < 2:
        continue

    try:
        total += int(parts[1], 16)
    except ValueError:
        pass

print(total)
' "$namespace"
}

tensor_unique_bytes() {
	local probe="$1"
	local namespace="$2"

	nm -S --size-sort --demangle "$probe" 2>/dev/null |
		"${PYTHON:-python3}" -c '
import sys

namespace = sys.argv[1]
needle = "typetorchW9typetorch6Tensor"
seen = set()
total = 0

for line in sys.stdin:
    if namespace in line:
        continue
    if needle not in line:
        continue

    parts = line.split()
    if len(parts) < 2:
        continue

    symbol_addr = parts[0]
    if symbol_addr in seen:
        continue
    seen.add(symbol_addr)

    try:
        total += int(parts[1], 16)
    except ValueError:
        pass

print(total)
' "$namespace"
}

tensor_count() {
  local probe="$1"
  local namespace="$2"
  nm -S --size-sort --demangle "$probe" 2>/dev/null |
    awk -v ns="$namespace" '$0 !~ ns && /typetorchW9typetorch6Tensor/ { count += 1 } END { print count + 0 }'
}


tensor_unique_count() {
  local probe="$1"
  local namespace="$2"
  nm -S --size-sort --demangle "$probe" 2>/dev/null |
    awk -v ns="$namespace" '$0 !~ ns && /typetorchW9typetorch6Tensor/ && !seen[$1]++ { count += 1 } END { print count + 0 }'
}


runtime_pattern='check_shape_compatible_runtime|check_tensor_compatible_runtime|dtype_matches|device_matches|layout_matches'

echo
echo "-- symbol summary --"
native_symbol_count="$(symbol_count "$native_probe" 'typetorch_binary_size_libtorch_probe')"
native_symbol_bytes="$(symbol_bytes "$native_probe" 'typetorch_binary_size_libtorch_probe')"
printf "native_probe_functions bytes=%s symbols=%s\n" "$native_symbol_bytes" "$native_symbol_count"

for entry in \
  "$checked_probe typetorch_binary_size_tensor_checked_probe checked" \
  "$unsafe_probe typetorch_binary_size_tensor_unsafe_probe unsafe"; do
  set -- $entry
  probe="$1"
  namespace="$2"
  label="$3"

  probe_count="$(symbol_count "$probe" "$namespace")"
  probe_bytes="$(symbol_bytes "$probe" "$namespace")"
  wrapper_count="$(tensor_count "$probe" "$namespace")"
  wrapper_bytes="$(tensor_bytes "$probe" "$namespace")"
  wrapper_unique_count="$(tensor_unique_count "$probe" "$namespace")"
  wrapper_unique_bytes="$(tensor_unique_bytes "$probe" "$namespace")"
  runtime_count="$(symbol_count "$probe" "$runtime_pattern")"
  runtime_bytes="$(symbol_bytes "$probe" "$runtime_pattern")"

  printf "%s_probe_functions bytes=%s symbols=%s\n" "$label" "$probe_bytes" "$probe_count"
  printf "%s_tensor_wrapper bytes=%s symbols=%s\n" "$label" "$wrapper_bytes" "$wrapper_count"
  printf "%s_tensor_wrapper_unique bytes=%s symbols=%s\n" "$label" "$wrapper_unique_bytes" "$wrapper_unique_count"
  printf "%s_runtime_checks bytes=%s symbols=%s\n" "$label" "$runtime_bytes" "$runtime_count"
done
