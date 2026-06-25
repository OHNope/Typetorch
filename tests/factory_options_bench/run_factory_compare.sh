#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/work/7/uw07387/libtorch-reflect-wrapper
cd "$PROJECT_DIR"
source scripts/env.sh

XMAKE_LUA=xmake.lua
CORE_MPP=src/typetorch/tensor/core.mpp
FACTORY_MPP=src/typetorch/tensor/factory.mpp

CORE_VAR=tests/core_static_var.mpp
CORE_FUNC=tests/core_static_func.mpp
FACTORY_FUNC=tests/factory_static_func.mpp

TEST_VAR=tests/blob_static_var.cpp
TEST_FUNC=tests/blob_static_func.cpp

RESULT_DIR=/tmp/factory_compare_results
rm -rf "$RESULT_DIR"
mkdir -p "$RESULT_DIR"

for f in "$CORE_VAR" "$CORE_FUNC" "$FACTORY_FUNC" "$TEST_VAR" "$TEST_FUNC"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found!"
        exit 1
    fi
done

echo "============================================"
echo "  factory_options: static var vs static func"
echo "============================================"
echo

cp "$CORE_MPP" "$RESULT_DIR/core.mpp.orig"
cp "$FACTORY_MPP" "$RESULT_DIR/factory.mpp.orig"
cp "$XMAKE_LUA" "$RESULT_DIR/xmake.lua.orig"

restore_originals() {
    cp "$RESULT_DIR/core.mpp.orig" "$CORE_MPP"
    cp "$RESULT_DIR/factory.mpp.orig" "$FACTORY_MPP"
    cp "$RESULT_DIR/xmake.lua.orig" "$XMAKE_LUA"
}
trap restore_originals EXIT

TARGET_NAME=factory_blob_test

insert_target() {
    local test_file=$1
    python3 - "$XMAKE_LUA" "$TARGET_NAME" "$test_file" << 'PYEOF'
import sys
xmake_lua = sys.argv[1]
target_name = sys.argv[2]
test_file = sys.argv[3]

with open(xmake_lua) as f:
    lines = f.readlines()

# Find the last line that's not empty
while lines and lines[-1].strip() == '':
    lines.pop()

block = f'''
target("{target_name}")
    configure_libtorch_target({{
        default = false,
        files = {{"{test_file}"}},
    }})
target_end()
'''

with open(xmake_lua, 'w') as f:
    for line in lines:
        f.write(line)
    f.write(block)
    f.write('\n')
PYEOF
}

# ============= TEST A: static var =============
echo ">>> TEST A: static inline member variable (current approach)"
echo

cp "$RESULT_DIR/core.mpp.orig" "$CORE_MPP"
cp "$RESULT_DIR/factory.mpp.orig" "$FACTORY_MPP"

insert_target "$TEST_VAR"

echo "Cleaning and building..."
xmake clean > /dev/null 2>&1 || true
START_A=$SECONDS
xmake build -a "$TARGET_NAME" 2>&1 | tail -5
TIME_A=$((SECONDS - START_A))
echo "Build time: ${TIME_A}s"

BIN_A=$(find build -name "$TARGET_NAME" -type f -executable 2>/dev/null | head -1)
if [ -z "$BIN_A" ]; then
    BIN_A=$(find build -name "$TARGET_NAME" -type f 2>/dev/null | head -1)
fi

if [ -n "$BIN_A" ] && [ -f "$BIN_A" ]; then
    SIZE_A=$(stat -c%s "$BIN_A" 2>/dev/null || stat -f%z "$BIN_A" 2>/dev/null || echo 0)
    SYM_A=$(nm -C "$BIN_A" 2>/dev/null | wc -l)
    SIZE_OUT_A=$(size "$BIN_A" 2>/dev/null)
    echo "Binary: $BIN_A"
    echo "Binary size: $SIZE_A bytes"
    echo "Symbol count (nm): $SYM_A"
    echo "Size output:"
    echo "$SIZE_OUT_A"
    cp "$BIN_A" "$RESULT_DIR/bin_static_var"
else
    echo "ERROR: Binary A not found!"
    SIZE_A=0; SYM_A=0
fi

cp "$RESULT_DIR/xmake.lua.orig" "$XMAKE_LUA"

echo
echo "========================================"
echo

# ============= TEST B: static func =============
echo ">>> TEST B: static inline member function (proposed approach)"
echo

cp "$CORE_FUNC" "$CORE_MPP"
cp "$FACTORY_FUNC" "$FACTORY_MPP"

insert_target "$TEST_FUNC"

echo "Cleaning and building..."
xmake clean > /dev/null 2>&1 || true
START_B=$SECONDS
xmake build -a "$TARGET_NAME" 2>&1 | tail -5
TIME_B=$((SECONDS - START_B))
echo "Build time: ${TIME_B}s"

BIN_B=$(find build -name "$TARGET_NAME" -type f -executable 2>/dev/null | head -1)
if [ -z "$BIN_B" ]; then
    BIN_B=$(find build -name "$TARGET_NAME" -type f 2>/dev/null | head -1)
fi

if [ -n "$BIN_B" ] && [ -f "$BIN_B" ]; then
    SIZE_B=$(stat -c%s "$BIN_B" 2>/dev/null || stat -f%z "$BIN_B" 2>/dev/null || echo 0)
    SYM_B=$(nm -C "$BIN_B" 2>/dev/null | wc -l)
    SIZE_OUT_B=$(size "$BIN_B" 2>/dev/null)
    echo "Binary: $BIN_B"
    echo "Binary size: $SIZE_B bytes"
    echo "Symbol count (nm): $SYM_B"
    echo "Size output:"
    echo "$SIZE_OUT_B"
    cp "$BIN_B" "$RESULT_DIR/bin_static_func"
else
    echo "ERROR: Binary B not found!"
    SIZE_B=0; SYM_B=0
fi

echo
echo "============================================"
echo "                 SUMMARY"
echo "============================================"
echo
printf "%-30s %15s %15s %15s\n" "Metric" "Static Var (A)" "Static Func (B)" "Diff (B-A)"
printf "%-30s %15s %15s %15s\n" "------------------------------" "---------------" "---------------" "---------------"
printf "%-30s %15d %15d %+15d\n" "Build time (seconds)" "$TIME_A" "$TIME_B" $((TIME_B - TIME_A))
printf "%-30s %15d %15d %+15d\n" "Binary size (bytes)" "$SIZE_A" "$SIZE_B" $((SIZE_B - SIZE_A))
printf "%-30s %15d %15d %+15d\n" "Symbol count (nm -C)" "$SYM_A" "$SYM_B" $((SYM_B - SYM_A))

echo
echo "Results saved in $RESULT_DIR/"
echo
echo "--- Test A: static var ---"
nm -C "$RESULT_DIR/bin_static_var" 2>/dev/null | grep -i 'factory_option\|TensorOption' | head -20 || echo "(no matching symbols)"
echo
echo "--- Test B: static func ---"
nm -C "$RESULT_DIR/bin_static_func" 2>/dev/null | grep -i 'parameter_option\|factory_option\|TensorOption' | head -20 || echo "(no matching symbols)"
