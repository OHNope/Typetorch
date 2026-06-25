#!/usr/bin/env bash
set -euo pipefail

project_dir=/work/7/uw07387/libtorch-reflect-wrapper
bench_dir="$project_dir/tests/index_range_bench"
result_dir="$bench_dir/results-lite"
rounds=${ROUNDS:-5}
runtime_rounds=${RUNTIME_ROUNDS:-7}

cd "$project_dir"
source scripts/env.sh
rm -rf "$result_dir"
mkdir -p "$result_dir"
read -r -a relocate_flags <<< "${TYPETORCH_GCC_RELOCATE_LDFLAGS:-}"

compile_once() {
	local name=$1
	local define=$2
	local round=$3
	local output="$result_dir/$name"

	start_ns=$(date +%s%N)
	"$CXX" -std=c++26 -freflection -O2 -ffunction-sections \
		-fdata-sections -Wl,--gc-sections \
		-DINDEX_RANGE_USE_IOTA="$define" \
		"$bench_dir/index_range_bench_lite.cpp" -o "$output" \
		"${relocate_flags[@]}"
	end_ns=$(date +%s%N)
	awk -v round="$round" -v start="$start_ns" -v end="$end_ns" \
		'BEGIN {printf "%d\t%.6f\n", round, (end - start) / 1000000000}' \
		>> "$result_dir/$name.compile.tsv"
}

for round in $(seq 1 "$rounds"); do
	if ((round % 2)); then
		compile_once index_array 0 "$round"
		compile_once iota_view 1 "$round"
	else
		compile_once iota_view 1 "$round"
		compile_once index_array 0 "$round"
	fi
done

for name in index_array iota_view; do
	bin="$result_dir/$name"
	stat -c '%s' "$bin" > "$result_dir/$name.file_size"
	size "$bin" > "$result_dir/$name.sections"
	nm -C --defined-only "$bin" > "$result_dir/$name.symbols"
	objcopy --dump-section .text="$result_dir/$name.text.bin" "$bin"
	: > "$result_dir/$name.runtime.tsv"
	for round in $(seq 1 "$runtime_rounds"); do
		start_ns=$(date +%s%N)
		"$bin"
		end_ns=$(date +%s%N)
		awk -v round="$round" -v start="$start_ns" -v end="$end_ns" \
			'BEGIN {printf "%d\t%.6f\n", round, (end - start) / 1000000000}' \
			>> "$result_dir/$name.runtime.tsv"
	done
done

for metric in compile runtime; do
	echo "$metric results:"
	for name in index_array iota_view; do
		awk -v name="$name" \
			'{sum += $2} END {printf "%-12s mean %.6f s (%d rounds)\n", name, sum / NR, NR}' \
			"$result_dir/$name.$metric.tsv"
	done
done

echo "size results:"
for name in index_array iota_view; do
	printf '%-12s file=%s defined_symbols=%s\n' \
		"$name" "$(cat "$result_dir/$name.file_size")" \
		"$(wc -l < "$result_dir/$name.symbols")"
	cat "$result_dir/$name.sections"
done

if cmp -s "$result_dir/index_array.text.bin" "$result_dir/iota_view.text.bin"; then
	echo ".text bytes: identical"
else
	echo ".text bytes: differ"
fi
