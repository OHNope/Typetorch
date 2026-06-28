# Performance and Size Notes

Last updated: 2026-06-28.

This project keeps LibTorch as the runtime backend. The wrapper should add type
contracts and compile-time validation while keeping forwarding overhead small.

The current checked-in measurements are also summarized in the README. This file
explains how to reproduce them and how to interpret the numbers.

## Build Measurement Targets

The binary-size probes are measurement targets, not CI gates. Build and run
them on an interactive machine where the toolchain and LibTorch package cache
are already warm.

```bash
source scripts/env.sh
xmake f -m release --yes --python="${PYTHON:-python3}"
xmake build typetorch_forwarding_benchmark
xmake build binary_size_libtorch_probe
xmake build binary_size_tensor_checked_probe
xmake build binary_size_tensor_unsafe_probe
xmake build typetorch_tensor_arithmetic_test
```

For symbol-level inspection, use the debug/no-inline probes:

```bash
bash scripts/nm_size_probe.sh debug
```

Probe roles:

| Target | Role |
| --- | --- |
| `typetorch_forwarding_benchmark` | Forwarding overhead benchmark comparing raw vs typed tensor ops. |
| `typetorch_tensor_arithmetic_test` | Full tensor arithmetic test suite exercising all typed ops. |
| `binary_size_libtorch_probe` | Native LibTorch baseline using only `torch::Tensor`. |
| `binary_size_tensor_checked_probe` | Typetorch path where every raw tensor enters via `retain()` and pays runtime shape/dtype/device/layout checks. |
| `binary_size_tensor_unsafe_probe` | Typetorch path where every raw tensor enters via `unsafe_retain()`; this models a trusted caller that guarantees the contract. |
| `binary_size_tensor_probe` | Legacy mixed probe that uses both checked and unsafe retain paths; keep it only for old-result comparison. |

## Compiler Attribute & Extension Optimization (2026-06-24)

The `extension-attribute-test` branch adds compiler hints to improve binary size
and runtime performance. These are configured in `src/attributes.inc` and applied
across all hot-path and error-path code.

### Attributes added

| Attribute | Macro | Applied to |
| --- | --- | --- |
| `hot` | `TYPETORCH_HOT` | All arithmetic ops, view ops, NN ops, factory ops, `forward()` methods |
| `cold` | `TYPETORCH_COLD` | All `noreturn` contract-violation functions, dtype/device/layout name helpers |
| `pure` | `TYPETORCH_PURE` | Runtime check predicates (`dtype_matches`, `device_matches`, `layout_matches`) |
| `always_inline` | `TYPETORCH_ALWAYS_INLINE` | Hot-path accessors (`unsafe_raw`, `unsafe_retain`, `unsafe_move`, `unsafe_wrap`) |
| `unlikely` | `TYPETORCH_UNLIKELY` | All runtime-contract-check branches |
| `returns_nonnull` | `TYPETORCH_RETURNS_NONNULL` | Functions that never return null |

### Build flags added

- `-ffunction-sections -fdata-sections`: Place each function/data item in its own ELF section.
- `-Wl,--gc-sections`: Linker garbage-collection of unreferenced input sections.

Together these allow the linker to discard unused code paths (e.g. error-reporting
functions that are never called in a given probe binary).

## Forwarding Benchmark

Run:

```bash
build/linux/x86_64/release/typetorch_forwarding_benchmark 50000
```

The executable prints lines like:

```text
operation, raw_ns=<raw>, typetorch_ns=<typed>, ratio=<typed/raw>
```

Ratio is `Typetorch / raw torch::Tensor`. A ratio close to `1.0` means the wrapper has
no measurable forwarding cost for that operation under the current build.

### Current snapshot (2026-06-28, with attribute/extensions, release, 50000 iterations)

| Operation | Raw ns/op | Typetorch ns/op | Ratio | Overhead |
| --- | ---: | ---: | ---: | ---: |
| `add.dynamic` | 1176.03 | 1180.86 | 1.0041 | +0.41% |
| `add.static` | 1181.66 | 1180.98 | 0.9994 | -0.06% |
| `matmul.static` | 6708.69 | 6695.06 | 0.9980 | -0.20% |
| `transpose.static` | 390.83 | 389.72 | 0.9971 | -0.29% |
| `view.static` | 295.69 | 294.99 | 0.9976 | -0.24% |
| `permute.static` | 455.98 | 457.28 | 1.0028 | +0.28% |

All ratios are within 0.5% of 1.0. The small positive and negative deltas are
measurement noise and/or optimizer differences; the wrapper itself adds no
measurable release forwarding cost.

### Debug mode snapshot (2026-06-28, 10000 iterations)

| Operation | Raw ns/op | Typetorch ns/op | Ratio |
| --- | ---: | ---: | ---: |
| `add.dynamic` | 1329.82 | 1384.02 | 1.0408 |
| `add.static` | 1330.73 | 1377.79 | 1.0354 |
| `matmul.static` | 6831.19 | 6890.71 | 1.0087 |
| `transpose.static` | 490.63 | 582.94 | 1.1881 |
| `view.static` | 437.21 | 494.34 | 1.1307 |
| `permute.static` | 586.99 | 658.39 | 1.1216 |

Debug mode adds 1–19% overhead because inlining is disabled (`-fno-inline`).
This is expected and irrelevant for production use; always benchmark with
release builds.

## Binary Size Results (2026-06-28, with attribute/extensions, release)

Use `size` on the release artifacts:

```bash
size \
  build/linux/x86_64/release/typetorch_forwarding_benchmark \
  build/linux/x86_64/release/binary_size_libtorch_probe \
  build/linux/x86_64/release/binary_size_tensor_checked_probe \
  build/linux/x86_64/release/binary_size_tensor_unsafe_probe \
  build/linux/x86_64/release/typetorch_tensor_arithmetic_test \
  build/linux/x86_64/release/typetorch_cpp_debug
```

### Current snapshot

| Target | text | data | bss | dec | hex |
| --- | ---: | ---: | ---: | ---: | ---: |
| `typetorch_forwarding_benchmark` | 49096 | 1824 | 200 | 51120 | c7b0 |
| `binary_size_libtorch_probe` | 21507 | 1712 | 200 | 23419 | 5b7b |
| `binary_size_tensor_checked_probe` | 29022 | 1744 | 200 | 30966 | 78f6 |
| `binary_size_tensor_unsafe_probe` | 21443 | 1712 | 200 | 23355 | 5b3b |
| `typetorch_tensor_arithmetic_test` | 81033 | 2336 | 200 | 83569 | 14671 |
| `typetorch_cpp_debug` | 120335 | 3520 | 200 | 124055 | 1e497 |

### Key comparisons

| Comparison | text diff | Meaning |
| --- | ---: | --- |
| checked vs native | +7515 bytes (+34.9%) | Cost of Typetorch wrapper + runtime boundary validation |
| unsafe vs native | -64 bytes (-0.3%) | Typetorch forwarding alone — **zero-cost abstraction** |
| checked vs unsafe | +7579 bytes (+35.3%) | Isolated cost of runtime contract checks |

The unsafe path is now slightly *smaller* than the native LibTorch baseline
because gc-sections discards dead code (including unused LibTorch symbols) more
effectively from the typed binary.

## Historical Binary Size Snapshot (pre-folder-restructure, 2026-06)

| Target | text | data | bss | dec | hex |
| --- | ---: | ---: | ---: | ---: | ---: |
| `binary_size_libtorch_probe` | 61973 | 1712 | 200 | 63885 | f98d |
| `binary_size_tensor_checked_probe` | 65873 | 1744 | 200 | 67817 | 108e9 |
| `binary_size_tensor_unsafe_probe` | 62205 | 1712 | 200 | 64117 | fa75 |

The older `binary_size_tensor_probe` mixed checked and unsafe retain paths, so
it should not be used as the headline Typetorch-vs-LibTorch comparison.

## Symbol Growth Probe (2026-06-28)

Release binaries are stripped by the current build, so use the debug probes for
symbol accounting:

```bash
bash scripts/nm_size_probe.sh debug
```

The script reports three executables: native LibTorch (`binary_size_libtorch_probe`),
fully checked Typetorch (`binary_size_tensor_checked_probe`), and fully unsafe
Typetorch (`binary_size_tensor_unsafe_probe`). All probes print `16` and execute
the same high-level retain/add/unwrap/numel flow. The checked and unsafe probes
differ only at the raw-to-typed boundary.

### Debug Binary Sizes

| Binary | text | data | bss | dec | hex |
| --- | ---: | ---: | ---: | ---: | ---: |
| native (raw LibTorch) | 89834 | 1520 | 200 | 91554 | 165a2 |
| unsafe (Typetorch, no checks) | 120552 | 1592 | 200 | 122344 | 1dde8 |
| checked (Typetorch, full checks) | 137544 | 1640 | 200 | 139384 | 22078 |

### Symbol-Level Accounting

| Metric | bytes | symbols | Meaning |
| --- | ---: | ---: | --- |
| `native_probe_functions` | 486 | 9 | Raw `torch::Tensor` helper functions in the native probe |
| `checked_probe_functions` | 628 | 8 | Typetorch-owned helpers in the fully checked probe |
| `checked_tensor_wrapper` | 594 | 11 | Tensor wrapper code (total nm records) in checked probe |
| `checked_tensor_wrapper_unique` | 410 | 6 | Tensor wrapper code (unique by address) in checked probe |
| `checked_runtime_checks` | 938 | 5 | Runtime contract check symbols pulled in by `retain()` |
| `unsafe_probe_functions` | 727 | 8 | Typetorch-owned helpers in the unsafe probe |
| `unsafe_tensor_wrapper` | 368 | 10 | Tensor wrapper code (total nm records) in unsafe probe |
| `unsafe_tensor_wrapper_unique` | 184 | 5 | Tensor wrapper code (unique by address) in unsafe probe |
| `unsafe_runtime_checks` | **0** | **0** | Zero runtime checks in the unsafe path |

### Key Takeaways

| Comparison | text delta | Meaning |
| --- | ---: | --- |
| unsafe vs native | +30718 bytes (+34.2%) | Debug/no-inline pulls in more wrapper-visible code than release; release remains zero-cost |
| checked vs native | +47710 bytes (+53.1%) | Typetorch wrapper + runtime boundary validation |
| checked vs unsafe | +16992 bytes (+14.1%) | Isolated cost of runtime checks (debug mode) |
| checked_tensor_wrapper_unique vs unsafe | +226 bytes | Extra wrapper code variants from the checked path |
| unsafe_runtime_checks | **0 bytes, 0 symbols** | Unsafe path pulls zero runtime check code |

The unsafe path in debug is larger than native because `-O0 -fno-inline` keeps
address-visible wrapper helpers that release mode inlines and garbage-collects.
The release comparison is the production signal: unsafe remains 64 bytes smaller
than native in this snapshot.

The checked path adds ~17 KB of text over unsafe in debug. The directly matched
runtime-check symbols account for 938 bytes; the rest comes from retained helper
and error-reporting paths plus no-inline codegen differences. This is the
expected cost of validating shape, dtype, device, and layout at the raw-to-typed
boundary.

`checked_tensor_wrapper_unique` (410 bytes, 6 unique address-visible symbols)
vs `unsafe_tensor_wrapper_unique` (184 bytes, 5 unique address-visible symbols)
shows that `retain()` pulls in additional wrapper code variants (e.g. the
`Layout::Any` variants used by the checked path) beyond what the pure unsafe
path needs. The unsafe path still pulls in **0 runtime-check bytes and 0
runtime-check symbols**.


## Reproducibility Notes

- Results vary with compiler snapshot, LibTorch build, CPU, optimization flags,
  and whether symbols are stripped.
- Use release mode for runtime overhead.
- Use debug/no-inline mode for readable symbol accounting.
- Run benchmarks on an interactive or otherwise dedicated machine when possible;
  shared login nodes can add noise and may impose resource limits.
- Attribute hints (`hot`/`cold`/`unlikely`) guide the optimizer but do not
  change semantics; forward-compatible with compiler upgrades.
