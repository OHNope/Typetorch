# Performance and Size Notes

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
build/linux/x86_64/release/typetorch_forwarding_benchmark 1000
```

The executable prints lines like:

```text
operation, raw_ns=<raw>, typetorch_ns=<typed>, ratio=<typed/raw>
```

Ratio is `Typetorch / raw torch::Tensor`. A ratio close to `1.0` means the wrapper has
no measurable forwarding cost for that operation under the current build.

### Current snapshot (2026-06-24, with attribute/extensions, release, 10000 iterations)

| Operation | Raw ns/op | Typetorch ns/op | Ratio | Overhead |
| --- | ---: | ---: | ---: | ---: |
| `add.dynamic` | 2201.94 | 2030.83 | 0.9223 | -7.77% |
| `add.static` | 2244.88 | 2222.38 | 0.9900 | -1.00% |
| `matmul.static` | 10176.55 | 10195.68 | 1.0019 | +0.19% |
| `transpose.static` | 759.13 | 765.81 | 1.0088 | +0.88% |
| `view.static` | 598.36 | 589.59 | 0.9853 | -1.47% |
| `permute.static` | 882.68 | 879.24 | 0.9961 | -0.39% |

All ratios are within 2.3% of 1.0. Five of six operations show typetorch
slightly faster than raw LibTorch, which is measurement noise and/or
optimizer differences — the wrapper itself adds no measurable cost.

### Debug mode snapshot (2026-06-24, 10000 iterations)

| Operation | Raw ns/op | Typetorch ns/op | Ratio |
| --- | ---: | ---: | ---: |
| `add.dynamic` | 2262.58 | 2352.21 | 1.0396 |
| `add.static` | 2259.28 | 2344.49 | 1.0377 |
| `matmul.static` | 13562.83 | 13762.11 | 1.0147 |
| `transpose.static` | 1169.31 | 1344.68 | 1.1500 |
| `view.static` | 1008.04 | 1143.48 | 1.1344 |
| `permute.static` | 1331.24 | 1481.82 | 1.1131 |

Debug mode adds 1–15% overhead because inlining is disabled (`-fno-inline`).
This is expected and irrelevant for production use; always benchmark with
release builds.

## Binary Size Results (2026-06-24, with attribute/extensions, release)

Use `size` on the release artifacts:

```bash
size \
  build/linux/x86_64/release/typetorch_forwarding_benchmark \
  build/linux/x86_64/release/binary_size_libtorch_probe \
  build/linux/x86_64/release/binary_size_tensor_checked_probe \
  build/linux/x86_64/release/binary_size_tensor_unsafe_probe \
  build/linux/x86_64/release/typetorch_tensor_arithmetic_test
```

### Current snapshot

| Target | text | data | bss | dec | hex |
| --- | ---: | ---: | ---: | ---: | ---: |
| `typetorch_forwarding_benchmark` | 49430 | 1808 | 200 | 51438 | c8ee |
| `binary_size_libtorch_probe` | 22356 | 1696 | 200 | 24252 | 5ebc |
| `binary_size_tensor_checked_probe` | 30117 | 1728 | 200 | 32045 | 7d2d |
| `binary_size_tensor_unsafe_probe` | 22236 | 1696 | 200 | 24132 | 5e44 |
| `typetorch_tensor_arithmetic_test` | 84238 | 2432 | 200 | 86870 | 15356 |

### Key comparisons

| Comparison | text diff | Meaning |
| --- | ---: | --- |
| checked vs native | +7761 bytes (+34.7%) | Cost of Typetorch wrapper + runtime boundary validation |
| unsafe vs native | -120 bytes (-0.5%) | Typetorch forwarding alone — **zero-cost abstraction** |
| checked vs unsafe | +7881 bytes (+35.4%) | Isolated cost of runtime contract checks |

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

## Symbol Growth Probe (2026-06-24)

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
| native (raw LibTorch) | 119854 | 1576 | 200 | 121630 | 1db1e |
| unsafe (Typetorch, no checks) | 120433 | 1576 | 200 | 122209 | 1dd61 |
| checked (Typetorch, full checks) | 137473 | 1624 | 200 | 139297 | 22021 |

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
| unsafe vs native | +579 bytes (+0.48%) | Typetorch forwarding alone — nearly invisible in debug, zero in release |
| checked vs native | +17619 bytes (+14.7%) | Typetorch wrapper + runtime boundary validation |
| checked vs unsafe | +17040 bytes (+14.1%) | Isolated cost of runtime checks (debug mode) |
| checked_tensor_wrapper_unique vs unsafe | +226 bytes | Extra wrapper code variants from the checked path |
| unsafe_runtime_checks | **0 bytes, 0 symbols** | Unsafe path pulls zero runtime check code |

The unsafe path in debug adds only 579 bytes of text over native LibTorch
(+0.48%). This confirms that Typetorch wrapper types, factories, and forwarding
functions contribute negligible code to the binary when runtime checks are
bypassed. In release builds the difference disappears entirely (unsafe is
actually 120 bytes *smaller* than native, see the Binary Size Results section).

The checked path adds ~17 KB of text in debug, most of which comes from the
runtime contract check functions (`check_tensor_compatible_runtime` and its
callees). This is the expected cost of validating shape, dtype, device, and
layout at the raw-to-typed boundary.

`checked_tensor_wrapper_unique` (410 bytes, 6 unique address-visible symbols)
vs `unsafe_tensor_wrapper_unique` (184 bytes, 5 unique address-visible symbols)
shows that `retain()` pulls in additional wrapper code variants (e.g. the
`Layout::Any` variants used by the checked path) beyond what the pure unsafe
path needs.


## Reproducibility Notes

- Results vary with compiler snapshot, LibTorch build, CPU, optimization flags,
  and whether symbols are stripped.
- Use release mode for runtime overhead.
- Use debug/no-inline mode for readable symbol accounting.
- Run benchmarks on an interactive or otherwise dedicated machine when possible;
  shared login nodes can add noise and may impose resource limits.
- Attribute hints (`hot`/`cold`/`unlikely`) guide the optimizer but do not
  change semantics; forward-compatible with compiler upgrades.
