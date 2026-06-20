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
```

For symbol-level inspection, use the debug/no-inline probes:

```bash
bash scripts/nm_size_probe.sh debug
```

Probe roles:

| Target | Role |
| --- | --- |
| `binary_size_libtorch_probe` | Native LibTorch baseline using only `at::Tensor`. |
| `binary_size_tensor_checked_probe` | Typetorch path where every raw tensor enters via `retain()` and pays runtime shape/dtype/device/layout checks. |
| `binary_size_tensor_unsafe_probe` | Typetorch path where every raw tensor enters via `unsafe_retain()`; this models a trusted caller that guarantees the contract. |
| `binary_size_tensor_probe` | Legacy mixed probe that uses both checked and unsafe retain paths; keep it only for old-result comparison. |

## Forwarding Benchmark

Run:

```bash
build/linux/x86_64/release/typetorch_forwarding_benchmark 1000
```

The executable prints lines like:

```text
operation, raw_ns=<raw>, typetorch_ns=<typed>, ratio=<typed/raw>
```

Ratio is `Typetorch / raw at::Tensor`. A ratio close to `1.0` means the wrapper has
no measurable forwarding cost for that operation under the current build.

Current snapshot, release build, CPU tensors, one thread, 1000 iterations:

| Operation | Raw ns/op | Typetorch ns/op | Ratio | Overhead |
| --- | ---: | ---: | ---: | ---: |
| `add.dynamic` | 1984.773 | 1977.399 | 0.9963 | -0.37% |
| `add.static` | 2008.366 | 1990.744 | 0.9912 | -0.88% |
| `matmul.static` | 12711.788 | 12760.018 | 1.0038 | +0.38% |
| `transpose.static` | 994.180 | 998.066 | 1.0039 | +0.39% |
| `view.static` | 748.612 | 745.286 | 0.9956 | -0.44% |
| `permute.static` | 1125.074 | 1112.860 | 0.9891 | -1.09% |

Small negative overheads mean measurement noise or optimizer differences, not
that the wrapper is inherently faster than LibTorch.

## Binary Size Snapshot

Use `size` on the release artifacts:

```bash
size \
  build/linux/x86_64/release/typetorch_forwarding_benchmark \
  build/linux/x86_64/release/binary_size_libtorch_probe \
  build/linux/x86_64/release/binary_size_tensor_checked_probe \
  build/linux/x86_64/release/binary_size_tensor_unsafe_probe \
  build/linux/x86_64/release/typetorch_tensor_arithmetic_test
```

The important comparison is no longer a single typed-vs-native number:

| Comparison | Meaning |
| --- | --- |
| checked vs native | Cost of Typetorch plus runtime boundary validation. |
| unsafe vs native | Cost of Typetorch forwarding/wrapper code when the user guarantees the typed contract. |
| checked vs unsafe | Isolated cost of runtime validation at the raw-to-typed boundary. |

Do not use the legacy `binary_size_tensor_probe` as the headline comparison: it
mixes one checked retain with one unsafe retain, so the program semantics are
not a clean match for either fully checked or fully trusted usage.

## Symbol Growth Probe

Release binaries are stripped by the current build, so use the debug probes for
symbol accounting:

```bash
bash scripts/nm_size_probe.sh debug
```

The script reports three executables: native LibTorch, fully checked Typetorch,
and fully unsafe Typetorch. All probes print `16` and execute the same
high-level retain/add/unwrap/numel flow. The checked and unsafe probes differ
only at the raw-to-typed boundary.

Key summary rows:

| Row | Meaning |
| --- | --- |
| `native_probe_functions` | Native `at::Tensor` helper functions. |
| `checked_probe_functions` | Typetorch helper functions in the fully checked probe. |
| `unsafe_probe_functions` | Typetorch helper functions in the fully unsafe probe. |
| `checked_runtime_checks` | Runtime contract check symbols pulled in by `retain()`. |
| `unsafe_runtime_checks` | Should be zero or much smaller; any non-zero value indicates checks were pulled in indirectly. |
| `*_tensor_wrapper_unique` | Wrapper code folded by address; this is the better code-entity count than raw nm records. |

## Reproducibility Notes

- Results vary with compiler snapshot, LibTorch build, CPU, optimization flags,
  and whether symbols are stripped.
- Use release mode for runtime overhead.
- Use debug/no-inline mode for readable symbol accounting.
- Run benchmarks on an interactive or otherwise dedicated machine when possible;
  shared login nodes can add noise and may impose resource limits.
