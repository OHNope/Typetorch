# Performance and Size Notes

This project keeps LibTorch as the runtime backend. The wrapper should add type
contracts and compile-time validation while keeping forwarding overhead small.

The current checked-in measurements are also summarized in the README. This file
explains how to reproduce them and how to interpret the numbers.

## Build Measurement Targets

```bash
source scripts/env.sh
xmake f -m release --yes --python="${PYTHON:-python3}"
xmake build tenspec_forwarding_benchmark
xmake build binary_size_tensor_probe
```

For symbol-level inspection, also build the debug/no-inline probe:

```bash
xmake f -m debug --yes --python="${PYTHON:-python3}"
xmake build binary_size_tensor_probe
```

## Forwarding Benchmark

Run:

```bash
build/linux/x86_64/release/tenspec_forwarding_benchmark 1000
```

The executable prints lines like:

```text
operation, raw_ns=<raw>, tenspec_ns=<typed>, ratio=<typed/raw>
```

Ratio is `Tenspec / raw at::Tensor`. A ratio close to `1.0` means the wrapper has
no measurable forwarding cost for that operation under the current build.

Current snapshot, release build, CPU tensors, one thread, 1000 iterations:

| Operation | Raw ns/op | Tenspec ns/op | Ratio | Overhead |
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
  build/linux/x86_64/release/tenspec_forwarding_benchmark \
  build/linux/x86_64/release/binary_size_tensor_probe \
  build/linux/x86_64/release/tenspec_tensor_arithmetic_test
```

Current snapshot:

| Target | Mode | File size | `.text` | `.data` | `.bss` | `size` total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `tenspec_forwarding_benchmark` | release | 51 KiB | 42,927 | 1,480 | 200 | 44,607 |
| `binary_size_tensor_probe` | release | 63 KiB | 49,707 | 1,400 | 200 | 51,307 |
| `tenspec_tensor_arithmetic_test` | release | 79 KiB | 68,345 | 1,648 | 200 | 70,193 |
| `binary_size_tensor_probe` | debug | 6.7 MiB | 222,494 | 1,320 | 200 | 224,014 |

## Symbol Growth Probe

Release binaries may be stripped. Use the debug probe for symbol accounting:

```bash
probe=build/linux/x86_64/debug/binary_size_tensor_probe
nm -S --size-sort --demangle "$probe" | sed -n '/tenspec/p'
```

The current summary was produced by grouping symbols containing `raw_`,
`typed_`, Tensor wrapper names, and runtime check names.

| Symbol group | Bytes | Symbols | Interpretation |
| --- | ---: | ---: | --- |
| Raw probe functions | 221 | 4 | Direct `at::Tensor` helper functions. |
| Typed probe functions | 322 | 6 | Equivalent Tenspec helper functions. |
| Typed/raw project function ratio | 1.457x | - | Small absolute growth in this probe. |
| Tensor-related symbols | 1018 | 14 | Wrapper methods and instantiated tensor helpers visible in the probe. |
| Runtime shape check symbols | 373 | 2 | Runtime contract checks. |
| dtype/device/layout match symbols | 353 | 3 | Runtime metadata predicates. |

The current repository does not include separate raw-only and typed-only binary
targets. For that reason, whole-binary relative growth is not reported as a
single percentage. The symbol table is the reproducible proxy available today.

## Reproducibility Notes

- Results vary with compiler snapshot, LibTorch build, CPU, optimization flags,
  and whether symbols are stripped.
- Use release mode for runtime overhead.
- Use debug/no-inline mode for readable symbol accounting.
- Run benchmarks on an interactive or otherwise dedicated machine when possible;
  shared login nodes can add noise and may impose resource limits.
