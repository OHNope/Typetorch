# Performance and Size Notes

This project keeps LibTorch as the runtime backend. The wrapper should add type
contracts and compile-time validation while keeping forwarding overhead small.

The current checked-in measurements are also summarized in the README. This file
explains how to reproduce them and how to interpret the numbers.

## Build Measurement Targets

```bash
source scripts/env.sh
xmake f -m release --yes --python="${PYTHON:-python3}"
xmake build typetorch_forwarding_benchmark
xmake build binary_size_libtorch_probe
xmake build binary_size_tensor_probe
```

For symbol-level inspection, also build the debug/no-inline probe:

```bash
xmake f -m debug --yes --python="${PYTHON:-python3}"
xmake build binary_size_libtorch_probe
xmake build binary_size_tensor_probe
```

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
  build/linux/x86_64/release/binary_size_tensor_probe \
  build/linux/x86_64/release/typetorch_tensor_arithmetic_test
```

Current snapshot:

| Target | Mode | File size | `.text` | `.data` | `.bss` | `size` total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `typetorch_forwarding_benchmark` | release | 51 KiB | 43,157 | 1,480 | 200 | 44,837 |
| `binary_size_libtorch_probe` | release | 59 KiB | 48,394 | 1,376 | 200 | 49,970 |
| `binary_size_tensor_probe` | release | 63 KiB | 49,659 | 1,400 | 200 | 51,259 |
| `typetorch_tensor_arithmetic_test` | release | 83 KiB | 72,948 | 1,672 | 200 | 74,820 |
| `binary_size_libtorch_probe` | debug | 3.9 MiB | 156,908 | 1,272 | 200 | 158,380 |
| `binary_size_tensor_probe` | debug | 6.2 MiB | 210,713 | 1,320 | 200 | 212,233 |

The release `binary_size_tensor_probe` is the same workload as
`binary_size_libtorch_probe`, with the typed wrapper used for retain/unwrap/add
paths. Compared to the native LibTorch probe, the typed probe is:

| Mode | File size delta | `.text` delta | `size` total delta |
| --- | ---: | ---: | ---: |
| release | +4,120 bytes (+6.9%) | +1,265 bytes (+2.6%) | +1,289 bytes (+2.6%) |
| debug | +2,386,632 bytes (+59.1%) | +53,805 bytes (+34.3%) | +53,853 bytes (+34.0%) |

## Symbol Growth Probe

Release binaries are stripped by the current build, so use the debug probes for
symbol accounting:

```bash
bash scripts/nm_size_probe.sh debug
```

`binary_size_libtorch_probe` is the native LibTorch equivalent of
`binary_size_tensor_probe`. Both programs print `16` and execute the same
high-level retain/add/unwrap/numel flow.

| Symbol group | Bytes | Symbols | Interpretation |
| --- | ---: | ---: | --- |
| Native probe functions | 486 | 9 | Native `at::Tensor` equivalent helper functions. |
| Typed probe functions | 768 | 9 | Typetorch helper functions with the same call flow. |
| Typed/native probe function ratio | 1.580x | - | +282 bytes in the noinline helper layer. |
| Tensor wrapper nm records | 432 | 7 | `Tensor` wrapper records, including ABI alias records. |
| Tensor wrapper unique code addresses | 321 | 4 | Same wrapper set folded by address; this is the better code-entity count. |
| Runtime check symbols | 700 | 5 | Runtime shape/dtype/device/layout contract checks. |

The wrapper is therefore not completely optimized away in the debug/noinline
probe. In the release whole-binary comparison, visible symbols are stripped, but
the typed probe still has about 1.3 KiB more `.text` than the native LibTorch
equivalent.

## Reproducibility Notes

- Results vary with compiler snapshot, LibTorch build, CPU, optimization flags,
  and whether symbols are stripped.
- Use release mode for runtime overhead.
- Use debug/no-inline mode for readable symbol accounting.
- Run benchmarks on an interactive or otherwise dedicated machine when possible;
  shared login nodes can add noise and may impose resource limits.
