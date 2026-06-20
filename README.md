# Tenspec

Tenspec is an experimental C++26 project for type-safe LibTorch tensor wrappers
and lightweight Python C API bindings.

The core wrapper carries tensor shape, dtype, device, and layout information in
C++ types while forwarding runtime work to LibTorch. LibTorch still owns tensor
storage, kernels, dispatch, autograd, and device behavior; Tenspec adds a typed
contract layer around `at::Tensor`.

## Status

This repository is a research/prototype codebase. It intentionally depends on
very recent C++ compiler features, especially C++26 modules and static
reflection. Expect toolchain sensitivity.

## Requirements

Minimum practical requirements:

| Component | Requirement | Notes |
| --- | --- | --- |
| C++ compiler | GCC 16.1 or newer with modules and static reflection support | Older GCC releases are not expected to build this project. |
| Build system | xmake | Used for C++26 module builds and local package resolution. |
| LibTorch | 2.8.0 + CUDA 12.8 archive | The repository includes an xmake package recipe in `packages/l/libtorch-bin/`. |
| Python headers | CPython development headers | Required for `tenspec_capi_ext`. |
| Python runtime | Python with `torch` installed | Required only to import and exercise the extension. |
| OS/tooling | Linux, `nm`, `size`, common binutils | Measurement scripts assume Unix-like tooling. |

The repository does not hard-code a machine layout. If GCC or xmake is not on
`PATH`, set these optional variables before sourcing `scripts/env.sh`:

```bash
export GCC_ROOT=/path/to/gcc-16.1-or-newer
export GCC_RUNTIME_LIB=/path/to/gcc/runtime/lib64   # optional
export XMAKE_ROOT=/path/to/xmake-prefix             # optional
export PYTHON=/path/to/python                       # optional
```

## Build From Scratch

Clone and initialize submodules:

```bash
git clone <repo-url> tenspec
cd tenspec
git submodule update --init --recursive
```

Prepare the environment:

```bash
source scripts/env.sh
bash scripts/check_env.sh
```

Configure and run the tensor API test:

```bash
xmake f -m debug --yes --python="${PYTHON:-python3}"
xmake build tenspec_tensor_arithmetic_test
xmake run tenspec_tensor_arithmetic_test
```

Build and run the C++ debug example:

```bash
xmake build tenspec_cpp_debug
xmake run tenspec_cpp_debug
```

Build the Python extension:

```bash
xmake build tenspec_capi_ext
```

The extension is emitted under `build/<platform>/<arch>/<mode>/` with basename
`tenspec_capi`.

For release measurements:

```bash
xmake f -m release --yes --python="${PYTHON:-python3}"
xmake build tenspec_forwarding_benchmark
xmake build binary_size_tensor_probe
```

## API Sketch

```cpp
import tenspec;

using Matrix = tenspec::Tensor<
    tenspec::Shape<2, 3>,
    tenspec::DType::F32,
    tenspec::Device::CPU,
    tenspec::Layout::Contiguous>;

at::Tensor raw = at::arange(6, options).view({2, 3});
auto typed = Matrix::retain(raw);        // runtime contract check
auto flat = typed.flatten<>();           // result type is Shape<6>
at::Tensor back = std::move(flat).unwrap();
```

Important differences from ordinary LibTorch code:

| Topic | LibTorch | Tenspec |
| --- | --- | --- |
| Tensor type | Mostly dynamic `at::Tensor` metadata | Shape, dtype, device, and layout are part of the C++ type. |
| Shape errors | Usually runtime errors | Statically known shape mismatches fail during compilation. |
| Dynamic dimensions | Tensor sizes are runtime values | `dyn` marks dimensions that remain runtime-checked. |
| Ownership style | `at::Tensor` is copyable handle type | `Tensor<...>` is move-only; use `unwrap()` to recover raw LibTorch. |
| Trust boundary | Any `at::Tensor` can flow anywhere | `retain()` validates raw tensors before they enter typed code. |
| Layout | Runtime property | `Layout::Contiguous`, `Layout::NonContiguous`, or `Layout::Any` can be tracked. |

See [docs/api.md](docs/api.md) for the supported operation style and the places
where Tenspec intentionally differs from LibTorch.

## Current Measurement Snapshot

These numbers are a snapshot from the checked-out build artifacts used during
repository cleanup. Reproduce them with the commands in
[docs/performance-and-size.md](docs/performance-and-size.md). Treat them as
indicative, not as a stable benchmark promise.

### Forwarding Performance

Release build, CPU tensors, 1 thread, 1000 iterations. Ratio is
`Tenspec / raw at::Tensor`; values near 1.0 mean no measurable forwarding loss.

| Operation | Raw ns/op | Tenspec ns/op | Ratio | Overhead |
| --- | ---: | ---: | ---: | ---: |
| `add.dynamic` | 1984.773 | 1977.399 | 0.9963 | -0.37% |
| `add.static` | 2008.366 | 1990.744 | 0.9912 | -0.88% |
| `matmul.static` | 12711.788 | 12760.018 | 1.0038 | +0.38% |
| `transpose.static` | 994.180 | 998.066 | 1.0039 | +0.39% |
| `view.static` | 748.612 | 745.286 | 0.9956 | -0.44% |
| `permute.static` | 1125.074 | 1112.860 | 0.9891 | -1.09% |

### Symbol Growth Probe

Debug/no-inline `binary_size_tensor_probe`, summarized with `nm -S`. This is a
symbol-level proxy for wrapper code growth, not a whole-program raw-only binary
comparison.

| Symbol group | Bytes | Symbols | Interpretation |
| --- | ---: | ---: | --- |
| Raw probe functions | 221 | 4 | Direct `at::Tensor` helper functions. |
| Typed probe functions | 322 | 6 | Equivalent Tenspec helper functions. |
| Typed/raw project function ratio | 1.457x | - | Small absolute growth in this probe. |
| Tensor-related symbols | 1018 | 14 | Wrapper methods and instantiated tensor helpers visible in the probe. |
| Runtime shape check symbols | 373 | 2 | Runtime contract checks. |
| dtype/device/layout match symbols | 353 | 3 | Runtime metadata predicates. |

### Binary Size Snapshot

| Target | Mode | File size | `.text` | `.data` | `.bss` | `size` total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `tenspec_forwarding_benchmark` | release | 51 KiB | 42,927 | 1,480 | 200 | 44,607 |
| `binary_size_tensor_probe` | release | 63 KiB | 49,707 | 1,400 | 200 | 51,307 |
| `tenspec_tensor_arithmetic_test` | release | 79 KiB | 68,345 | 1,648 | 200 | 70,193 |
| `binary_size_tensor_probe` | debug | 6.7 MiB | 222,494 | 1,320 | 200 | 224,014 |

The current repository does not include separate raw-only and typed-only binary
targets, so whole-binary relative growth is not reported as a percentage. The
symbol table above is the reproducible comparison currently available.

## Documentation

- [API Notes](docs/api.md)
- [Design Notes](docs/design.md)
- [Performance and Size Notes](docs/performance-and-size.md)
- [Implementation Report](docs/implementation-report-2026-06-06.md)
