# Implementation Report

This note summarizes the current modules-based implementation state.

## Current Shape

- Core tensor contracts live in `src/typetorch_types.mpp`,
  `src/typetorch_shape_meta.mpp`, and `src/typetorch_tensor.mpp`.
- Runtime compatibility checks are separated into
  `src/typetorch_runtime_checks.mpp`.
- `src/libtorch.mpp` exports the small LibTorch surface used by the wrapper.
- Python C API integration is split across `bindings/` and
  `src/typetorch_capi_reflect.mpp`.
- Public API notes live in `docs/api.md`.
- Measurement methodology lives in `docs/performance-and-size.md`.

## Build Model

The repository uses xmake with C++26 modules enabled. LibTorch is consumed via a
local xmake package recipe for the prebuilt CUDA archive. Python extension builds
need Python development headers and a Python environment compatible with the
installed `torch` package.

## Cleanup Rules

Machine-specific paths, private benchmark outputs, and local setup notes should
stay out of committed documentation. Public docs should focus on architecture,
contracts, operation semantics, and reproducible commands.
