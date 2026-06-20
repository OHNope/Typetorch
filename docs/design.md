# Design Notes

typetorch is built around one rule: LibTorch stays responsible for tensor
storage, kernels, dispatch, autograd, and device behavior. typetorch adds a typed
contract layer around `::at::Tensor` and tries to make the contract cheap enough
that users can keep it in ordinary C++ code.

## Typed Tensor Contract

The central type is:

```cpp
typetorch::Tensor<Shape, DType, Device, Layout>
```

The type parameters describe the contract:

- `Shape<...>` stores static dimensions and `dyn` dynamic dimensions.
- `DType` records the expected scalar type.
- `Device` records the expected device class.
- `Layout` records whether the tensor is known contiguous, non-contiguous,
  sparse, or layout-agnostic.

The wrapper owns an `::at::Tensor` handle. It is move-only so moving back to raw
LibTorch is explicit:

```cpp
auto typed = Matrix::retain(raw);
::at::Tensor raw_again = ::std::move(typed).unwrap();
```

Use `retain()` at trust boundaries. It validates shape, dtype, device, and
layout at runtime. Use `unsafe_retain()` or `unsafe_move()` only when a previous
operation or invariant already proves the contract.

## Static and Dynamic Checks

Static dimensions are checked in the type system. Dynamic dimensions use `dyn`
and are validated when a raw tensor enters the typed world.

Operations such as `add`, `matmul`, `transpose`, `permute`, `view`, and
`contiguous` compute result types at compile time where possible. Invalid static
contracts fail during compilation through `consteval` diagnostics. The remaining
runtime checks are kept near raw tensor ingress.

This is why the codebase still contains `throw` statements in shape-meta code:
they are compile-time reflection/`consteval` diagnostics, not runtime exception
paths.

## Python Binding Design

The current Python binding path is intentionally small:

```text
reflected C++ function
-> generated PyCFunction thunk
-> Python module registration
```

The project previously used a dynamic bridge shaped like:

```text
Type enum + Value struct + invoke(Value*, Value*)
```

That design was removed. It made sense for a general C ABI, but this project
only targets C++/Python interop. The current design lets each generated wrapper
parse Python arguments directly according to the reflected C++ signature.

This removes:

- the runtime type enum;
- the generic value array;
- duplicated argument and result switches;
- extra temporary string/tensor storage in the bridge.

Parse errors use Python C API error reporting and return `nullptr`. The wrapper
does not use C++ exceptions for Python argument conversion.

## Module Boundaries

The codebase uses C++26 modules to keep heavy dependencies controlled:

- Core typetorch modules live in `src/typetorch_*.mpp`.
- `src/libtorch.mpp` wraps LibTorch imports.
- `bindings/python.mpp` wraps `Python.h` and torch Python C API headers.
- `src/typetorch_capi_reflect.mpp` imports both the typed tensor modules and the
  Python module to generate wrappers.

This organization avoids mixing Python C headers and module imports in ordinary
binding implementation files.

## Visibility and ABI Surface

The extension is built with hidden symbol visibility. The only symbol that must
be visible to Python's dynamic loader is:

```cpp
PyInit_typetorch_capi
```

Generated wrappers, reflection helpers, and local tensor wrapper symbols should
remain internal to the shared object.

## Non-Goals

typetorch is not trying to:

- replace PyTorch dispatch or kernels;
- provide a stable public C ABI;
- provide a full Python binding framework;
- infer all tensor invariants statically;
- hide LibTorch from C++ users.

The intended user already writes LibTorch C++ and wants stronger contracts at
module boundaries plus a low-overhead Python extension path.
