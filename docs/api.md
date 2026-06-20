# API Notes

Tenspec wraps `at::Tensor` in a move-only type carrying a static tensor contract:

```cpp
tenspec::Tensor<Shape, DType, Device, Layout>
```

## Contract Parameters

| Parameter | Meaning |
| --- | --- |
| `Shape<...>` | Static dimensions. Use `tenspec::dyn` for dimensions known only at runtime. |
| `DType` | `Any`, `F32`, `F16`, `BF16`, `I64`, or `Bool`. |
| `Device` | `Any`, `CPU`, or `CUDA`. |
| `Layout` | `Any`, `Contiguous`, or `NonContiguous`. |

## Entering and Leaving Typed Code

Use `retain()` at trust boundaries:

```cpp
auto typed = Matrix::retain(raw);
```

It checks shape, dtype, device, and layout. Use `unsafe_retain()` or
`unsafe_move()` only when the surrounding code has already established the
contract.

Use `unwrap()` to move the raw tensor handle back out:

```cpp
at::Tensor raw = std::move(typed).unwrap();
```

`Tensor<...>` is move-only. This is intentionally different from `at::Tensor`,
which is a copyable handle.

## Operation Style

Most tensor operations are member functions that forward to the corresponding
LibTorch operation after compile-time contract checks:

```cpp
auto c = a.add(b);
auto y = x.matmul(w);
auto t = std::move(x).transpose<0, 1>();
auto v = std::move(x).view<batch, hidden>();
```

Operations with dimensions that should be known statically use template
parameters, not runtime integers:

```cpp
auto y = x.flatten<1, -1>();
auto z = x.unsqueeze<0>();
auto q = x.squeeze<2>();
```

`where` is a static three-argument operation because no operand is naturally the
receiver:

```cpp
auto out = Matrix::where(condition, x, y);
```

## Shape and Layout Propagation

Tenspec computes result types at compile time where possible:

- broadcast operations check static compatibility and produce a broadcast result
  shape;
- `matmul` checks inner dimensions and batch broadcast compatibility;
- `transpose`, `permute`, `view`, `flatten`, `squeeze`, and `unsqueeze` produce
  static output shapes;
- view-like operations preserve `Layout::Contiguous` when the current wrapper can
  prove it;
- operations that allocate or combine multiple tensors often return
  `Layout::Any` unless a stronger guarantee is obvious.

The project tries not to overuse `Layout::Any`: if an operation is just a view
of a contiguous tensor, the result type should keep that information.

## Differences From LibTorch

| Area | LibTorch behavior | Tenspec behavior |
| --- | --- | --- |
| Invalid static shapes | Error appears when the operation runs | Static mismatch can fail at compile time. |
| Runtime dimensions | All dimensions are runtime metadata | Known dimensions live in the type; `dyn` is explicit. |
| Runtime validation | Usually implicit inside kernels | `retain()` validates at typed boundaries. |
| Dimension arguments | Mostly runtime `int64_t` | Many shape-changing ops use template dimensions. |
| Tensor copying | `at::Tensor` is a copyable reference-counted handle | `Tensor<...>` is move-only to keep boundary transitions explicit. |
| Raw access | Raw tensor is the primary API | `unsafe_raw()` is available, but typed functions should preserve contracts. |

## Current Operation Coverage

The core wrapper currently covers selected arithmetic, matrix, shape, factory,
and conversion operations. It is intentionally not a full LibTorch facade.
Unsupported operations should be added only when their shape/dtype/device/layout
contract is clear enough to document and test.
