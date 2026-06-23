# API Notes

Typetorch wraps `torch::Tensor` in a move-only type carrying a static tensor contract:

```cpp
typetorch::Tensor<Shape, DType, Device, Layout>
```

Raw tensor interop is expressed through the LibTorch C++ frontend namespace.
User code should pass and receive `torch::Tensor` handles. Internally those
handles still use the same LibTorch storage, dispatch, autograd, and device
behavior.

## Contract Parameters

| Parameter | Meaning |
| --- | --- |
| `Shape<...>` | Static dimensions. Use `typetorch::dyn` for dimensions known only at runtime. |
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
torch::Tensor raw = std::move(typed).unwrap();
```

`Tensor<...>` is move-only. This is intentionally different from `torch::Tensor`,
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

Factory functions and free/static operations follow the same convention:
Typetorch wrappers call the `torch::` C++ frontend surface for typed API calls.

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

Typetorch computes result types at compile time where possible:

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

| Area | LibTorch behavior | Typetorch behavior |
| --- | --- | --- |
| Invalid static shapes | Error appears when the operation runs | Static mismatch can fail at compile time. |
| Runtime dimensions | All dimensions are runtime metadata | Known dimensions live in the type; `dyn` is explicit. |
| Runtime validation | Usually implicit inside kernels | `retain()` validates at typed boundaries. |
| Dimension arguments | Mostly runtime `int64_t` | Many shape-changing ops use template dimensions. |
| Tensor copying | `torch::Tensor` is a copyable reference-counted handle | `Tensor<...>` is move-only to keep boundary transitions explicit. |
| Raw access | Raw tensor is the primary API | `unsafe_raw()` is available, but typed functions should preserve contracts. |



## Operation Quick Reference

### Factory (no input tensor)
| Method | Static Shape Required | Notes |
| --- | :---: | --- |
| `empty()` | Yes | Uninitialized |
| `zeros()` | Yes | All zeros |
| `ones()` | Yes | All ones |
| `rand()` | Yes | Uniform [0,1) |
| `randn()` | Yes | Normal(0,1) |
| `arange<N>()` | Partially | Shape computed from args |

### Arithmetic (tensor-tensor, tensor-scalar)
| Method | Shape Rule |
| --- | --- |
| `add(other, alpha=1)` | Broadcast |
| `sub(other, alpha=1)` | Broadcast |
| `mul(other)` | Broadcast |
| `div(other)` | Broadcast |
| `matmul(other)` | Inner dim match + batch broadcast |

### View / Shape (consume `this` via `&&`)
| Method | Notes |
| --- | --- |
| `transpose<D0, D1>()` | Layout degrades unless identity |
| `permute<Dims...>()` | Layout preserved only for identity |
| `view<Dims...>()` | Requires Contiguous; supports `infer` |
| `reshape<Dims...>()` | Auto-contiguous for NonContiguous |
| `flatten<S, E>()` | StartDim, EndDim (default: all) |
| `squeeze()` / `squeeze<Dim>()` | Removes dims of size 1 |
| `unsqueeze<Dim>()` | Inserts dim of size 1 |
| `contiguous()` | Copies to contiguous if needed |

### NN (selected)
| Method | Notes |
| --- | --- |
| `relu()` / `relu_()` | In-place variant available |
| `gelu(approx)` / `gelu_()` | Requires floating dtype |
| `softmax(dim)` | Requires floating dtype |
| `layer_norm<Dims...>(eps)` | Normalized shape must match trailing dims |
| `layer_norm(weight, bias, eps)` | Same, with affine params |
| `rms_norm<Dims...>()` | Same shape rules as layer_norm |
| `masked_fill(mask, value)` | Mask must be Bool dtype |

### Aggregate / Condition
| Method | Notes |
| --- | --- |
| `where(cond, x, y)` | Static 3-argument; cond must be Bool |
| `cat<Dim>(first, rest...)` | Same rank, cat dim summed |
| `stack<Dim>(first, rest...)` | Same shape, new dim inserted |

### Conversion
| Method | Notes |
| --- | --- |
| `to<Device, DType>()` | Device and/or dtype conversion |
| `clone()` | Deep copy |

## Current Operation Coverage

The core wrapper currently covers selected arithmetic, matrix, shape, factory,
and conversion operations. It is intentionally not a full LibTorch facade.
Unsupported operations should be added only when their shape/dtype/device/layout
contract is clear enough to document and test.
