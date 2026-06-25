# factory_options: Static Member Variable vs Static Member Function

Comparison of two approaches for providing `TensorOptions` to factory methods
(`empty`, `zeros`, `ones`, `rand`, `randn`, `arange`) in the typed `Tensor`
class template.

## Approaches compared

```cpp
// A: static inline member variable (current)
template <ShapeLike ShapeKind, DType DTypeKind, Device DeviceKind, Layout LayoutKind>
class Tensor {
    // ...
    static inline ::torch::TensorOptions const factory_options{
        ::torch::TensorOptions{}
            .dtype(detail::to_torch_dtype(dtype))
            .device(detail::to_torch_device(device))};
};

// B: static inline member function (proposed)
template <ShapeLike ShapeKind, DType DTypeKind, Device DeviceKind, Layout LayoutKind>
class Tensor {
    // ...
    static inline auto parameter_options() -> ::torch::TensorOptions {
        return ::torch::TensorOptions{}
            .dtype(detail::to_torch_dtype(dtype))
            .device(detail::to_torch_device(device));
    }
};
```

## Benchmark

14 distinct `Tensor` instantiations varying rank, dtype, device, and layout.
Debug build (`-O0 -g3 -fno-inline`). Measured by `run_factory_compare.sh`.

| Metric | Static Var (A) | Static Func (B) | Delta (B - A) |
|---|---:|---:|---:|
| Build time (s) | 138 | 134 | -4 |
| Binary size (bytes) | 21,300,392 | 21,298,592 | -1,800 |
| Symbol count (`nm -C`) | 550 | 534 | -16 |
| .text (bytes) | 72,135 | 72,111 | -24 |
| .data (bytes) | 1,160 | 1,152 | -8 |
| .bss (bytes) | 424 | 200 | -224 |

## Symbol-level analysis

### Static var (A)

Per `Tensor` instantiation, two symbols are emitted:

- `_ZN...factory_optionsE` — data object (`u` / local GOT-relative)
- `_ZGV...factory_optionsE` — guard variable (`u`) for thread-safe
  dynamic initialization

Both occupy storage in the data sections. With 14 specializations, 28
extra symbols appear.

### Static func (B)

Per `Tensor` instantiation, one symbol is emitted:

- `_ZN...parameter_optionsEv` — function (`t` / local text)

No guard variable. No data storage. The function returns a fresh
`TensorOptions` value each call.

## Reasoning

`TensorOptions` is a lightweight value type (two fields: `ScalarType` +
`Device`). Storing a `static` copy per specialization is unnecessary
overhead. A static member function that constructs and returns the value
on each call carries no data-section cost and the call is trivially
inlined at any optimization level above `-O0`.

The 28 extra symbols from guard + data variables per specialization can
become meaningful at scale (hundreds of distinct `Tensor` specializations
across shape, dtype, device, and layout combinations).

In `-O2` or `-O3` builds the advantage of approach B increases: the
function body is unconditionally inlined at every call site, and the
function symbol itself is eliminated. The static variable (approach A)
must still reserve storage because it has an address.

## Verdict

The difference is negligible with the current number of specializations
(~14 in tests, ~few dozen in real use). Switching is not urgent but is
semantically cleaner and scales better. Worth adopting when touching
this area for other reasons.

## Test artifacts

Archived under `tests/factory_options_bench/`.
