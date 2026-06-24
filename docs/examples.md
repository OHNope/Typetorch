# Compile-Time Contract Examples

This file collects the full set of examples showing what Typetorch accepts and
rejects at compile time. The README includes a shorter selection; this is the
exhaustive reference.

## What Compiles

### Broadcasting — shapes checked at compile time

```cpp
using Matrix23 = Tensor<Shape<2, 3>, DType::F32, Device::CPU, Layout::Contiguous>;
using Vector3  = Tensor<Shape<3>,    DType::F32, Device::CPU, Layout::Contiguous>;
using Vector2  = Tensor<Shape<2>,    DType::F32, Device::CPU, Layout::Contiguous>;

auto x = Matrix23::ones();
auto b = Vector3::ones();

auto y = x.add(b);   // OK: Shape<3> broadcasts with Shape<2,3> -> result is Shape<2,3>
```

```cpp
// OK: scalar add just changes values, shape preserved
auto y = x.add(torch::Scalar{2.0F});   // result: same Matrix23 type
```

### matmul — inner dimensions verified, batch broadcast computed

```cpp
using In  = Tensor<Shape<dyn, 3>, DType::F32, Device::CPU, Layout::Any>;
using Wt  = Tensor<Shape<3, 2>,   DType::F32, Device::CPU, Layout::Any>;
using Out = Tensor<Shape<dyn, 2>, DType::F32, Device::CPU, Layout::Any>;

auto x = In::retain(raw_input);
auto w = Wt::retain(raw_weight);
auto y = x.matmul(w);   // OK: inner dims (3,3) match -> result Shape<dyn, 2>
```

### transpose/permute — shape and layout computed statically

```cpp
using Cube   = Tensor<Shape<2, 3, 4>, DType::F32, Device::CPU, Layout::Contiguous>;

auto x = Cube::ones();

auto t = std::move(x).transpose<0, 1>();
// OK: result Shape<3, 2, 4>, Layout::Any (transpose breaks contiguity)

auto p = std::move(x).permute<0, 2, 1>();
// OK: result Shape<2, 4, 3>, Layout::Any

auto id = std::move(x).permute<0, 1, 2>();
// OK: identity permute -> result Shape<2, 3, 4>, Layout::Contiguous (preserved!)

auto c = std::move(p).contiguous();
// OK: copy into contiguous layout -> result Shape<2, 4, 3>, Layout::Contiguous
```

### view/reshape — element count verified, -1 inference

```cpp
using Matrix = Tensor<Shape<2, 3>, DType::F32, Device::CPU, Layout::Contiguous>;
using Flat   = Tensor<Shape<6>,    DType::F32, Device::CPU, Layout::Contiguous>;

auto x = Matrix::ones();

auto v = std::move(x).view<6>();
// OK: 2 x 3 = 6 elements -> Shape<6>, Layout::Contiguous

auto v2 = std::move(x).view<typetorch::infer>();
// OK: -1 inference -> Shape<6>, inferred from total elements
```

### squeeze/unsqueeze/flatten — dimensions tracked

```cpp
using Squeezable = Tensor<Shape<1, 2, 1, 3>, DType::F32, Device::CPU, Layout::Contiguous>;
using Squeezed   = Tensor<Shape<2, 3>,       DType::F32, Device::CPU, Layout::Contiguous>;

auto x = Squeezable::ones();
auto s = x.squeeze();              // OK: removes all dim=1 -> Shape<2, 3>
auto s1 = x.squeeze<2>();          // OK: removes dim=1 at index 2 -> Shape<1, 2, 3>

auto u = Squeezed::ones().unsqueeze<0>();  // OK: adds dim=1 at index 0 -> Shape<1, 2, 3>

auto f = Squeezed::ones().flatten<>();     // OK: flattens all -> Shape<6>
auto f2 = Squeezed::ones().flatten<1, 2>();// OK: flattens dims 1..2 -> Shape<2, 3> (no-op here)
```

### where — condition dtype enforced

```cpp
using BoolMat = Tensor<Shape<2, 3>, DType::Bool, Device::CPU, Layout::Contiguous>;
auto cond = BoolMat::retain(mask);
auto x    = Matrix23::ones();
auto y    = Matrix23::zeros();

auto out = Matrix23::where(cond, x, y);  // OK: all shapes broadcast-compatible -> Matrix23
```

### cat/stack — rank and shape merging

```cpp
auto a = Matrix23::ones();
auto b = Matrix23::ones();

auto c0 = Matrix23::cat<0>(a, b);
// OK: cat along dim 0 -> Shape<4, 3>

auto c1 = typetorch::cat<1>(a, b);
// OK: free-function form -> Shape<2, 6>

auto s0 = Matrix23::stack<0>(a, b);
// OK: stack along new dim 0 -> Shape<2, 2, 3>
```

### NN ops — selected operations with static checks

```cpp
auto x = Matrix23::randn();

auto r = x.relu();                  // OK: no shape change
auto g = x.gelu();                  // OK: static_assert ensures floating dtype
auto s = x.softmax(1);              // OK: static_assert ensures floating dtype

auto ln = x.layer_norm<3>();        // OK: normalized_shape matches trailing dims -> Shape<2,3>
auto rn = x.rms_norm<3>();          // OK: same check for RMS norm

auto m = x.masked_fill(BoolMat::ones(), torch::Scalar{0.0F});  // OK: mask must be Bool dtype
```

### arange — fully static tensor creation

```cpp
using Matrix23 = Tensor<Shape<2, 3>, DType::F32, Device::CPU, Layout::Contiguous>;

auto v  = Matrix23::arange<6>();                // OK: 0..5 -> Shape<6>, DType::F32
auto v2 = Matrix23::arange<0, 6>();             // OK: same
auto v3 = Matrix23::arange<0, 6, 2, DType::I64>();
// OK: step=2 -> 3 elements -> Shape<3>, DType::I64
```

### Dynamic dimensions with `dyn`

```cpp
// dyn = runtime-only dimensions; checks that CAN be static still are
using DynMat = Tensor<Shape<dyn, dyn>, DType::F32, Device::CPU, Layout::Any>;

auto x = DynMat::retain(some_runtime_sized_tensor);  // OK
auto y = DynMat::ones();   // ERROR: ones() requires fully static shape
```

## What Does NOT Compile

### Shape mismatch — caught at compile time

```cpp
using Matrix23 = Tensor<Shape<2, 3>, DType::F32, Device::CPU, Layout::Contiguous>;
using Matrix32 = Tensor<Shape<3, 2>, DType::F32, Device::CPU, Layout::Contiguous>;

auto x = Matrix23::ones();
auto y = Matrix32::ones();

auto z = x.add(y);
// ERROR: static assertion failed: tensor shapes are not broadcast-compatible
//   BROADCAST_COMPATIBLE(Shape<2,3>, Shape<3,2>) -> false
//   dim 0: 2 vs 3, neither is 1 -> fail at consteval time
```

### matmul inner dimension mismatch

```cpp
using A = Tensor<Shape<2, 3>, DType::F32, Device::CPU, Layout::Contiguous>;
using B = Tensor<Shape<2, 4>, DType::F32, Device::CPU, Layout::Contiguous>;

auto z = A::ones().matmul(B::ones());
// ERROR: consteval throw: matrix matmul inner dim mismatch
//   A's last dim (3) != B's second-to-last dim (2)
```

### view with wrong element count

```cpp
using Matrix23 = Tensor<Shape<2, 3>, DType::F32, Device::CPU, Layout::Contiguous>;

auto z = std::move(Matrix23::ones()).view<3, 3>();
// ERROR: consteval throw: the sizes of the elements of the two
//   tensors must be the same! (2x3=6 != 3x3=9)
```

### Factory on dynamic shape

```cpp
using DynVec = Tensor<Shape<dyn>, DType::F32, Device::CPU, Layout::Contiguous>;

auto v = DynVec::zeros();
// ERROR: static assertion failed: Tensor::zeros() requires a fully static shape
//   DynVec::shape::dynamic_rank is non-zero
```

### gelu / softmax on integer dtype

```cpp
using IntMat = Tensor<Shape<2, 3>, DType::I64, Device::CPU, Layout::Contiguous>;

auto g = IntMat::ones().gelu();
// ERROR: static assertion failed: Tensor::gelu() requires floating tensor dtype
//   detail::is_floating_dtype(DType::I64) -> false
```

### view/reshape on non-contiguous tensor

```cpp
using Cube = Tensor<Shape<2, 3, 4>, DType::F32, Device::CPU, Layout::Contiguous>;

auto x = Cube::ones();
auto t = std::move(x).transpose<0, 2>();   // layout degrades to Layout::Any
auto v = std::move(t).view<2, 4, 3>();
// ERROR: consteval throw: Tensor should be contiguous!
//   Must call .contiguous() first
```

Fix:

```cpp
auto v = std::move(t).contiguous().view<2, 4, 3>();  // OK
```

### masked_fill with non-bool mask

```cpp
auto x = Matrix23::ones();
auto m = Matrix23::ones();  // F32 mask — not allowed

auto z = x.masked_fill(m, torch::Scalar{0.0F});
// ERROR: static assertion failed: Tensor::masked_fill() requires a bool mask tensor
//   Mask::dtype is F32, not Bool
```

### DType conflict in arithmetic (conservative widening)

```cpp
using F32Mat = Tensor<Shape<2, 3>, DType::F32, Device::CPU, Layout::Contiguous>;
using I64Mat = Tensor<Shape<2, 3>, DType::I64, Device::CPU, Layout::Contiguous>;

auto z = F32Mat::ones().add(I64Mat::ones());
// Compiles, BUT result dtype is DType::Any — the type system conservatively
// widens to Any when dtypes differ. This propagates through subsequent ops.
```

### Device mismatch (conservative widening)

```cpp
using CPUMat  = Tensor<Shape<2, 3>, DType::F32, Device::CPU,  Layout::Contiguous>;
using CUDAMat = Tensor<Shape<2, 3>, DType::F32, Device::CUDA, Layout::Contiguous>;

auto z = CPUMat::ones().add(CUDAMat::ones());
// Compiles, BUT result device is Device::Any — same conservative widening.
```

### Runtime retain with wrong shape

```cpp
auto raw = torch::arange(9, opts).view({3, 3});   // 3x3 at runtime
auto x = Matrix23::retain(raw);
// RUNTIME ERROR: check_tensor_compatible_runtime fails
//   Expected shape <2, 3>, got <3, 3>
```
