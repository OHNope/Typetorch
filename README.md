# Typetorch

Typetorch is an experimental C++26 library that makes tensor shape, dtype, device, and layout part of
the **C++ type system**. It wraps LibTorch's `torch::Tensor` in a move-only typed contract layer:
shape mismatches, dtype conflicts, and layout violations are caught at **compile time** rather than
at runtime. LibTorch still owns all storage, kernels, autograd, and device execution.

## At a Glance

```cpp
import typetorch;

// Define typed tensor aliases — these ARE the types
using Matrix = typetorch::Tensor<
    typetorch::Shape<2, 3>,          // 2x3 matrix — both dims known at compile time
    typetorch::DType::F32,           // float32
    typetorch::Device::CPU,          // CPU tensor
    typetorch::Layout::Contiguous>;  // dense row-major storage

using Vector  = typetorch::Tensor<typetorch::Shape<typetorch::dyn>, ...>;
//                                     ^^^ runtime-only dimension

// Enter typed code: validate raw tensor at the boundary
auto raw  = torch::arange(6, opts).view({2, 3});
auto x    = Matrix::retain(raw);   // runtime contract check
auto y    = Matrix::ones();        // factory, no raw tensor needed
auto z    = x.add(y);             // shapes match -> compiles, result type computed statically
auto flat = x.flatten<>();        // Shape<2,3> -> Shape<6>, all at compile time

// Leave typed code: recover the raw handle
torch::Tensor out = std::move(flat).unwrap();
```

## Why Typetorch

In ordinary LibTorch code, tensor metadata is dynamic:
- `x.sizes()` returns a runtime vector.
- `x.add(y)` succeeds or fails at runtime based on broadcast compatibility.
- `x.view({...})` throws if the view is invalid.

Typetorch moves these checks **earlier**:
- Shape, dtype, device, and layout are template parameters of `Tensor`.
- Operations compute result types at compile time via `consteval` reflection.
- Mismatches become `static_assert` failures or concept violations — the code
  never reaches the linker.

The runtime cost is **zero**: all wrapper methods are unconditionally inlined,
forwarding directly to the underlying LibTorch call.

## The Type System

```cpp
typetorch::Tensor</*Shape*/ Shape, /*DType*/ DType, /*Device*/ Device, /*Layout*/ Layout>
```

| Parameter  | Values | Meaning |
| --- | --- | --- |
| `Shape` | `Shape<2, 3, 4>`, `Shape<dyn, 3>`, ... | Static dimensions; use `typetorch::dyn` (-2) for runtime-only dims, `typetorch::infer` (-1) for view inference |
| `DType` | `Any`, `F32`, `F16`, `BF16`, `I64`, `Bool` | Element type |
| `Device` | `Any`, `CPU`, `CUDA` | Device placement |
| `Layout` | `Any`, `Contiguous`, `NonContiguous` | Memory layout; many ops preserve or degrade this statically |

Type aliases keep code readable:

```cpp
using Matrix = typetorch::Tensor<typetorch::Shape<2, 3>, typetorch::DType::F32,
                                 typetorch::Device::CPU, typetorch::Layout::Contiguous>;
```

## What Compiles (and Why)

### Broadcasting — shapes checked at compile time

```cpp
using Matrix23 = Tensor<Shape<2, 3>, DType::F32, Device::CPU, Layout::Contiguous>;
using Vector3  = Tensor<Shape<3>,    DType::F32, Device::CPU, Layout::Contiguous>;

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

These are the errors Typetorch exists to catch.

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

## Trust Boundary: retain vs unsafe

Typetorch provides two paths for raw tensor entry:

| Method | Behavior | When to use |
| --- | --- | --- |
| `Tensor::retain(raw)` | Runtime checks: shape, dtype, device, layout | At API boundaries, untrusted inputs |
| `Tensor::unsafe_retain(raw)` | No checks, zero overhead | Inside trusted code where contract is already established |
| `Tensor::unsafe_move(raw&&)` | No checks, moves the raw handle | Same, but avoids the retain bump |

```cpp
// Trusted internal code — zero-overhead path
auto typed = Matrix::unsafe_retain(internal_tensor);
auto result = typed.add(Matrix::ones());

// Untrusted boundary — checked path
auto typed2 = Matrix::retain(user_provided_tensor);
```

`retain()` internally calls `check_tensor_compatible_runtime`, which validates
all four contract parameters against the raw tensor's actual metadata. After
validation, it delegates to `unsafe_retain`.

## Move-Only Semantics

`Tensor<...>` is move-only. This is intentional:

```cpp
auto a = Matrix::ones();
auto b = a;              // ERROR: deleted copy constructor
auto c = std::move(a);   // OK: move is allowed
```

Use `unsafe_raw()` for const reference access without giving up ownership:

```cpp
void inspect(Matrix const &x) {
    torch::Tensor const &raw = x.unsafe_raw();  // OK: read-only access
}
```

Use `unwrap()` to permanently leave the typed world:

```cpp
torch::Tensor raw = std::move(typed).unwrap();  // OK: typed handle consumed
```

## Operation Quick Reference

See [docs/api.md](docs/api.md) for the full operation catalog and contract details.

## Requirements

| Component | Requirement |
| --- | --- |
| C++ compiler | GCC 16.1+ with `-fmodules -freflection` |
| Build system | xmake |
| LibTorch | 2.8.0 + CUDA 12.8 |
| Python headers | For `typetorch_capi_ext` target only |

## Quick Start

### 1. Clone and prepare

```bash
git clone <repo-url> typetorch
cd typetorch
git submodule update --init --recursive
```

### 2. Configure local paths

Typetorch needs to know where your GCC 16.1+ toolchain lives. Create two config
files (both are git-ignored):

```bash
cp scripts/env.local.example.sh scripts/env.local.sh
cp typetorch.local.example.lua typetorch.local.lua
```

**`scripts/env.local.sh`** — shell environment, sourced by `scripts/env.sh`:

```bash
# REQUIRED: path to your GCC 16.1+ installation prefix.
# $GCC_ROOT/bin/gcc and $GCC_ROOT/bin/g++ must exist.
export GCC_ROOT="/home/7/uw07387/gcc-16.1.0"

# OPTIONAL: if your GCC was built as a relocated toolchain (common in HPC
# environments), set these so the linker finds crti.o, crtbeginS.o, etc.
# Leave unset if `gcc -v` shows the right sysroot automatically.
export GCC_TOOLCHAIN_ROOT="/home/7/uw07387/x-tools/x86_64-focal-linux-gnu"
export GCC_TRIPLET="x86_64-focal-linux-gnu"
export GCC_VERSION="16.1.0"
export GCC_SYSROOT="$GCC_TOOLCHAIN_ROOT/$GCC_TRIPLET/sysroot"

# OPTIONAL: extra GCC runtime library path if libstdc++.so is not on the
# default linker search path.
export GCC_RUNTIME_LIB="$GCC_TOOLCHAIN_ROOT/$GCC_TRIPLET/lib64"

# OPTIONAL: custom xmake installation prefix (adds $XMAKE_ROOT/bin to PATH).
export XMAKE_ROOT="/home/7/uw07387/xmake"

# OPTIONAL: Python interpreter. scripts/env.sh auto-detects the include dir.
export PYTHON="python3"
```

Only `GCC_ROOT` is mandatory. If you installed GCC from a tarball into a single
prefix and `gcc -v` prints reasonable `--with-sysroot` and `COLLECT_GCC_OPTIONS`,
you likely only need the first line.

**`typetorch.local.lua`** — xmake configuration, included by `xmake.lua`:

```lua
typetorch_config({
    -- REQUIRED: same value as GCC_ROOT above.
    gcc_root = "/home/7/uw07387/gcc-16.1.0",

    -- OPTIONAL: same values as the shell env vars above.
    -- Only needed when the GCC driver is relocated.
    gcc_toolchain_root = "/home/7/uw07387/x-tools/x86_64-focal-linux-gnu",
    gcc_triplet = "x86_64-focal-linux-gnu",
    gcc_version = "16.1.0",
    gcc_sysroot = "/home/7/uw07387/x-tools/x86_64-focal-linux-gnu/x86_64-focal-linux-gnu/sysroot",

    -- OPTIONAL: escape hatch for manual -B/--sysroot linker flags.
    -- Only set this if the toolchain-root settings above are not enough.
    gcc_relocate_ldflags = "",

    -- OPTIONAL: Python include dir; auto-detected in most cases.
    python_include_dir = "/home/7/uw07387/.venv/include/python3.12",
})
```

### 3. Build and run

```bash
source scripts/env.sh
bash scripts/check_env.sh

xmake f -m debug --yes
xmake build typetorch_cpp_debug
xmake run typetorch_cpp_debug

# Run tensor arithmetic tests
xmake build typetorch_tensor_arithmetic_test
xmake run typetorch_tensor_arithmetic_test
```

## Performance

Release-build forwarding overhead is within measurement noise for all core
operations — see [docs/performance-and-size.md](docs/performance-and-size.md) for
the full benchmark tables and methodology.

### Binary Size Snapshot (release, 2026-06)

| Target | text | data | bss | dec | hex |
| --- | ---: | ---: | ---: | ---: | ---: |
| `typetorch_forwarding_benchmark` | 73363 | 1816 | 200 | 75379 | 12673 |
| `binary_size_libtorch_probe` | 61968 | 1712 | 200 | 63880 | f988 |
| `binary_size_tensor_checked_probe` | 66156 | 1744 | 200 | 68100 | 10a04 |
| `binary_size_tensor_unsafe_probe` | 62488 | 1712 | 200 | 64400 | fb90 |
| `typetorch_tensor_arithmetic_test` | 124074 | 2448 | 200 | 126722 | 1ef02 |

Key comparisons:
- **checked vs native** (66156 vs 61968): +4188 bytes (+6.8%) — full cost of
  Typetorch wrapper + runtime boundary validation.
- **unsafe vs native** (62488 vs 61968): +520 bytes (+0.8%) — cost of Typetorch
  forwarding alone when the caller guarantees the contract.
- **checked vs unsafe** (+3668 bytes, +5.9%) — isolated cost of runtime checks.

The arithmetic test (~124K) is about 2x the baseline probes because it exercises
all op categories (arithmetic, view, NN, aggregate).

## Documentation

- [API Notes](docs/api.md) — contract details, operation style, differences from LibTorch
- [Design Notes](docs/design.md) — architecture decisions
- [Performance and Size Notes](docs/performance-and-size.md) — benchmarks and binary analysis
