import std;
import libtorch;
import typetorch;
import fast_io;

#if defined(__GNUC__) || defined(__clang__)
#define TYPETORCH_SIZE_NOINLINE __attribute__((noinline, used))
#else
#define TYPETORCH_SIZE_NOINLINE
#endif

namespace typetorch_binary_size_tensor_probe
{

using Vector = typetorch::Tensor<typetorch::Shape<typetorch::dyn>,
								 typetorch::DType::F32, typetorch::Device::CPU,
								 typetorch::Layout::Any>;
TYPETORCH_SIZE_NOINLINE auto raw_identity(::at::Tensor const &x) -> ::at::Tensor
{
	return x;
}

TYPETORCH_SIZE_NOINLINE auto raw_add(::at::Tensor const &a, ::at::Tensor const &b)
	-> ::at::Tensor
{
	return a.add(b);
}

TYPETORCH_SIZE_NOINLINE auto raw_numel(::at::Tensor const &x) -> ::std::int64_t
{
	return x.numel();
}

TYPETORCH_SIZE_NOINLINE auto typed_raw_ref(Vector const &x)
	-> ::at::Tensor const &
{
	return x.unsafe_raw();
}

TYPETORCH_SIZE_NOINLINE auto typed_retain_checked(::at::Tensor const &x) -> Vector
{
	return Vector::retain(x);
}

TYPETORCH_SIZE_NOINLINE auto typed_retain_unsafe(::at::Tensor const &x) -> Vector
{
	return Vector::unsafe_retain(x);
}

TYPETORCH_SIZE_NOINLINE auto typed_unwrap(Vector x) -> ::at::Tensor
{
	return ::std::move(x).unwrap();
}

TYPETORCH_SIZE_NOINLINE auto typed_add(Vector const &a, Vector const &b)
	-> Vector
{
	return a.add(b);
}

TYPETORCH_SIZE_NOINLINE auto typed_numel(Vector const &x) -> ::std::int64_t
{
	return x.unsafe_raw().numel();
}

} // namespace typetorch_binary_size_tensor_probe

int main()
{
	namespace probe = typetorch_binary_size_tensor_probe;

	auto options{::at::TensorOptions().dtype(::at::kFloat).device(::at::kCPU)};
	auto a{::at::ones({4}, options)};
	auto b{::at::ones({4}, options)};

	auto raw_sum{probe::raw_add(a, b)};
	auto typed_a{probe::typed_retain_unsafe(a)};
	auto typed_b{probe::typed_retain_checked(b)};
	auto typed_sum{probe::typed_add(typed_a, typed_b)};
	auto unwrapped{probe::typed_unwrap(::std::move(typed_sum))};

	auto const total{probe::raw_numel(raw_sum) + probe::typed_numel(typed_a) +
					 probe::typed_raw_ref(typed_a).numel() +
					 probe::raw_identity(unwrapped).numel()};
	::fast_io::io::println(total);
}
