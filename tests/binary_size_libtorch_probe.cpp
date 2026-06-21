import std;
import libtorch;
import fast_io;

#if defined(__GNUC__) || defined(__clang__)
#define TYPETORCH_SIZE_NOINLINE __attribute__((noinline, used))
#else
#define TYPETORCH_SIZE_NOINLINE
#endif

namespace typetorch_binary_size_libtorch_probe
{

TYPETORCH_SIZE_NOINLINE auto raw_identity(::torch::Tensor const &x) -> ::torch::Tensor
{
	return x;
}

TYPETORCH_SIZE_NOINLINE auto raw_add(::torch::Tensor const &a, ::torch::Tensor const &b)
	-> ::torch::Tensor
{
	return a.add(b);
}

TYPETORCH_SIZE_NOINLINE auto raw_numel(::torch::Tensor const &x) -> ::std::int64_t
{
	return x.numel();
}

TYPETORCH_SIZE_NOINLINE auto native_raw_ref(::torch::Tensor const &x)
	-> ::torch::Tensor const &
{
	return x;
}

TYPETORCH_SIZE_NOINLINE auto native_retain_checked(::torch::Tensor const &x)
	-> ::torch::Tensor
{
	return x;
}

TYPETORCH_SIZE_NOINLINE auto native_retain_unsafe(::torch::Tensor const &x)
	-> ::torch::Tensor
{
	return x;
}

TYPETORCH_SIZE_NOINLINE auto native_unwrap(::torch::Tensor x) -> ::torch::Tensor
{
	return x;
}

TYPETORCH_SIZE_NOINLINE auto native_add(::torch::Tensor const &a,
										::torch::Tensor const &b) -> ::torch::Tensor
{
	return a.add(b);
}

TYPETORCH_SIZE_NOINLINE auto native_numel(::torch::Tensor const &x) -> ::std::int64_t
{
	return x.numel();
}

} // namespace typetorch_binary_size_libtorch_probe

int main()
{
	namespace probe = typetorch_binary_size_libtorch_probe;

	auto options{::torch::TensorOptions().dtype(::torch::kFloat).device(::torch::kCPU)};
	auto a{::torch::ones({4}, options)};
	auto b{::torch::ones({4}, options)};

	auto raw_sum{probe::raw_add(a, b)};
	auto native_a{probe::native_retain_unsafe(a)};
	auto native_b{probe::native_retain_checked(b)};
	auto native_sum{probe::native_add(native_a, native_b)};
	auto unwrapped{probe::native_unwrap(::std::move(native_sum))};

	auto const total{probe::raw_numel(raw_sum) + probe::native_numel(native_a) +
					 probe::native_raw_ref(native_a).numel() +
					 probe::raw_identity(unwrapped).numel()};
	::fast_io::io::println(total);
}
