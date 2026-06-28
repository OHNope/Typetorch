import std;
import libtorch;
import typetorch;
import fast_io;

#include "../src/attributes.inc"
#include "../src/torch_macros.inc"

namespace typetorch_binary_size_tensor_checked_probe
{

using InputVector = TYPETORCH_TENSOR((4));
using OutputVector =
	TYPETORCH_TENSOR_LAYOUT((4), typetorch::DType::F32,
							typetorch::Device::CPU, typetorch::Layout::Any);

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

TYPETORCH_SIZE_NOINLINE auto typed_raw_ref(InputVector const &x)
	-> ::torch::Tensor const &
{
	return x.unsafe_raw();
}

TYPETORCH_SIZE_NOINLINE auto typed_retain_checked(::torch::Tensor const &x)
	-> InputVector
{
	return InputVector::retain(x);
}

TYPETORCH_SIZE_NOINLINE auto typed_unwrap(OutputVector x) -> ::torch::Tensor
{
	return ::std::move(x).unwrap();
}

TYPETORCH_SIZE_NOINLINE auto typed_add(InputVector const &a,
									   InputVector const &b) -> OutputVector
{
	return a.add(b);
}

TYPETORCH_SIZE_NOINLINE auto typed_numel(InputVector const &x) -> ::std::int64_t
{
	return x.unsafe_raw().numel();
}

} // namespace typetorch_binary_size_tensor_checked_probe

int main()
{
	namespace probe = typetorch_binary_size_tensor_checked_probe;

	auto options{::torch::TensorOptions().dtype(::torch::kFloat).device(::torch::kCPU)};
	auto a{::torch::ones({4}, options)};
	auto b{::torch::ones({4}, options)};

	auto raw_sum{probe::raw_add(a, b)};
	auto typed_a{probe::typed_retain_checked(a)};
	auto typed_b{probe::typed_retain_checked(b)};
	auto typed_sum{probe::typed_add(typed_a, typed_b)};
	auto unwrapped{probe::typed_unwrap(::std::move(typed_sum))};

	auto const total{probe::raw_numel(raw_sum) + probe::typed_numel(typed_a) +
					 probe::typed_raw_ref(typed_a).numel() +
					 probe::raw_identity(unwrapped).numel()};
	::fast_io::io::println(total);
}
