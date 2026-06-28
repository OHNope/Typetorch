import std;
import libtorch;
import typetorch;
import fast_io;

#include "../test_support.inc"
#include "../../src/torch_macros.inc"

namespace
{

using Input = TYPETORCH_TENSOR((2, 3, 4, 5));
using ReluOutput = Input;
using FlattenOutput =
	TYPETORCH_TENSOR((2, 60));
using Sequence =
	typetorch::Sequential<typetorch::ReLU, typetorch::Flatten<1, -1>>;


} // namespace

int main()
{
	static_assert(::std::same_as<
				  decltype(::std::declval<typetorch::ReLU::Impl &>().forward(
					  ::std::declval<Input const &>())),
				  ReluOutput>);
	static_assert(::std::same_as<
				  decltype(::std::declval<
						   typetorch::Flatten<1, -1>::Impl &>()
							   .forward(::std::declval<Input const &>())),
				  FlattenOutput>);
	static_assert(::std::same_as<
				  decltype(::std::declval<Sequence::Impl &>().forward(
					  ::std::declval<Input const &>())),
				  FlattenOutput>);
	static_assert(typetorch::Flatten<1, -1>::start_dim == 1);
	static_assert(typetorch::Flatten<1, -1>::end_dim == -1);

	auto raw{::torch::randn({2, 3, 4, 5}, typetorch_test::f32_cpu_options())};
	auto sequence{
		typetorch::Sequential{typetorch::ReLU{},
							 typetorch::Flatten<1, -1>{}}};
	auto actual{sequence->forward(TYPETORCH_RETAIN(raw, (2, 3, 4, 5)))};
	auto expected{::torch::relu(raw).flatten(1, -1)};
	if (!::torch::allclose(actual.unsafe_raw(), expected))
	{
		::fast_io::io::perrln("ReLU/Flatten Sequential mismatch");
            ::std::exit(1);
	}

	auto converted{
		sequence.template to<typetorch::Device::CUDA,
							 typetorch::DType::F16>()};
	static_assert(::std::same_as<decltype(converted), Sequence>);

	::fast_io::io::println("typetorch nnModules wrapper tests passed");
}
