import std;
import libtorch;
import typetorch;
import fast_io;

#include "../test_support.inc"
#include "../../src/torch_macros.inc"

namespace
{

using MLP = typetorch::TypedMLP<typetorch::Shape<3, 5, 4, 2>>;
using Input = TYPETORCH_TENSOR((7, 3));
using Output =
	TYPETORCH_TENSOR_LAYOUT((7, 2), typetorch::DType::F32,
							typetorch::Device::CPU, typetorch::Layout::Any);


} // namespace

int main()
{
	static_assert(MLP::layer_count == 3);
	static_assert(MLP::input_features == 3);
	static_assert(MLP::output_features == 2);
	static_assert(::std::same_as<
				  decltype(::std::declval<MLP::Impl &>().forward(
					  ::std::declval<Input const &>())),
				  Output>);
	static_assert(::std::same_as<
				  decltype(::std::declval<MLP &>().template get<0>()),
				  typetorch::Linear<3, 5> &>);
	static_assert(::std::same_as<
				  decltype(::std::declval<MLP &>().template get<2>()),
				  typetorch::Linear<4, 2> &>);
	static_assert(::std::same_as<
				  decltype(::std::declval<MLP const &>()
							   .template to<typetorch::DType::F16>()),
				  typetorch::TypedMLP<typetorch::Shape<3, 5, 4, 2>,
									  typetorch::DType::F16,
									  typetorch::Device::CPU>>);

	MLP typed;
	::torch::nn::Linear raw0{::torch::nn::LinearOptions{3, 5}};
	::torch::nn::Linear raw1{::torch::nn::LinearOptions{5, 4}};
	::torch::nn::Linear raw2{::torch::nn::LinearOptions{4, 2}};
	typed.template get<0>()->weight.set_data(raw0->weight);
	typed.template get<0>()->bias.set_data(raw0->bias);
	typed.template get<1>()->weight.set_data(raw1->weight);
	typed.template get<1>()->bias.set_data(raw1->bias);
	typed.template get<2>()->weight.set_data(raw2->weight);
	typed.template get<2>()->bias.set_data(raw2->bias);

	auto input{::torch::randn({7, 3}, typetorch_test::f32_cpu_options())};
	auto expected{
		raw2->forward(::torch::relu(raw1->forward(
			::torch::relu(raw0->forward(input)))))};
	auto actual{typed->forward(TYPETORCH_RETAIN(input, (7, 3)))};
	if (!::torch::allclose(actual.unsafe_raw(), expected))
	{
		::fast_io::io::perrln("TypedMLP forward mismatch");
		::std::exit(1);
	}

	typetorch::TypedMLP<typetorch::Shape<3, 4, 2>> no_bias{false};
	if (no_bias.template get<0>()->bias.defined() ||
		no_bias.template get<1>()->bias.defined())
	{
		::fast_io::io::perrln("TypedMLP no-bias construction failed");
		::std::exit(1);
	}

	::fast_io::io::println("typetorch TypedMLP tests passed");
}
