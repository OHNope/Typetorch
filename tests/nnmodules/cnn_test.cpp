import std;
import libtorch;
import typetorch;
import fast_io;

#include "../test_support.inc"
#include "../../src/torch_macros.inc"

namespace
{

using CNN = typetorch::TypedCNN<typetorch::Shape<3, 5, 7>, 4>;
using Input = TYPETORCH_TENSOR((2, 3, 11, 13));
using Output =
	TYPETORCH_TENSOR_LAYOUT((2, 4), typetorch::DType::F32,
							typetorch::Device::CPU, typetorch::Layout::Any);

} // namespace

int main()
{
	static_assert(CNN::conv_count == 2);
	static_assert(CNN::input_channels == 3);
	static_assert(CNN::feature_channels == 7);
	static_assert(CNN::pooled_features == 7);
	static_assert(CNN::num_classes == 4);
	static_assert(::std::same_as<
				  decltype(::std::declval<CNN::Impl &>().forward(
					  ::std::declval<Input const &>())),
				  Output>);
	static_assert(::std::same_as<
				  decltype(::std::declval<CNN &>().template get_conv<0>()),
				  typetorch::Conv2d<3, 5, typetorch::Shape<3, 3>,
									typetorch::Shape<1, 1>,
									typetorch::Shape<1, 1>> &>);
	static_assert(::std::same_as<
				  decltype(::std::declval<CNN &>().classifier()),
				  typetorch::Linear<7, 4> &>);
	static_assert(::std::same_as<
				  decltype(::std::declval<CNN const &>()
							   .template to<typetorch::DType::F16>()),
				  typetorch::TypedCNN<typetorch::Shape<3, 5, 7>, 4,
									  typetorch::Shape<3, 3>,
									  typetorch::Shape<1, 1>,
									  typetorch::Shape<1, 1>,
									  typetorch::DType::F16,
									  typetorch::Device::CPU>>);

	CNN typed;
	::torch::nn::Conv2d raw0{
		::torch::nn::Conv2dOptions(3, 5, 3).padding(1).bias(true)};
	::torch::nn::Conv2d raw1{
		::torch::nn::Conv2dOptions(5, 7, 3).padding(1).bias(true)};
	::torch::nn::Linear raw_classifier{
		::torch::nn::LinearOptions(7, 4).bias(true)};

	typed.template get_conv<0>()->weight.set_data(raw0->weight);
	typed.template get_conv<0>()->bias.set_data(raw0->bias);
	typed.template get_conv<1>()->weight.set_data(raw1->weight);
	typed.template get_conv<1>()->bias.set_data(raw1->bias);
	typed.classifier()->weight.set_data(raw_classifier->weight);
	typed.classifier()->bias.set_data(raw_classifier->bias);

	auto input{::torch::randn({2, 3, 11, 13},
							  typetorch_test::f32_cpu_options())};
	auto expected{
		raw_classifier->forward(
			::torch::adaptive_avg_pool2d(
				::torch::relu(raw1->forward(
					::torch::relu(raw0->forward(input)))),
				::std::array<::std::int64_t, 2>{1, 1})
				.flatten(1, -1))};
	auto actual{typed->forward(TYPETORCH_RETAIN(input, (2, 3, 11, 13)))};
	typetorch_test::expect_allclose("TypedCNN forward",
									actual.unsafe_raw(), expected);

	typetorch::TypedCNN<typetorch::Shape<3, 4>, 2> no_bias{false};
	if (no_bias.template get_conv<0>()->bias.defined() ||
		no_bias.classifier()->bias.defined())
	{
		::fast_io::io::perrln("TypedCNN no-bias construction failed");
		::std::exit(1);
	}

	auto f16{typed.template to<typetorch::DType::F16>()};
	if (!f16.template get_conv<0>()->weight.equal(
			raw0->weight.to(::torch::kHalf)))
	{
		::fast_io::io::perrln("TypedCNN to<F16>: conv weight mismatch");
		::std::exit(1);
	}
	if (!f16.classifier()->weight.equal(
			raw_classifier->weight.to(::torch::kHalf)))
	{
		::fast_io::io::perrln("TypedCNN to<F16>: classifier weight mismatch");
		::std::exit(1);
	}

	::fast_io::io::println("typetorch TypedCNN tests passed");
}
