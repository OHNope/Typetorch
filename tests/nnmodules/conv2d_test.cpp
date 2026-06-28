import std;
import libtorch;
import typetorch;
import fast_io;

#include "../test_support.inc"
#include "../../src/torch_macros.inc"

int main()
{
	// --- Test 1: forward with bias, default stride/pad ---
	{
		::torch::nn::Conv2d raw(
			::torch::nn::Conv2dOptions(3, 6, 3).bias(true));
		using Conv = typetorch::Conv2d<3, 6, typetorch::Shape<3, 3>>;
		Conv typed;

		typed->weight.set_data(raw->weight);
		typed->bias.set_data(raw->bias);

		auto typed_input = TYPETORCH_RANDN((2, 3, 8, 8));
		auto input = typed_input.unsafe_raw();
		auto expected = raw->forward(input);
		auto actual = typed->forward(TYPETORCH_RETAIN(input, (2, 3, 8, 8)));

		typetorch_test::expect_allclose("conv2d_default", actual.unsafe_raw(), expected);
	}

	// --- Test 2: forward with stride=2, pad=1, no bias ---
	{
		::torch::nn::Conv2d raw(
			::torch::nn::Conv2dOptions(3, 6, 3)
				.stride(2)
				.padding(1)
				.bias(false));
		using Conv = typetorch::Conv2d<3, 6, typetorch::Shape<3, 3>,
									   typetorch::Shape<2, 2>, typetorch::Shape<1, 1>>;
		Conv typed{/*with_bias=*/false};

		typed->weight.set_data(raw->weight);

		auto typed_input = TYPETORCH_RANDN((2, 3, 8, 8));
		auto input = typed_input.unsafe_raw();
		auto expected = raw->forward(input);
		auto actual = typed->forward(::std::move(typed_input));

		typetorch_test::expect_allclose("conv2d_stride2_pad1", actual.unsafe_raw(), expected);
	}

	// --- Test 3: to(dtype) F16 ---
	{
		::torch::nn::Conv2d raw(
			::torch::nn::Conv2dOptions(3, 6, 3).bias(true));
		using Conv = typetorch::Conv2d<3, 6, typetorch::Shape<3, 3>>;
		Conv typed;
		typed->weight.set_data(raw->weight);
		typed->bias.set_data(raw->bias);

		auto f16 = typed->template to<typetorch::DType::F16>();
		if (!f16->weight.equal(raw->weight.to(::torch::kHalf)))
		{
			::fast_io::io::perrln("conv2d_to_f16: weight mismatch");
			::std::exit(1);
		}
	}

	// --- Test 4: typed_weight / typed_bias shapes ---
	{
		using Conv = typetorch::Conv2d<3, 6, typetorch::Shape<3, 3>>;
		Conv typed;
		auto w = typed->typed_weight();
		auto b = typed->typed_bias();
		static_assert(::std::same_as<decltype(w)::shape, typetorch::Shape<6, 3, 3, 3>>);
		static_assert(::std::same_as<decltype(b)::shape, typetorch::Shape<6>>);
	}

	// --- Test 5: static constexpr members ---
	{
		using Conv = typetorch::Conv2d<3, 6, typetorch::Shape<3, 3>>;
		static_assert(Conv::in_channels == 3);
		static_assert(Conv::out_channels == 6);
		static_assert(Conv::groups == 1);
	}

	::fast_io::io::println("typetorch nnModules conv2d tests passed");
}
