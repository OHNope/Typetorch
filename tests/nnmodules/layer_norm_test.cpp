import std;
import libtorch;
import typetorch;
import fast_io;

#include "../test_support.inc"
#include "../../src/torch_macros.inc"

namespace {

using Input23 = TYPETORCH_TENSOR((2, 3));
using Input224 = TYPETORCH_TENSOR((2, 2, 4));



} // namespace

int main() {
    // --- Test 1: forward affine=true ---
    {
        ::torch::nn::LayerNorm raw(
            ::torch::nn::LayerNormOptions({2, 3}));
        using LN = typetorch::LayerNorm<typetorch::Shape<2, 3>>;
        LN typed;

        typed->weight.set_data(raw->weight);
        typed->bias.set_data(raw->bias);

        auto input = ::torch::randn({2, 3}, typetorch_test::f32_cpu_options());
        auto expected = raw->forward(input);
        auto actual = typed->forward(TYPETORCH_RETAIN(input, (2, 3)));

        if (!::torch::allclose(actual.unsafe_raw(), expected)) {
            ::fast_io::io::perrln("layernorm_affine: mismatch");
            ::std::exit(1);
        }
    }

    // --- Test 2: forward affine=false ---
    {
        ::torch::nn::LayerNorm raw(
            ::torch::nn::LayerNormOptions({4}).elementwise_affine(false));
        using LN = typetorch::LayerNorm<typetorch::Shape<4>>;
        LN typed(/*eps=*/1e-5, /*elementwise_affine=*/false);

        auto input = ::torch::randn({2, 2, 4}, typetorch_test::f32_cpu_options());
        auto expected = raw->forward(input);
        auto actual = typed->forward(TYPETORCH_RETAIN(input, (2, 2, 4)));

        if (!::torch::allclose(actual.unsafe_raw(), expected)) {
            ::fast_io::io::perrln("layernorm_no_affine: mismatch");
            ::std::exit(1);
        }
    }

    // --- Test 3: to(dtype) F16 ---
    {
        ::torch::nn::LayerNorm raw(
            ::torch::nn::LayerNormOptions({2, 3}));
        using LN = typetorch::LayerNorm<typetorch::Shape<2, 3>>;
        LN typed;
        typed->weight.set_data(raw->weight);
        typed->bias.set_data(raw->bias);

        auto f16 = typed->template to<typetorch::DType::F16>();
        if (!f16->weight.equal(raw->weight.to(::torch::kHalf))) {
            ::fast_io::io::perrln("layernorm_to_f16: weight mismatch");
            ::std::exit(1);
        }
        if (!f16->bias.equal(raw->bias.to(::torch::kHalf))) {
            ::fast_io::io::perrln("layernorm_to_f16: bias mismatch");
            ::std::exit(1);
        }
    }

    // --- Test 4: typed_weight / typed_bias shapes ---
    {
        using LN = typetorch::LayerNorm<typetorch::Shape<2, 3>>;
        LN typed;
        auto w = typed->typed_weight();
        auto b = typed->typed_bias();
        static_assert(::std::same_as<decltype(w)::shape, typetorch::Shape<2, 3>>);
        static_assert(::std::same_as<decltype(b)::shape, typetorch::Shape<2, 3>>);
    }

    ::fast_io::io::println("typetorch nnModules layer_norm tests passed");
}
