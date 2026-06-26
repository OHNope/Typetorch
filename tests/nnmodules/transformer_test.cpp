import std;
import libtorch;
import typetorch;
import fast_io;

#include "../test_support.inc"

namespace
{

using Attention = typetorch::CausalSelfAttention<2, 3, 8, 2>;
using Transformer = typetorch::TypedTransformer<2, 3, 11, 8, 2, 16, 2>;
using Tokens = typetorch::Tensor<typetorch::Shape<2, 3>, typetorch::DType::I64,
                                 typetorch::Device::CPU,
                                 typetorch::Layout::Contiguous>;
using Hidden = typetorch::Tensor<typetorch::Shape<2, 3, 8>,
                                 typetorch::DType::F32,
                                 typetorch::Device::CPU,
                                 typetorch::Layout::Contiguous>;
using Logits = typetorch::Tensor<typetorch::Shape<2, 3, 11>,
                                 typetorch::DType::F32,
                                 typetorch::Device::CPU,
                                 typetorch::Layout::Any>;



} // namespace

int main()
{
    static_assert(Attention::head_dim == 4);
    static_assert(Transformer::batch == 2);
    static_assert(Transformer::sequence_length == 3);
    static_assert(Transformer::vocab_size == 11);
    static_assert(Transformer::embed_dim == 8);
    static_assert(Transformer::layer_count == 2);
    static_assert(::std::same_as<
                  decltype(::std::declval<Transformer::Impl &>().forward(
                      ::std::declval<Tokens const &>())),
                  Logits>);

    Attention attention;
    auto hidden{::torch::randn({2, 3, 8}, typetorch_test::f32_cpu_options())};
    auto attended{attention->forward(Hidden::unsafe_retain(hidden))};
    typetorch_test::expect_shape<2, 3, 8>("CausalSelfAttention", attended.unsafe_raw());

    Transformer transformer;
    auto tokens{::torch::rand({2, 3}, typetorch_test::f32_cpu_options()).mul(11).to(typetorch_test::i64_cpu_options())};
    auto logits{transformer->forward(Tokens::unsafe_retain(tokens))};
    typetorch_test::expect_shape<2, 3, 11>("TypedTransformer logits", logits.unsafe_raw());

    ::fast_io::io::println("typetorch Transformer tests passed");
}
