import std;
import libtorch;
import typetorch;
import fast_io;

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

auto float_options() -> ::torch::TensorOptions
{
    return ::torch::TensorOptions{}.dtype(::torch::kFloat).device(::torch::kCPU);
}

auto index_options() -> ::torch::TensorOptions
{
    return ::torch::TensorOptions{}.dtype(::torch::kLong).device(::torch::kCPU);
}

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
    auto hidden{::torch::randn({2, 3, 8}, float_options())};
    auto attended{attention->forward(Hidden::unsafe_retain(hidden))};
    if (attended.unsafe_raw().size(0) != 2 || attended.unsafe_raw().size(1) != 3 || attended.unsafe_raw().size(2) != 8)
    {
        ::fast_io::io::perrln("CausalSelfAttention shape mismatch");
        ::std::exit(1);
    }

    Transformer transformer;
    auto tokens{::torch::rand({2, 3}, float_options()).mul(11).to(index_options())};
    auto logits{transformer->forward(Tokens::unsafe_retain(tokens))};
    if (logits.unsafe_raw().size(0) != 2 || logits.unsafe_raw().size(1) != 3 || logits.unsafe_raw().size(2) != 11)
    {
        ::fast_io::io::perrln("TypedTransformer logits shape mismatch");
        ::std::exit(1);
    }

    ::fast_io::io::println("typetorch Transformer tests passed");
}
