import std;
import libtorch;
import tenspec;
import fastio;

int main()
{
    using Matrix = tenspec::Tensor<tenspec::Shape<2, 3>, tenspec::DType::F32,
        tenspec::Device::CPU, tenspec::Layout::Contiguous>;

    static_assert(Matrix::shape::rank == 2);
    static_assert(Matrix::dtype == tenspec::DType::F32);

    double const raw_ns{1.25};
    double const tenspec_ns{2.5};
    auto const ratio{tenspec_ns / raw_ns};

    ::fast_io::io::println("benchmark result was optimized away");
    ::fast_io::io::println(::std::string_view{"sample"}, ", raw_ns=",
        raw_ns, ", tenspec_ns=", tenspec_ns, ", ratio=", ratio);
}
