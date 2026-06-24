import std;
import typetorch;

int main()
{
    using Matrix = typetorch::Tensor<typetorch::Shape<2, 3>,
                                     typetorch::DType::F32,
                                     typetorch::Device::CPU,
                                     typetorch::Layout::Contiguous>;
    auto x = Matrix::ones();
    auto y = x.add(x);
    auto flat = std::move(y).contiguous().view<6>();
    auto raw = std::move(flat).unwrap();
    if (raw.numel() != 6)
        return 1;
    return 0;
}
