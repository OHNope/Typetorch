// Test A: uses current parameter_options() (static inline member variable)
// Instantiates many Tensor types to measure binary/symbol bloat
import std;
import typetorch;
import fast_io;

int main() {
    using T00 = typetorch::Tensor<typetorch::Shape<1>, typetorch::DType::F32, typetorch::Device::CPU, typetorch::Layout::Contiguous>;
    using T01 = typetorch::Tensor<typetorch::Shape<2>, typetorch::DType::F32, typetorch::Device::CPU, typetorch::Layout::Contiguous>;
    using T02 = typetorch::Tensor<typetorch::Shape<1,2>, typetorch::DType::F32, typetorch::Device::CPU, typetorch::Layout::Contiguous>;
    using T03 = typetorch::Tensor<typetorch::Shape<2,3>, typetorch::DType::F32, typetorch::Device::CPU, typetorch::Layout::Contiguous>;
    using T04 = typetorch::Tensor<typetorch::Shape<3,4>, typetorch::DType::F32, typetorch::Device::CPU, typetorch::Layout::Contiguous>;
    using T05 = typetorch::Tensor<typetorch::Shape<1,2,3>, typetorch::DType::F32, typetorch::Device::CPU, typetorch::Layout::Contiguous>;
    using T06 = typetorch::Tensor<typetorch::Shape<2,3,4>, typetorch::DType::F32, typetorch::Device::CPU, typetorch::Layout::Contiguous>;
    using T07 = typetorch::Tensor<typetorch::Shape<1,1,1,1>, typetorch::DType::F32, typetorch::Device::CPU, typetorch::Layout::Contiguous>;
    
    using T08 = typetorch::Tensor<typetorch::Shape<2,3>, typetorch::DType::F16, typetorch::Device::CPU, typetorch::Layout::Contiguous>;
    using T09 = typetorch::Tensor<typetorch::Shape<2,3>, typetorch::DType::BF16, typetorch::Device::CPU, typetorch::Layout::Contiguous>;
    using T10 = typetorch::Tensor<typetorch::Shape<2,3>, typetorch::DType::I64, typetorch::Device::CPU, typetorch::Layout::Contiguous>;
    using T11 = typetorch::Tensor<typetorch::Shape<2,3>, typetorch::DType::Bool, typetorch::Device::CPU, typetorch::Layout::Contiguous>;
    
    using T12 = typetorch::Tensor<typetorch::Shape<2,3>, typetorch::DType::F32, typetorch::Device::CPU, typetorch::Layout::NonContiguous>;
    using T13 = typetorch::Tensor<typetorch::Shape<4,5>, typetorch::DType::F32, typetorch::Device::CPU, typetorch::Layout::NonContiguous>;

    // Trigger parameter_options() access for each
    auto v00 = T00::parameter_options();
    auto v01 = T01::parameter_options();
    auto v02 = T02::parameter_options();
    auto v03 = T03::parameter_options();
    auto v04 = T04::parameter_options();
    auto v05 = T05::parameter_options();
    auto v06 = T06::parameter_options();
    auto v07 = T07::parameter_options();
    auto v08 = T08::parameter_options();
    auto v09 = T09::parameter_options();
    auto v10 = T10::parameter_options();
    auto v11 = T11::parameter_options();
    auto v12 = T12::parameter_options();
    auto v13 = T13::parameter_options();
    
    ::fast_io::io::println("static func test ok");
}
