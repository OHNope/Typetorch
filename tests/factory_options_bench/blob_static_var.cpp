// Test A: uses current factory_options (static inline member variable)
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

    // Trigger factory_options access for each
    auto v00 = T00::factory_options;
    auto v01 = T01::factory_options;
    auto v02 = T02::factory_options;
    auto v03 = T03::factory_options;
    auto v04 = T04::factory_options;
    auto v05 = T05::factory_options;
    auto v06 = T06::factory_options;
    auto v07 = T07::factory_options;
    auto v08 = T08::factory_options;
    auto v09 = T09::factory_options;
    auto v10 = T10::factory_options;
    auto v11 = T11::factory_options;
    auto v12 = T12::factory_options;
    auto v13 = T13::factory_options;
    
    ::fast_io::io::println("static var test ok");
}
