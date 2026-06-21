import std;
import libtorch;
import typetorch.examples;
import typetorch;
import fast_io;

namespace
{
::torch::Tensor tensor(::std::initializer_list<float> values)
{
	auto options{::torch::TensorOptions().dtype(::torch::kFloat).device(::torch::kCPU)};
	return ::torch::tensor(::std::vector<float>{values}, options);
}

void print_sizes(::torch::Tensor const &x)
{
	::fast_io::io::print("[");
	auto const sizes{x.sizes()};
	for (::std::size_t i{0}; i < sizes.size(); ++i)
	{
		if (i != 0)
		{
			::fast_io::io::print(", ");
		}
		::fast_io::io::print(sizes[i]);
	}
	::fast_io::io::print("]");
}

void print_tensor_summary(char const *name, ::torch::Tensor const &x)
{
	::fast_io::io::print(::std::string_view{name}, ": sizes=");
	print_sizes(x);
	auto const device{x.device().str()};
	::fast_io::io::println(", numel=", x.numel(), ", dtype=",
						   ::std::string_view{c10::toString(x.scalar_type())},
						   ", device=", ::std::string_view{device});
}
} // namespace

int main()
{
	auto x{tensor({1.0F, 2.0F, 3.0F})};
	auto y{tensor({10.0F, 20.0F, 30.0F})};
	auto rows{tensor({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}).view({2, 3})};
	auto weight{tensor({1.0F, 2.0F, 0.0F, 1.0F, 1.0F, 0.0F}).view({3, 2})};
	auto bias{tensor({0.5F, -0.5F, 1.0F})};

	auto x_t{typetorch_examples::Vector::retain(x)};
	auto y_t{typetorch_examples::Vector::retain(y)};
	auto rows_t{typetorch_examples::Matrix::retain(rows)};
	auto rows3_t{typetorch_examples::Matrix3In::retain(rows)};
	auto weight_t{typetorch_examples::Matrix3Out::retain(weight)};
	auto bias_t{typetorch_examples::Vector::retain(bias)};

	print_tensor_summary(
		"checked_vector_sizes",
		::std::move(typetorch_examples::checked_vector_sizes(x_t)).unwrap());
	print_tensor_summary(
		"add_vectors",
		::std::move(typetorch_examples::add_vectors(x_t, y_t)).unwrap());
	print_tensor_summary(
		"add_bias",
		::std::move(typetorch_examples::add_bias(rows_t, bias_t)).unwrap());
	print_tensor_summary(
		"matmul_3_inner",
		::std::move(typetorch_examples::matmul_3_inner(rows3_t, weight_t)).unwrap());
	print_tensor_summary(
		"linear_3_to_2",
		::std::move(typetorch_examples::linear_3_to_2(rows3_t)).unwrap());
	print_tensor_summary(
		"transpose_matrix",
		::std::move(typetorch_examples::transpose_matrix(rows_t)).unwrap());
	::fast_io::io::println("numel_plus: ",
						   typetorch_examples::numel_plus(rows_t, 4));

	using StaticVector =
		typetorch::Tensor<typetorch::Shape<6>, typetorch::DType::F32,
						  typetorch::Device::CPU, typetorch::Layout::Contiguous>;
	using StaticMatrix =
		typetorch::Tensor<typetorch::Shape<2, 3>, typetorch::DType::F32,
						  typetorch::Device::CPU, typetorch::Layout::Contiguous>;
	using StaticCube =
		typetorch::Tensor<typetorch::Shape<2, 3, 4>, typetorch::DType::F32,
						  typetorch::Device::CPU, typetorch::Layout::Contiguous>;

	auto options{::torch::TensorOptions().dtype(::torch::kFloat).device(::torch::kCPU)};
	auto matrix_raw{::torch::arange(6, options).view({2, 3})};
	auto cube_raw{::torch::arange(24, options).view({2, 3, 4})};

	print_tensor_summary("Tensor::empty",
						 ::std::move(StaticMatrix::empty()).unwrap());
	print_tensor_summary("Tensor::zeros",
						 ::std::move(StaticMatrix::zeros()).unwrap());
	print_tensor_summary("Tensor::ones",
						 ::std::move(StaticMatrix::ones()).unwrap());
	print_tensor_summary("Tensor::rand",
						 ::std::move(StaticMatrix::rand()).unwrap());
	print_tensor_summary("Tensor::randn",
						 ::std::move(StaticMatrix::randn()).unwrap());
	print_tensor_summary("Tensor::arange_seq",
						 ::std::move(StaticMatrix::arange<6>()).unwrap());
	print_tensor_summary("Tensor::arange_start_end",
						 ::std::move(StaticMatrix::arange<0, 6>()).unwrap());
	print_tensor_summary(
		"Tensor::arange_step",
		::std::move(StaticMatrix::arange<0, 6, 2, typetorch::DType::I64>())
			.unwrap());
	print_tensor_summary(
		"unsafe_retain_raw",
		StaticMatrix::unsafe_retain(matrix_raw).unsafe_raw());
	print_tensor_summary(
		"add_scalar",
		::std::move(StaticMatrix::retain(matrix_raw)
						.add(::torch::Scalar{2.0F}))
			.unwrap());
	print_tensor_summary(
		"typed_transpose",
		::std::move(StaticMatrix::retain(matrix_raw))
			.transpose<0, 1>()
			.unwrap());
	print_tensor_summary(
		"typed_view",
		::std::move(StaticMatrix::retain(matrix_raw))
			.view<6>()
			.unwrap());
	print_tensor_summary(
		"typed_permute",
		::std::move(StaticCube::retain(cube_raw))
			.permute<0, 2, 1>()
			.unwrap());
	print_tensor_summary(
		"typed_permute_negative",
		::std::move(StaticCube::retain(cube_raw))
			.permute<0, -1, 1>()
			.unwrap());
	print_tensor_summary(
		"typed_permute_identity",
		::std::move(StaticCube::retain(cube_raw))
			.permute<0, 1, 2>()
			.unwrap());
	print_tensor_summary(
		"typed_permute_contiguous",
		::std::move(::std::move(StaticCube::retain(cube_raw))
						.permute<0, 2, 1>())
			.contiguous()
			.unwrap());
	auto static_vector{StaticVector::retain(
		::torch::arange(6, options).view({6}))};
	::fast_io::io::println("typetorch::sizes: ",
						   static_vector.sizes().size());
}
