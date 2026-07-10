import std;
import libtorch;
import typetorch;
import fast_io;

#include "test_support.inc"
#include "../src/torch_macros.inc"

namespace
{

using Matrix = TYPETORCH_TENSOR((2, 3));
using Vector6 = TYPETORCH_TENSOR((6));
using MatrixT =
	TYPETORCH_TENSOR_LAYOUT((3, 2), typetorch::DType::F32,
							typetorch::Device::CPU, typetorch::Layout::Any);
using Cube = TYPETORCH_TENSOR((2, 3, 4));
using CubePermuted =
	TYPETORCH_TENSOR_LAYOUT((2, 4, 3), typetorch::DType::F32,
							typetorch::Device::CPU, typetorch::Layout::Any);
using CubePermutedContiguous = TYPETORCH_TENSOR((2, 4, 3));
using Reshaped = TYPETORCH_TENSOR((2, 12));
using Unsqueezed = TYPETORCH_TENSOR((1, 2, 3));
using Squeezable = TYPETORCH_TENSOR((1, 2, 1, 3));
using Squeezed = TYPETORCH_TENSOR((2, 3));
using ArangeI64 = TYPETORCH_TENSOR_OPT((3), typetorch::DType::I64, typetorch::Device::CPU);

static_assert(::std::same_as<decltype(Matrix::empty()), Matrix>);
static_assert(::std::same_as<decltype(Matrix::zeros()), Matrix>);
static_assert(::std::same_as<decltype(Matrix::ones()), Matrix>);
static_assert(::std::same_as<decltype(Matrix::rand()), Matrix>);
static_assert(::std::same_as<decltype(Matrix::randn()), Matrix>);
static_assert(::std::same_as<decltype(Matrix::arange<6>()), Vector6>);
static_assert(::std::same_as<decltype(Matrix::arange<0, 6>()), Vector6>);
static_assert(::std::same_as<decltype(Matrix::arange<0, 6, 2, typetorch::DType::I64>()), ArangeI64>);
static_assert(::std::same_as<decltype(::std::declval<Matrix &&>().transpose<0, 1>()), MatrixT>);
static_assert(::std::same_as<decltype(::std::declval<Cube &&>().permute<0, 2, 1>()), CubePermuted>);
static_assert(::std::same_as<decltype(::std::declval<CubePermuted &&>().contiguous()), CubePermutedContiguous>);
static_assert(::std::same_as<decltype(::std::declval<Matrix &&>().contiguous()), Matrix>);
static_assert(::std::same_as<decltype(::std::declval<Matrix const &>().detach()), Matrix>);
static_assert(!noexcept(::std::declval<Matrix const &>().sizes()));
static_assert(::std::same_as<
	decltype(::std::declval<Matrix &>().unsafe_wrap(
		::std::declval<::torch::Tensor &&>())),
	Matrix>);
static_assert(::std::same_as<decltype(::std::declval<Matrix &&>().view<6>()), Vector6>);
static_assert(::std::same_as<decltype(::std::declval<Matrix &&>().reshape<6>()), Vector6>);
static_assert(::std::same_as<decltype(::std::declval<CubePermuted &&>().reshape<2, 12>()), Reshaped>);
static_assert(::std::same_as<decltype(::std::declval<Matrix const &>().flatten<>()), Vector6>);
static_assert(::std::same_as<decltype(::std::declval<Matrix const &>().unsqueeze<0>()), Unsqueezed>);
static_assert(::std::same_as<decltype(::std::declval<Squeezable const &>().squeeze()), Squeezed>);
static_assert(::std::same_as<decltype(::std::declval<Squeezable const &>().squeeze<0>()),
							 TYPETORCH_TENSOR((2, 1, 3))>);

[[nodiscard]] auto matrix_raw() -> ::torch::Tensor
{
	return ::torch::arange(6, typetorch_test::f32_cpu_options()).view({2, 3});
}

[[nodiscard]] auto cube_raw() -> ::torch::Tensor
{
	return ::torch::arange(24, typetorch_test::f32_cpu_options()).view({2, 3, 4});
}

} // namespace

int main()
{
	typetorch_test::expect_shape<2, 3>("empty", Matrix::empty().unsafe_raw());
	typetorch_test::expect_allclose("zeros", Matrix::zeros().unsafe_raw(),
									::torch::zeros({2, 3}, typetorch_test::f32_cpu_options()));
	typetorch_test::expect_allclose("ones", Matrix::ones().unsafe_raw(),
									::torch::ones({2, 3}, typetorch_test::f32_cpu_options()));
	typetorch_test::expect_shape<2, 3>("rand", Matrix::rand().unsafe_raw());
	typetorch_test::expect_shape<2, 3>("randn", Matrix::randn().unsafe_raw());
	typetorch_test::expect_allclose("arange_seq", Matrix::arange<6>().unsafe_raw(),
									::torch::arange(6, typetorch_test::f32_cpu_options()));
	typetorch_test::expect_allclose("arange_start_end", Matrix::arange<0, 6>().unsafe_raw(),
									::torch::arange(0, 6, typetorch_test::f32_cpu_options()));
	typetorch_test::expect_allclose(
		"arange_step_i64",
		Matrix::arange<0, 6, 2, typetorch::DType::I64>().unsafe_raw(),
		::torch::arange(0, 6, 2, typetorch_test::i64_cpu_options()));

	auto matrix{matrix_raw()};
	typetorch_test::expect_allclose(
		"transpose",
		TYPETORCH_RETAIN(matrix, (2, 3)).transpose<0, 1>().unsafe_raw(),
		matrix.transpose(0, 1));
	typetorch_test::expect_allclose(
		"view",
		TYPETORCH_RETAIN(matrix, (2, 3)).view<6>().unsafe_raw(),
		matrix.view({6}));
	typetorch_test::expect_allclose(
		"reshape_contiguous",
		TYPETORCH_RETAIN(matrix, (2, 3)).reshape<6>().unsafe_raw(),
		matrix.reshape({6}));
	typetorch_test::expect_allclose(
		"contiguous_already_contiguous",
		TYPETORCH_RETAIN(matrix, (2, 3)).contiguous().unsafe_raw(), matrix);
	typetorch_test::expect_allclose(
		"detach",
		TYPETORCH_RETAIN(matrix, (2, 3)).detach().unsafe_raw(), matrix.detach());
	typetorch_test::expect_allclose(
		"flatten",
		TYPETORCH_RETAIN(matrix, (2, 3)).flatten<>().unsafe_raw(),
		matrix.flatten());
	typetorch_test::expect_allclose(
		"unsqueeze",
		TYPETORCH_RETAIN(matrix, (2, 3)).unsqueeze<0>().unsafe_raw(),
		matrix.unsqueeze(0));

	auto cube{cube_raw()};
	auto expected_permuted{cube.permute({0, 2, 1})};
	typetorch_test::expect_allclose(
		"permute",
		TYPETORCH_RETAIN(cube, (2, 3, 4)).permute<0, 2, 1>().unsafe_raw(),
		expected_permuted);
	typetorch_test::expect_allclose(
		"contiguous",
		TYPETORCH_RETAIN(cube, (2, 3, 4)).permute<0, 2, 1>().contiguous().unsafe_raw(),
		expected_permuted.contiguous());
	typetorch_test::expect_allclose(
		"reshape_non_contiguous",
		TYPETORCH_RETAIN(cube, (2, 3, 4)).permute<0, 2, 1>().reshape<2, 12>().unsafe_raw(),
		expected_permuted.contiguous().view({2, 12}));

	auto squeezable{matrix.view({1, 2, 1, 3})};
	typetorch_test::expect_allclose(
		"squeeze_all",
		TYPETORCH_RETAIN(squeezable, (1, 2, 1, 3)).squeeze().unsafe_raw(),
		squeezable.squeeze());
	typetorch_test::expect_allclose(
		"squeeze_dim",
		TYPETORCH_RETAIN(squeezable, (1, 2, 1, 3)).squeeze<0>().unsafe_raw(),
		squeezable.squeeze(0));

	::fast_io::io::println("typetorch tensor factory/view tests passed");
}
