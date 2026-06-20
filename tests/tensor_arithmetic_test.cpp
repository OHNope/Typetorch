import std;
import libtorch;
import tenspec;
import fastio;

namespace
{

using Matrix = tenspec::Tensor<tenspec::Shape<2, 3>, tenspec::DType::F32,
							   tenspec::Device::CPU, tenspec::Layout::Contiguous>;
using MatrixAny = tenspec::Tensor<tenspec::Shape<2, 3>, tenspec::DType::F32,
								  tenspec::Device::CPU, tenspec::Layout::Any>;
using Bias = tenspec::Tensor<tenspec::Shape<3>, tenspec::DType::F32,
							 tenspec::Device::CPU, tenspec::Layout::Contiguous>;
using Column = tenspec::Tensor<tenspec::Shape<2, 1>, tenspec::DType::F32,
							   tenspec::Device::CPU, tenspec::Layout::Contiguous>;
using MatrixFlat = tenspec::Tensor<tenspec::Shape<6>, tenspec::DType::F32,
							   tenspec::Device::CPU, tenspec::Layout::Contiguous>;
using MatrixUnsqueezed = tenspec::Tensor<tenspec::Shape<1, 2, 3>, tenspec::DType::F32,
								 tenspec::Device::CPU, tenspec::Layout::Contiguous>;
using Squeezable = tenspec::Tensor<tenspec::Shape<1, 2, 1, 3>, tenspec::DType::F32,
								  tenspec::Device::CPU, tenspec::Layout::Contiguous>;
using Squeezed = tenspec::Tensor<tenspec::Shape<2, 3>, tenspec::DType::F32,
								 tenspec::Device::CPU, tenspec::Layout::Contiguous>;
using BoolMatrix = tenspec::Tensor<tenspec::Shape<2, 3>, tenspec::DType::Bool,
								  tenspec::Device::CPU, tenspec::Layout::Contiguous>;

static_assert(::std::same_as<
			  decltype(::std::declval<Matrix const &>().sub(
				  ::std::declval<Bias const &>())),
			  MatrixAny>);
static_assert(::std::same_as<
			  decltype(::std::declval<Matrix const &>().mul(
				  ::std::declval<Column const &>())),
			  MatrixAny>);
static_assert(::std::same_as<
			  decltype(::std::declval<Matrix const &>().div(
				  ::std::declval<Bias const &>())),
			  MatrixAny>);
static_assert(::std::same_as<
			  decltype(::std::declval<Matrix const &>().sub(::at::Scalar{1.0F})),
			  Matrix>);
static_assert(::std::same_as<
			  decltype(::std::declval<Matrix const &>().mul(::at::Scalar{2.0F})),
			  Matrix>);
static_assert(::std::same_as<
			  decltype(::std::declval<Matrix const &>().div(::at::Scalar{2.0F})),
			  Matrix>);
static_assert(::std::same_as<
			  decltype(::std::declval<Matrix const &>().softmax(1)),
			  Matrix>);
static_assert(::std::same_as<
			  decltype(::std::declval<Matrix const &>().flatten<>()),
			  MatrixFlat>);
static_assert(::std::same_as<
			  decltype(::std::declval<Matrix const &>().unsqueeze<0>()),
			  MatrixUnsqueezed>);
static_assert(::std::same_as<
			  decltype(::std::declval<Squeezable const &>().squeeze()),
			  Squeezed>);
static_assert(::std::same_as<
			  decltype(::std::declval<Matrix const &>() > ::at::Scalar{2.0F}),
			  BoolMatrix>);
static_assert(::std::same_as<
			  decltype(::std::declval<Matrix const &>().masked_fill(
				  ::std::declval<BoolMatrix const &>(), ::at::Scalar{0.0F})),
			  MatrixAny>);
static_assert(::std::same_as<
			  decltype(Matrix::where(::std::declval<BoolMatrix const &>(),
							 ::std::declval<Matrix const &>(),
							 ::std::declval<Matrix const &>())),
			  MatrixAny>);

auto options() -> ::at::TensorOptions
{
	return ::at::TensorOptions{}.dtype(::at::kFloat).device(::at::kCPU);
}

auto matrix_raw() -> ::at::Tensor
{
	return ::at::arange(6, options()).view({2, 3});
}

void expect_allclose(char const *name, ::at::Tensor const &actual,
					 ::at::Tensor const &expected)
{
	if (!actual.equal(expected))
	{
		throw ::std::runtime_error{
			::std::string{name} + " mismatch; actual=" + actual.toString() +
			", expected=" + expected.toString()};
	}
}

} // namespace

int main()
{
	auto bias_raw{::at::tensor({10.0F, 20.0F, 30.0F}, options())};
	auto column_raw{::at::tensor({1.0F, 2.0F}, options()).view({2, 1})};
	auto twos_raw{::at::ones({2, 3}, options()).mul(2.0F)};

	expect_allclose(
		"sub_tensor_same_shape",
		Matrix::retain(matrix_raw()).sub(Matrix::retain(twos_raw)).unsafe_raw(),
		matrix_raw().sub(twos_raw));
	expect_allclose(
		"sub_tensor_broadcast_alpha",
		Matrix::retain(matrix_raw()).sub(Bias::retain(bias_raw), 0.5F).unsafe_raw(),
		matrix_raw().sub(bias_raw, 0.5F));
	expect_allclose("sub_scalar",
					Matrix::retain(matrix_raw()).sub(::at::Scalar{1.5F}).unsafe_raw(),
					matrix_raw().sub(1.5F));

	expect_allclose(
		"mul_tensor_same_shape",
		Matrix::retain(matrix_raw()).mul(Matrix::retain(twos_raw)).unsafe_raw(),
		matrix_raw().mul(twos_raw));
	expect_allclose(
		"mul_tensor_broadcast",
		Matrix::retain(matrix_raw()).mul(Column::retain(column_raw)).unsafe_raw(),
		matrix_raw().mul(column_raw));
	expect_allclose("mul_scalar",
					Matrix::retain(matrix_raw()).mul(::at::Scalar{3.0F}).unsafe_raw(),
					matrix_raw().mul(3.0F));

	expect_allclose(
		"div_tensor_same_shape",
		Matrix::retain(matrix_raw()).div(Matrix::retain(twos_raw)).unsafe_raw(),
		matrix_raw().div(twos_raw));
	expect_allclose(
		"div_tensor_broadcast",
		Matrix::retain(matrix_raw()).div(Bias::retain(bias_raw)).unsafe_raw(),
		matrix_raw().div(bias_raw));
	expect_allclose("div_scalar",
					Matrix::retain(matrix_raw()).div(::at::Scalar{2.0F}).unsafe_raw(),
					matrix_raw().div(2.0F));

	expect_allclose("softmax_runtime_dim",
					Matrix::retain(matrix_raw()).softmax(1).unsafe_raw(),
					matrix_raw().softmax(1));

	expect_allclose("flatten_static",
					Matrix::retain(matrix_raw()).flatten<>().unsafe_raw(),
					matrix_raw().flatten());
	expect_allclose("unsqueeze_static",
					Matrix::retain(matrix_raw()).unsqueeze<0>().unsafe_raw(),
					matrix_raw().unsqueeze(0));
	expect_allclose(
		"squeeze_static",
		Squeezable::retain(matrix_raw().view({1, 2, 1, 3})).squeeze().unsafe_raw(),
		matrix_raw().view({1, 2, 1, 3}).squeeze());
	auto mask_raw{::at::gt(matrix_raw(), ::at::Scalar{2.0F})};
	expect_allclose("comparison_bool",
					(Matrix::retain(matrix_raw()) > ::at::Scalar{2.0F}).unsafe_raw(),
					mask_raw);
	expect_allclose(
		"masked_fill_static",
		Matrix::retain(matrix_raw())
			.masked_fill(BoolMatrix::retain(mask_raw), ::at::Scalar{-1.0F})
			.unsafe_raw(),
		matrix_raw().masked_fill(mask_raw, -1.0F));
	expect_allclose(
		"where_static",
		Matrix::where(BoolMatrix::retain(mask_raw), Matrix::retain(matrix_raw()),
					  Matrix::retain(twos_raw))
			.unsafe_raw(),
		::at::where(mask_raw, matrix_raw(), twos_raw));

	::fast_io::io::println("tenspec tensor arithmetic tests passed");
}
