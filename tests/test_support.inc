namespace typetorch_test
{

[[nodiscard]] inline auto f32_cpu_options() -> ::torch::TensorOptions
{
	return ::torch::TensorOptions{}.dtype(::torch::kFloat).device(::torch::kCPU);
}

[[nodiscard]] inline auto i64_cpu_options() -> ::torch::TensorOptions
{
	return ::torch::TensorOptions{}.dtype(::torch::kLong).device(::torch::kCPU);
}

inline void expect_allclose(char const *name, ::torch::Tensor const &actual,
							::torch::Tensor const &expected, double rtol = 1e-5,
							double atol = 1e-8, int exit_code = 1)
{
	if (!::torch::allclose(actual, expected, rtol, atol))
	{
		::fast_io::io::perrln(name, " mismatch; actual=", actual.toString(),
						   ", expected=", expected.toString());
		::std::exit(exit_code);
	}
}

inline void expect_allclose_detailed(char const *name, int exit_code,
									 ::torch::Tensor const &actual,
									 ::torch::Tensor const &expected,
									 double rtol = 1e-4, double atol = 1e-5)
{
	if (!::torch::allclose(actual, expected, rtol, atol))
	{
		auto actual_f32{actual.to(::torch::kFloat).reshape({-1})};
		auto expected_f32{expected.to(::torch::kFloat).reshape({-1})};
		auto diff{actual_f32.sub(expected_f32).abs()};
		auto max_index{diff.argmax().item<::std::int64_t>()};
		::fast_io::io::perrln(
			::std::string_view{name}, " mismatch; actual=", actual.toString(),
			", expected=", expected.toString(),
			", max_abs_diff=", diff.max().item<double>(),
			", max_index=", max_index,
			", actual_value=", actual_f32[max_index].item<double>(),
			", expected_value=", expected_f32[max_index].item<double>());
		::std::exit(exit_code);
	}
}

template <::std::int64_t... Dims>
inline void expect_shape(char const *name, ::torch::Tensor const &tensor)
{
	constexpr ::std::int64_t expected[]{Dims...};
	if (tensor.dim() != static_cast<::std::int64_t>(sizeof...(Dims)))
	{
		::fast_io::io::perrln(name, " rank mismatch");
		::std::exit(1);
	}
	for (::std::size_t i{}; i < sizeof...(Dims); ++i)
	{
		if (tensor.size(static_cast<::std::int64_t>(i)) != expected[i])
		{
			::fast_io::io::perrln(name, " shape mismatch");
			::std::exit(1);
		}
	}
}

} // namespace typetorch_test
