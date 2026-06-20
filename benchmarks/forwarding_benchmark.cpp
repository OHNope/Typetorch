import std;
import libtorch;
import typetorch;
import fast_io;

namespace
{
using Clock = ::std::chrono::steady_clock;
using Ns = ::std::chrono::nanoseconds;

using Matrix = typetorch::Tensor<typetorch::Shape<typetorch::dyn, typetorch::dyn>,
								 typetorch::DType::F32, typetorch::Device::CPU,
								 typetorch::Layout::Any>;
using StaticMatrix = typetorch::Tensor<typetorch::Shape<64, 64>, typetorch::DType::F32,
									   typetorch::Device::CPU,
									   typetorch::Layout::Contiguous>;
using StaticCube = typetorch::Tensor<typetorch::Shape<16, 32, 8>, typetorch::DType::F32,
									 typetorch::Device::CPU,
									 typetorch::Layout::Contiguous>;

struct Sample
{
	char const *name;
	double raw_ns;
	double typetorch_ns;
};

[[nodiscard]] auto read_iterations(int argc, char **argv) -> ::std::int64_t
{
	if (argc < 2)
	{
		return 10000;
	}

	char *end{};
	auto const parsed{::std::strtoll(argv[1], &end, 10)};
	if (end == argv[1] || parsed <= 0)
	{
		return 10000;
	}
	return parsed;
}

[[nodiscard]] auto elapsed_ns(Clock::time_point begin, Clock::time_point end)
	-> double
{
	return static_cast<double>(
		::std::chrono::duration_cast<Ns>(end - begin).count());
}

template <class Fn>
[[nodiscard]] auto run_loop(::std::int64_t iterations, Fn &&fn) -> double
{
	::at::Tensor sink;
	::std::int64_t observed{};

	auto const begin{Clock::now()};
	for (::std::int64_t i{0}; i < iterations; ++i)
	{
		sink = fn();
		observed += sink.numel();
	}
	auto const end{Clock::now()};

	if (observed == 0)
	{
		::fast_io::io::println("benchmark result was optimized away");
		::std::abort();
	}
	return elapsed_ns(begin, end) / static_cast<double>(iterations);
}

template <class RawFn, class TypetorchFn>
[[nodiscard]] auto benchmark(char const *name, ::std::int64_t iterations,
							 RawFn &&raw_fn, TypetorchFn &&typetorch_fn) -> Sample
{
	constexpr ::std::int64_t warmup{128};
	static_cast<void>(run_loop(warmup, raw_fn));
	static_cast<void>(run_loop(warmup, typetorch_fn));

	return Sample{name, run_loop(iterations, raw_fn),
				  run_loop(iterations, typetorch_fn)};
}

void print_sample(Sample const &sample)
{
	auto const ratio{sample.typetorch_ns / sample.raw_ns};
	::fast_io::io::println(::std::string_view{sample.name}, ", raw_ns=",
						   sample.raw_ns, ", typetorch_ns=", sample.typetorch_ns,
						   ", ratio=", ratio);
}

} // namespace

int main(int argc, char **argv)
{
	::at::NoGradGuard no_grad;
	::at::set_num_threads(1);
	::at::set_num_interop_threads(1);

	auto const iterations{read_iterations(argc, argv)};
	auto const options{::at::TensorOptions().dtype(::at::kFloat).device(::at::kCPU)};

	auto const lhs{::at::randn({64, 64}, options)};
	auto const rhs{::at::randn({64, 64}, options)};
	auto const cube{::at::randn({16, 32, 8}, options)};

	auto lhs_dyn{Matrix::unsafe_retain(lhs)};
	auto rhs_dyn{Matrix::unsafe_retain(rhs)};
	auto lhs_static{StaticMatrix::unsafe_retain(lhs)};
	auto rhs_static{StaticMatrix::unsafe_retain(rhs)};

	::std::vector<Sample> samples;
	samples.reserve(6);

	samples.push_back(benchmark(
		"add.dynamic", iterations, [&] { return lhs.add(rhs); },
		[&] { return lhs_dyn.add(rhs_dyn).unwrap(); }));
	samples.push_back(benchmark(
		"add.static", iterations, [&] { return lhs.add(rhs); },
		[&] { return lhs_static.add(rhs_static).unwrap(); }));
	samples.push_back(benchmark(
		"matmul.static", iterations, [&] { return lhs.matmul(rhs); },
		[&] { return lhs_static.matmul(rhs_static).unwrap(); }));
	samples.push_back(benchmark(
		"transpose.static", iterations, [&] { auto x{lhs}; return x.transpose(0, 1); },
		[&] { return StaticMatrix::unsafe_retain(lhs).transpose<0, 1>().unwrap(); }));
	samples.push_back(benchmark(
		"view.static", iterations, [&] { auto x{lhs}; return x.view({4096}); },
		[&] { return StaticMatrix::unsafe_retain(lhs).view<4096>().unwrap(); }));
	samples.push_back(benchmark(
		"permute.static", iterations, [&] { auto x{cube}; return x.permute({0, 2, 1}); },
		[&] { return StaticCube::unsafe_retain(cube).permute<0, 2, 1>().unwrap(); }));

	::fast_io::io::println("iterations=", iterations,
						   ", threads=", ::at::get_num_threads());
	::fast_io::io::println("ratio = typetorch / raw ::at::Tensor; near 1.0 means no measurable forwarding overhead");
	for (auto const &sample : samples)
	{
		print_sample(sample);
	}
}
