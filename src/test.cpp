import std;
import libtorch;
import typetorch;
import fast_io;

#include "torch_macros.inc"

int main()
{
	using Matrix = TYPETORCH_TENSOR((2, 3));

	static_assert(Matrix::shape::rank == 2);
	static_assert(Matrix::dtype == typetorch::DType::F32);

	double const raw_ns{1.25};
	double const typetorch_ns{2.5};
	auto const ratio{typetorch_ns / raw_ns};

	::fast_io::io::println("benchmark result was optimized away");
	::fast_io::io::println(::std::string_view{"sample"}, ", raw_ns=",
						   raw_ns, ", typetorch_ns=", typetorch_ns, ", ratio=", ratio);
}
