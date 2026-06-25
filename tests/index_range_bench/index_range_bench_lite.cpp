#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <utility>

namespace index_range_bench
{

template <::std::size_t N>
consteval inline auto make_index_array()
{
	::std::array<::std::size_t, N> result{};
	for (::std::size_t i{}; i < N; ++i)
	{
		result[i] = i;
	}
	return result;
}

template <::std::size_t N>
consteval inline auto index_range()
{
#if INDEX_RANGE_USE_IOTA
	return ::std::views::iota(::std::size_t{}, N);
#else
	return make_index_array<N>();
#endif
}

template <::std::size_t N, ::std::size_t Salt>
[[gnu::noinline]] auto fold_indices(::std::uint64_t value) noexcept
	-> ::std::uint64_t
{
	template for (constexpr ::std::size_t I : index_range<N>())
	{
		value ^= I + Salt + 0x9e3779b97f4a7c15ULL;
		value = ::std::rotl(value, 7);
		value *= 0xbf58476d1ce4e5b9ULL;
	}
	return value;
}

template <::std::size_t... Is>
[[gnu::noinline]] auto run_suite(
	::std::uint64_t value, ::std::index_sequence<Is...>) noexcept
	-> ::std::uint64_t
{
	((value = fold_indices<(Is % 16) + 1, Is>(value)), ...);
	return value;
}

volatile ::std::uint64_t sink{};

} // namespace index_range_bench

int main()
{
	::std::uint64_t value{0x123456789abcdef0ULL};
	for (::std::size_t iteration{}; iteration < 1'000'000; ++iteration)
	{
		value = index_range_bench::run_suite(
			value + iteration, ::std::make_index_sequence<16>{});
	}
	index_range_bench::sink = value;
}
