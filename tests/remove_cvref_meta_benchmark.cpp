#include <cstddef>
#include <cstdio>
#include <meta>
#include <type_traits>
#include <utility>

#define NORMALIZE_TYPE(Type) [: ::std::meta::remove_cvref(^^Type) :]
#include "remove_cvref_benchmark.inc"
