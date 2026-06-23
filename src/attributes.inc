#ifndef TYPETORCH_ATTR

#define TYPETORCH_ATTR

#pragma once


// GCC,Clang
#if defined(__GNUC__) || defined(__CLANG__)
#define TYPETORCH_ALWAYS_INLINE __attribute__((always_inline)) inline
#define TYPETORCH_NO_INLINE __attribute__((noinline))
// MSVC
#elif defined(_MSC_VER)
#define TYPETORCH_ALWAYS_INLINE __forceinline
#define TYPETORCH_NO_INLINE __declspec(noinline)
// Default
#else
#define TYPETORCH_ALWAYS_INLINE inline
#define TYPETORCH_NO_INLINE
#endif

#ifdef __CUDACC__
#define TYPETORCH_UNUSED
#else
#define TYPETORCH_UNUSED [[maybe_unused]]
#endif

#if defined(__GNUC__) || defined(__clang__)
#define TYPETORCH_SIZE_NOINLINE __attribute__((noinline, used))
#else
#define TYPETORCH_SIZE_NOINLINE
#endif

#endif