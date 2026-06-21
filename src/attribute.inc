#ifndef TYPETORCH_ATTR

#define TYPETORCH_ATTR

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

#endif