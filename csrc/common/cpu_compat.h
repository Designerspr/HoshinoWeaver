#pragma once

#include "common/compat.h"

#if defined(_MSC_VER)
#define HNW_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define HNW_RESTRICT __restrict__
#else
#define HNW_RESTRICT
#endif

#ifndef HNW_ENABLE_OMP_SIMD
#define HNW_ENABLE_OMP_SIMD 0
#endif

// MSVC's OpenMP frontend does not recognize any multi-word directive name that
// includes "simd" (neither the combined "parallel for simd" nor the split
// "for simd" — both fail with C3002), even under /openmp:experimental. "parallel
// for" is the only multi-word form it accepts. So on MSVC this macro drops the
// simd hint and falls back to plain thread parallelism; kernel pointers already
// carry HNW_RESTRICT, which lets cl.exe's own /O2 auto-vectorizer do the rest.
// GCC/Clang/ICX get the real combined pragma.
#if defined(_OPENMP) && defined(_MSC_VER)
#define HNW_PRAGMA_OMP_PARALLEL_FOR_SIMD(clause) \
    __pragma(omp parallel for clause)
#elif defined(_OPENMP)
#define HNW_PRAGMA_OMP_STRINGIZE_IMPL(x) #x
#define HNW_PRAGMA_OMP_PARALLEL_FOR_SIMD(clause) \
    _Pragma(HNW_PRAGMA_OMP_STRINGIZE_IMPL(omp parallel for simd clause))
#else
#define HNW_PRAGMA_OMP_PARALLEL_FOR_SIMD(clause)
#endif

// Split-form counterpart for a worksharing loop nested inside an already-open
// `#pragma omp parallel { ... }` block. MSVC rejects "for simd" the same way it
// rejects "parallel for simd" (C3002) — confirmed even with real braces around
// the parallel region — so it drops the simd hint here too.
#if defined(_OPENMP) && defined(_MSC_VER)
#define HNW_PRAGMA_OMP_FOR_SIMD(clause) \
    __pragma(omp for clause)
#elif defined(_OPENMP)
#define HNW_PRAGMA_OMP_FOR_SIMD(clause) \
    _Pragma(HNW_PRAGMA_OMP_STRINGIZE_IMPL(omp for simd clause))
#else
#define HNW_PRAGMA_OMP_FOR_SIMD(clause)
#endif
