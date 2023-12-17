#pragma once
#include <cstddef>
#include <complex>

#ifdef LALIB_BLAS_BACKEND
#include <cblas.h>
#endif

template<typename T>
inline auto __add_core_simd(const T* v1, const T* v2, T* vr, size_t size) noexcept -> T* {
    #pragma omp simd
    for (auto i = 0u; i < size; ++i) {
        vr[i] = v1[i] + v2[i];
    }
    return vr;
}

template<typename T>
inline auto __add_core_parallel(const T* v1, const T* v2, T* vr, size_t size) noexcept -> T* {
    #pragma omp parallel for simd
    for (auto i = 0u; i < size; ++i) {
        vr[i] = v1[i] + v2[i];
    }
    return vr;
}

template<typename T>
inline auto __add_core_accelerator(const T* v1, const T* v2, T* vr, size_t size) noexcept -> T* {
    #pragma omp target teams distribute parallel for
    for (auto i = 0u; i < size; ++i) {
        vr[i] = v1[i] + v2[i];
    }
    return vr;
}

/// @brief      Performs addition of two vector for arbitral addable element type `T`.
/// @details    Performs `vr <- v1 + v2`, returns `vr`
/// @tparam T   an addable element type
/// @param v1   a pointer to the head of the first vector 
/// @param v2   a pointer to the head of the second vector
/// @param vr   a pointer to the head of the vector storing the result.
/// @param size number of elements in the vectors. if `n <= 0`, this function returns without any computation.
/// @return     a vector resulted in this operation
template<typename T>
inline auto add_core(const T* v1, const T* v2, T* vr, size_t size) noexcept -> T* {
    if (size < 16) {
        __add_core_simd(v1, v2, vr, size);
    } else {
        #ifdef LALIB_USE_ACCELERATOR
        __add_core_accelerator(v1, v2, vr, size);
        #else
        __add_core_parallel(v1, v2, vr, size);
        #endif
    }
    return vr;
}

template<typename T, size_t N>
inline auto add_core_sized(const T* v1, const T* v2, T* vr) noexcept -> T* {
    __add_core_simd(v1, v2, vr, N);
    return vr;
}