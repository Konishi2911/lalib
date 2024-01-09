/**
 * @file mat_vec_ops_core.hpp
 * @author Kohei KONISHI
 * @brief   This file defines the BLAS level 2 - equivalent APIs.
 * @note    The API collection treat the matrices as **Column-major**.
 * @version 0.1
 * @date 2024-01-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once
#include <cstddef>
#include <complex>
#include <algorithm>


#ifdef LALIB_BLAS_BACKEND
#include <cblas.h>
#endif

namespace lalib {

template<typename T>
inline auto __mul_core_simd(size_t n, size_t m, T alpha, const T* mat, const T* x, T beta, T* y) noexcept -> T* {
    #pragma omp simd
    for (auto i = 0u; i < n; ++i) {
        y[i] = beta * y[i];
        for (auto k = 0u; k < m; ++k) {
            y[i] += alpha * mat[i * m + k] * x[k];
        }
    }
    return y;
}

template<typename T>
inline auto mul_core(size_t n, size_t m, T alpha, const T* mat, const T* x, T beta, T* y) noexcept -> T* {
    __mul_core_simd(n, m, alpha, mat, x, beta, y);
    return y;
}

template<>
inline auto mul_core<float>(size_t n, size_t m, float alpha, const float* mat, const float* x, float beta, float* y) noexcept -> float* {
    #if defined(LALIB_BLAS_BACKEND)
    cblas_sgemv(CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, n, m, alpha, mat, std::max<size_t>(1u, m), x, 1, beta, y, 1);
    #else
    __mul_core_simd(n, m, alpha, mat, x, beta, y);
    #endif
    return y;
}

template<>
inline auto mul_core<double>(size_t n, size_t m, double alpha, const double* mat, const double* x, double beta, double* y) noexcept -> double* {
    #if defined(LALIB_BLAS_BACKEND)
    cblas_dgemv(CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, n, m, alpha, mat, std::max<size_t>(1u, m), x, 1, beta, y, 1);
    #else
    __mul_core_simd(n, m, alpha, mat, x, beta, y);
    #endif
    return y;
}

}