/**
 * @file mat_mat_ops_core.hpp
 * @author Kohei KONISHI 
 * @brief  This file defines the BLAS level 2 - equivalent APIs.
 * @version 0.1
 * @date 2024-01-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once
#ifndef LALIB_MAT_MAT_OPS_CORE_HPP
#define LALIB_MAT_MAT_OPS_CORE_HPP

#include <cstddef>
#include <cstdint>

#ifdef LALIB_BLAS_BACKEND
#include <cblas.h>
#endif

namespace lalib {

template<typename T>
inline auto __mul_core_simd(size_t n, size_t m, size_t o, T alpha, const T* mata, const T* matb, T beta, T* matc) {
    if (mata == matc || matb == matc) {
        auto tmpc = std::make_unique<T[]>(n * m);
        for (auto i = 0u; i < n; ++i) {
            for (auto j = 0u; j < m; ++j) {
                tmpc[i * m + j] = beta * matc[i * m + j];
                for (auto k = 0u; k < o; ++k) {
                    tmpc[i * m + j] += alpha * mata[i * o + k] * matb[k * m + j];
                }
            }
        }
        std::copy(tmpc.get(), tmpc.get() + n * m, matc);
    } else {
        for (auto i = 0u; i < n; ++i) {
            for (auto j = 0u; j < m; ++j) {
                matc[i * m + j] *= beta;
                for (auto k = 0u; k < o; ++k) {
                    matc[i * m + j] += alpha * mata[i * o + k] * matb[k * m + j];
                }
            }
        }
    }
    return matc;
}

template<typename T>
inline auto mul_core(size_t n, size_t m, size_t o, T alpha, const T* mata, const T* matb, T beta, T* matc) noexcept -> T* {
    __mul_core_simd(n, m, o, alpha, mata, matb, beta, matc);
    return matc;
}

template<>
inline auto mul_core<float>(size_t n, size_t m, size_t l, float alpha, const float* mata, const float* matb, float beta, float* matc) noexcept -> float* {
    #if defined(LALIB_BLAS_BACKEND)
    if (mata == matc || matb == matc) {
        auto tmp = std::make_unique<float[]>(n * m);
        std::copy(matc, matc + n * m, tmp.get());
        cblas_sgemm(CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans, n, m, l, alpha, mata, l, matb, m, beta, tmp.get(), n);
        std::copy(tmp.get(), tmp.get() + n * m, matc);
    } else {
        cblas_sgemm(CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans, n, m, l, alpha, mata, l, matb, m, beta, matc, n);
    }
    #else
    __mul_core_simd(n, m, l, alpha, mata, matb, beta, matc);
    #endif
    return matc;
}

template<>
inline auto mul_core<double>(size_t n, size_t m, size_t l, double alpha, const double* mata, const double* matb, double beta, double* matc) noexcept -> double* {
    #if defined(LALIB_BLAS_BACKEND)
    if (mata == matc || matb == matc) {
        auto tmp = std::make_unique<double[]>(n * m);
        std::copy(matc, matc + n * m, tmp.get());
        cblas_dgemm(CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans, n, m, l, alpha, mata, l, matb, m, beta, tmp.get(), n);
        std::copy(tmp.get(), tmp.get() + n * m, matc);
    } else {
        cblas_dgemm(CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans, n, m, l, alpha, mata, l, matb, m, beta, matc, n);
    }
    #else
    __mul_core_simd(n, m, l, alpha, mata, matb, beta, matc);
    #endif
    return matc;
}

}

#endif