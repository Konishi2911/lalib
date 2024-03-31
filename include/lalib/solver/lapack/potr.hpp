#pragma once
#ifndef LALIB_SOLVER_LAPACK_POTRF_HPP
#define LALIB_SOLVER_LAPACK_POTRF_HPP

#include <cstdint>
#include <lapacke.h>

namespace lalib::solver::_lapack_ {

template<typename T>
auto potrf(int32_t n, T* l, int32_t lda) -> int32_t = delete;

template<typename T>
auto potrs(int32_t n, int32_t nrhs, const T* l, int32_t lda, T* b, int32_t ldb) -> int32_t = delete;


// Spacialization of POTRF

template<>
inline auto potrf<float>(int32_t n, float* l, int32_t lda) -> int32_t {
    auto info = LAPACKE_spotrf(LAPACK_ROW_MAJOR, 'L', n, l, lda);
    return info;
}

template<>
inline auto potrf<double>(int32_t n, double* l, int32_t lda) -> int32_t {
    auto info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n, l, lda);
    return info;
}

template<>
inline auto potrf<std::complex<float>>(int32_t n, std::complex<float>* l, int32_t lda) -> int32_t {
    auto info = LAPACKE_cpotrf(LAPACK_ROW_MAJOR, 'L', n, reinterpret_cast<float __complex__ *>(l), lda);
    return info;
}

template<>
inline auto potrf<std::complex<double>>(int32_t n, std::complex<double>* l, int32_t lda) -> int32_t {
    auto info = LAPACKE_zpotrf(LAPACK_ROW_MAJOR, 'L', n, reinterpret_cast<double __complex__ *>(l), lda);
    return info;
}


// Spacialization of POTRS

template<>
inline auto potrs<float>(int32_t n, int32_t nrhs, const float* l, int32_t lda, float* b, int32_t ldb) -> int32_t {
    auto info = LAPACKE_spotrs(LAPACK_ROW_MAJOR, 'L', n, nrhs, l, lda, b, ldb);
    return info;
}

template<>
inline auto potrs<double>(int32_t n, int32_t nrhs, const double* l, int32_t lda, double* b, int32_t ldb) -> int32_t {
    auto info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', n, nrhs, l, lda, b, ldb);
    return info;
}

template<>
inline auto potrs<std::complex<float>>(int32_t n, int32_t nrhs, const std::complex<float>* l, int32_t lda, std::complex<float>* b, int32_t ldb) -> int32_t {
    auto info = LAPACKE_cpotrs(LAPACK_ROW_MAJOR, 'L', n, nrhs, reinterpret_cast<const float __complex__ *>(l), lda, reinterpret_cast<float __complex__ *>(b), ldb);
    return info;
}

template<>
inline auto potrs<std::complex<double>>(int32_t n, int32_t nrhs, const std::complex<double>* l, int32_t lda, std::complex<double>* b, int32_t ldb) -> int32_t {
    auto info = LAPACKE_zpotrs(LAPACK_ROW_MAJOR, 'L', n, nrhs, reinterpret_cast<const double __complex__ *>(l), lda, reinterpret_cast<double __complex__ *>(b), ldb);
    return info;
}

}

#endif