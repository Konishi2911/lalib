#pragma once
#ifndef LALIB_SOLVER_LAPACK_GTSV_HPP
#define LALIB_SOLVER_LAPACK_GTSV_HPP

#include <cstdint>
#include <cstddef>
#include <complex>

#include <lapacke.h>

namespace lalib::solver::__lapack {
    
/// @brief 
/// @tparam T 
/// @param n 
/// @param nrhs 
/// @param dl 
/// @param d 
/// @param du 
/// @param b 
/// @param ldb  In row major layout, the number of columns, otherwise, the number of rows.
/// @return 
template<typename T>
auto gtsv(int32_t n, int32_t nrhs, T* dl, T* d, T* du, T* b, int32_t ldb) -> int32_t = delete;

template<>
auto gtsv<float>(int32_t n, int32_t nrhs, float* dl, float* d, float* du, float* b, int32_t ldb) -> int32_t {
    auto info = LAPACKE_sgtsv(LAPACK_ROW_MAJOR, n, nrhs, dl, d, du, b, ldb);
    return info;
}

template<>
auto gtsv<double>(int32_t n, int32_t nrhs, double* dl, double* d, double* du, double* b, int32_t ldb) -> int32_t {
    auto info = LAPACKE_dgtsv(LAPACK_ROW_MAJOR, n, nrhs, dl, d, du, b, ldb);
    return info;
}

template<>
auto gtsv<std::complex<float>>(int32_t n, int32_t nrhs, std::complex<float>* dl, std::complex<float>* d, std::complex<float>* du, std::complex<float>* b, int32_t ldb) -> int32_t {
    auto info = LAPACKE_cgtsv(LAPACK_ROW_MAJOR, n, nrhs, 
        reinterpret_cast<float __complex__ *>(dl), 
        reinterpret_cast<float __complex__ *>(d), 
        reinterpret_cast<float __complex__ *>(du), 
        reinterpret_cast<float __complex__ *>(b), 
        ldb
    );
    return info;
}

template<>
auto gtsv<std::complex<double>>(int32_t n, int32_t nrhs, std::complex<double>* dl, std::complex<double>* d, std::complex<double>* du, std::complex<double>* b, int32_t ldb) -> int32_t {
    auto info = LAPACKE_zgtsv(LAPACK_ROW_MAJOR, n, nrhs, 
        reinterpret_cast<double __complex__ *>(dl), 
        reinterpret_cast<double __complex__ *>(d), 
        reinterpret_cast<double __complex__ *>(du), 
        reinterpret_cast<double __complex__ *>(b), 
        ldb
    );
    return info;
}

}

#endif