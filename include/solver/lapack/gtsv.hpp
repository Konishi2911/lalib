#pragma once
#include <cstdint>
#include <cstddef>
#include <complex>

namespace lalib::solver::__lapack {
    
extern "C" {
extern void sgtsv_(const int32_t* n, const int32_t* nrow, float* dl, float* d, float* du, float* b, const int32_t* ldb, int32_t* info);
extern void dgtsv_(const int32_t* n, const int32_t* nrow, double* dl, double* d, double* du, double* b, const int32_t* ldb, int32_t* info);
extern void cgtsv_(const int32_t* n, const int32_t* nrow, void* dl, void* d, void* du, void* b, const int32_t* ldb, int32_t* info);
extern void zgtsv_(const int32_t* n, const int32_t* nrow, void* dl, void* d, void* du, void* b, const int32_t* ldb, int32_t* info);
}

template<typename T>
auto gtsv(int32_t n, int32_t nrow, T* dl, T* d, T* du, T* b, int32_t ldb) -> int32_t = delete;

template<>
auto gtsv<float>(int32_t n, int32_t nrow, float* dl, float* d, float* du, float* b, int32_t ldb) -> int32_t {
    int32_t info;
    sgtsv_(&n, &nrow, dl, d, du, b, &ldb, &info);
    return info;
}

template<>
auto gtsv<double>(int32_t n, int32_t nrow, double* dl, double* d, double* du, double* b, int32_t ldb) -> int32_t {
    int32_t info;
    dgtsv_(&n, &nrow, dl, d, du, b, &ldb, &info);
    return info;
}

template<>
auto gtsv<std::complex<float>>(int32_t n, int32_t nrow, std::complex<float>* dl, std::complex<float>* d, std::complex<float>* du, std::complex<float>* b, int32_t ldb) -> int32_t {
    int32_t info;
    cgtsv_(&n, &nrow, dl, d, du, b, &ldb, &info);
    return info;
}

template<>
auto gtsv<std::complex<double>>(int32_t n, int32_t nrow, std::complex<double>* dl, std::complex<double>* d, std::complex<double>* du, std::complex<double>* b, int32_t ldb) -> int32_t {
    int32_t info;
    zgtsv_(&n, &nrow, dl, d, du, b, &ldb, &info);
    return info;
}

}