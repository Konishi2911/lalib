#pragma once
#include <cstddef>
#include <complex>

#ifdef LALIB_BLAS_BACKEND
#include <cblas.h>
#endif

template<typename T>
inline auto __neg_core_simd(const T* v, T* vr, size_t size) noexcept -> T* {
    #pragma omp simd
    for (auto i = 0u; i < size; ++i) {
        vr[i] = -v[i];
    }
    return vr;
}

template<typename T>
inline auto __neg_core_parallel(const T* v, T* vr, size_t size) noexcept -> T* {
    #pragma omp parallel for simd
    for (auto i = 0u; i < size; ++i) {
        vr[i] = -v[i];
    }
    return vr;
}

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


template<typename T>
inline auto __sub_core_simd(const T* v1, const T* v2, T* vr, size_t size) noexcept -> T* {
    #pragma omp simd
    for (auto i = 0u; i < size; ++i) {
        vr[i] = v1[i] - v2[i];
    }
    return vr;
}

template<typename T>
inline auto __sub_core_parallel(const T* v1, const T* v2, T* vr, size_t size) noexcept -> T* {
    #pragma omp parallel for simd
    for (auto i = 0u; i < size; ++i) {
        vr[i] = v1[i] - v2[i];
    }
    return vr;
}

template<typename T>
inline auto __sub_core_accelerator(const T* v1, const T* v2, T* vr, size_t size) noexcept -> T* {
    #pragma omp target teams distribute parallel for
    for (auto i = 0u; i < size; ++i) {
        vr[i] = v1[i] - v2[i];
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
inline auto sub_core(const T* v1, const T* v2, T* vr, size_t size) noexcept -> T* {
    if (size < 16) {
        __sub_core_simd(v1, v2, vr, size);
    } else {
        #ifdef LALIB_USE_ACCELERATOR
        __sub_core_accelerator(v1, v2, vr, size);
        #else
        __sub_core_parallel(v1, v2, vr, size);
        #endif
    }
    return vr;
}


template<typename T>
inline auto __axpy_core_simd(T alpha, const T* vx, T* vy, size_t size) noexcept -> T* {
    #pragma omp simd
    for (auto i = 0u; i < size; ++i) {
        vy[i] = alpha * vx[i] + vy[i];
    }
    return vy;
}

template<typename T>
inline auto __axpy_core_parallel(T alpha, const T* vx, T* vy, size_t size) noexcept -> T* {
    #pragma omp parallel for simd
    for (auto i = 0u; i < size; ++i) {
        vy[i] = alpha * vx[i] + vy[i];
    }
    return vy;
}

template<typename T>
inline auto __axpy_core_accelerator(T alpha, const T* vx, T* vy, size_t size) noexcept -> T* {
    #pragma omp target teams distribute parallel for
    for (auto i = 0u; i < size; ++i) {
        vy[i] = alpha * vx[i] + vy[i];
    }
    return vy;
}

template<typename T>
inline auto axpy_core(T alpha, const T* vx, T* vy, size_t size) noexcept -> T* {
    __axpy_core_simd(alpha, vx, vy, size);
    return vy;
}

template<>
inline auto axpy_core<float>(float alpha, const float* vx, float* vy, size_t size) noexcept -> float* {
    #ifdef LALIB_BLAS_BACKEND
    cblas_saxpy(size, alpha, vx, 1, vy, 1);
    #else
    __axpy_core_simd(alpha, vx, vy, size);
    #endif
    return vy;
}

template<>
inline auto axpy_core<double>(double alpha, const double* vx, double* vy, size_t size) noexcept -> double* { 
    #ifdef LALIB_BLAS_BACKEND
    cblas_daxpy(size, alpha, vx, 1, vy, 1);
    #else
    __axpy_core_simd(alpha, vx, vy, size);
    #endif
    return vy;
}

template<>
inline auto axpy_core<std::complex<float>>(std::complex<float> alpha, const std::complex<float>* vx, std::complex<float>* vy, size_t size) noexcept -> std::complex<float>* {
    #ifdef LALIB_BLAS_BACKEND
    cblas_caxpy(size, &alpha, vx, 1, vy, 1);
    #else
    __axpy_core_simd(alpha, vx, vy, size);
    #endif
    return vy;
}

template<>
inline auto axpy_core<std::complex<double>>(std::complex<double> alpha, const std::complex<double>* vx, std::complex<double>* vy, size_t size) noexcept -> std::complex<double>* {
    #ifdef LALIB_BLAS_BACKEND
    cblas_zaxpy(size, &alpha, vx, 1, vy, 1);
    #else
    __axpy_core_simd(alpha, vx, vy, size);
    #endif
    return vy;
}


// ==== Dots ==== //

template<typename T>
inline auto __dot_core_simd(const T* v1, const T* v2, size_t size) -> T {
    auto r = 0;
    #pragma omp simd reduction(+:r)
    for (auto i = 0u; i < size; ++i) {
        r += v1[i] * v2[i];
    }
    return r;
}

template<typename T>
inline auto dot_core(const T* v1, const T* v2, size_t size) -> T {
    __dot_core_simd(v1, v2, size);
    return v2;
}

template<>
inline auto dot_core<float>(const float* v1, const float* v2, size_t size) -> float {
    float d;
    #if defined(LALIB_BLAS_BACKEND)
    d = cblas_sdot(size, v1, 1, v2, 1);
    #else
    d = __dot_core_simd(v1, v2, size);
    #endif
    return d;
}

template<>
inline auto dot_core<double>(const double* v1, const double* v2, size_t size) -> double {
    double d;
    #if defined(LALIB_BLAS_BACKEND)
    d = cblas_ddot(size, v1, 1, v2, 1);
    #else
    d = __dot_core_simd(v1, v2, size);
    #endif
    return d;
}


// ==== Scales ==== //

template<typename T>
inline auto __scal_core_simd(T alpha, T* v1, size_t size) -> T* {
    #pragma omp simd
    for (auto i = 0u; i < size; ++i) {
        v1[i] = alpha * v1[i];
    }
    return v1;
}

template<typename T>
inline auto scal_core(T alpha, T* v1, size_t size) -> T* {
    __scal_core_simd(alpha, v1, size);
    return v1;
}

template<>
inline auto scal_core<float>(float alpha, float* v1, size_t size) -> float* {
    #if defined(LALIB_BLAS_BACKEND)
    cblas_sscal(size, alpha, v1, 1);
    #else 
    __scal_core_simd(alpha, v1, size);
    #endif
    return v1;
}

template<>
inline auto scal_core<double>(double alpha, double* v1, size_t size) -> double* {
    #if defined(LALIB_BLAS_BACKEND)
    cblas_dscal(size, alpha, v1, 1);
    #else 
    __scal_core_simd(alpha, v1, size);
    #endif
    return v1;
}


// ==== Eucledian Norm ==== //

template<typename T>
inline auto __norm2_core_simd(const T* v1, size_t size) -> T {
    T r = 0;
    #pragma omp simd reduction(+:r)
    for (auto i = 0u; i < size; ++i) {
        r += v1[i] * v1[i];
    }
    return std::sqrt(r);
}

template<typename T>
inline auto norm2_core(const T* v1, size_t size) -> T {
    auto norm = __norm2_core_simd(v1, size);
    return norm;
}

template<>
inline auto norm2_core<float>(const float* v1, size_t size) -> float {
    float norm = 0;
    #if defined(LALIB_BLAS_BACKEND)    
    norm = cblas_snrm2(size, v1, 1);
    #else 
    norm = __norm2_core_simd(v1, size);
    #endif
    return norm;
}

template<>
inline auto norm2_core<double>(const double* v1, size_t size) -> double {
    double norm = 0;
    #if defined(LALIB_BLAS_BACKEND)    
    norm = cblas_dnrm2(size, v1, 1);
    #else 
    norm = __norm2_core_simd(v1, size);
    #endif
    return norm;
}