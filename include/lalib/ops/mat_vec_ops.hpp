#pragma once
#ifndef LALIB_MAT_VEC_OPS_HPP
#define LALIB_MAT_VEC_OPS_HPP

#include "mat_vec_ops_core.hpp"
#include "lalib/mat/sized_mat.hpp"
#include "lalib/mat/dyn_mat.hpp"
#include "lalib/vec/sized_vec.hpp"
#include "lalib/vec/dyn_vec.hpp"
#include <cassert>


namespace lalib {

template<typename T, size_t N, size_t M>
inline auto mul(T alpha, const SizedMat<T, N, M>& mat, const SizedVec<T, M>& v, T beta, SizedVec<T, N>& vr) noexcept -> SizedVec<T, N>& {
    auto [n, m] = mat.shape();
    mul_core(n, m, alpha, mat.data(), v.data(), beta, vr.data());
    return vr;
}

template<typename T, size_t N, size_t M>
inline auto mul(T alpha, const SizedMat<T, N, M>& mat, const DynVec<T>& v, T beta, SizedVec<T, N>& vr) noexcept -> SizedVec<T, N>& {
    auto [n, m] = mat.shape();
    assert(M == v.size());
    mul_core(n, m, alpha, mat.data(), v.data(), beta, vr.data());
    return vr;
}

template<typename T, size_t N>
inline auto mul(T alpha, const DynMat<T>& mat, const SizedVec<T, N>& v, T beta, DynVec<T>& vr) noexcept -> DynVec<T>& {
    auto [n, m] = mat.shape();
    assert(m == v.size());
    assert(n == vr.size());
    mul_core(n, m, alpha, mat.data(), v.data(), beta, vr.data());
    return vr;
}

template<typename T>
inline auto mul(T alpha, const DynMat<T>& mat, const DynVec<T>& v, T beta, DynVec<T>& vr) noexcept -> DynVec<T>& {
    auto [n, m] = mat.shape();
    assert(m == v.size());
    assert(n == vr.size());
    mul_core(n, m, alpha, mat.data(), v.data(), beta, vr.data());
    return vr;
}


template<typename T, size_t N, size_t M>
inline auto operator*(const SizedMat<T, N, M>& mat, const SizedVec<T, M>& vec) noexcept -> SizedVec<T, N> {
    auto vr = SizedVec<T, N>::uninit();
    mul_core(N, M, 1.0, mat.data(), vec.data(), 0.0, vr.data());
    return vr;
}

template<typename T, size_t N, size_t M>
inline auto operator*(const SizedMat<T, N, M>& mat, const DynVec<T>& vec) noexcept -> SizedVec<T, N> {
    assert(M == vec.size());
    auto vr = SizedVec<T, N>::uninit();
    mul_core(N, M, 1.0, mat.data(), vec.data(), 0.0, vr.data());
    return vr;
}

template<typename T, size_t N>
inline auto operator*(const DynMat<T>& mat, const SizedVec<T, N>& vec) noexcept -> DynVec<T> {
    auto [n, m] = mat.shape();
    assert(m == vec.size());
    auto vr = DynVec<T>::uninit(n);
    mul_core(n, m, 1.0, mat.data(), vec.data(), 0.0, vr.data());
    return vr;
}

template<typename T>
inline auto operator*(const DynMat<T>& mat, const DynVec<T>& vec) noexcept -> DynVec<T> {
    auto [n, m] = mat.shape();
    assert(m == vec.size());
    auto vr = DynVec<T>::uninit(n);
    mul_core(n, m, 1.0, mat.data(), vec.data(), 0.0, vr.data());
    return vr;
}

}

#endif