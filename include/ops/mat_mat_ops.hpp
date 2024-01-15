#pragma once
#ifndef LALIB_MAT_MAT_OPS_HPP
#define LALIB_MAT_MAT_OPS_HPP

#include "mat_mat_ops_core.hpp"
#include "../mat/sized_mat.hpp"
#include "../mat/dyn_mat.hpp"
#include "../vec/sized_vec.hpp"
#include "../vec/dyn_vec.hpp"
#include <cassert>

namespace lalib {

template<typename T, size_t N, size_t M, size_t L>
inline auto mul(T alpha, const SizedMat<T, N, L>& a, const SizedMat<T, L, M>& b, T beta, SizedMat<T, N, M>& c) noexcept -> SizedMat<T, N, M>& {
    mul_core(N, M, L, alpha, a.data(), b.data(), beta, c.data());
    return c;
}

template<typename T, size_t N, size_t M, size_t L>
inline auto mul(T alpha, const DynMat<T>& a, const SizedMat<T, L, M>& b, T beta, SizedMat<T, N, M>& c) noexcept -> SizedMat<T, N, M>& {
    assert(a.shape().first == N);
    assert(a.shape().second == L);
    mul_core(N, M, L, alpha, a.data(), b.data(), beta, c.data());
    return c;
}

template<typename T, size_t N, size_t M, size_t L>
inline auto mul(T alpha, const SizedMat<T, N, L>& a, const DynMat<T>& b, T beta, SizedMat<T, N, M>& c) noexcept -> SizedMat<T, N, M>& {
    assert(b.shape().first == L);
    assert(b.shape().second == M);
    mul_core(N, M, L, alpha, a.data(), b.data(), beta, c.data());
    return c;
}

template<typename T, size_t N, size_t M>
inline auto mul(T alpha, const DynMat<T>& a, const DynMat<T>& b, T beta, SizedMat<T, N, M>& c) noexcept -> SizedMat<T, N, M>& {
    auto l = a.shape().second;
    assert(a.shape().first == N);
    assert(b.shape().second == M);
    assert(a.shape().second == b.shape().first);
    mul_core(N, M, l, alpha, a.data(), b.data(), beta, c.data());
    return c;
}

template<typename T, size_t N, size_t M, size_t L>
inline auto mul(T alpha, const SizedMat<T, N, L>& a, const SizedMat<T, L, M>& b, T beta, DynMat<T>& c) noexcept -> DynMat<T>& {
    assert(c.shape().frist == N);
    assert(c.shape().second == M);
    mul_core(N, M, L, alpha, a.data(), b.data(), beta, c.data());
    return c;
}

template<typename T, size_t M, size_t L>
inline auto mul(T alpha, const DynMat<T>& a, const SizedMat<T, L, M>& b, T beta, DynMat<T>& c) noexcept -> DynMat<T>& {
    auto n = a.shape().first;
    assert(a.shape().first == c.shape().first);
    assert(b.shape().second == c.shape().second);
    assert(a.shape().second == L);
    mul_core(n, M, L, alpha, a.data(), b.data(), beta, c.data());
    return c;
}

template<typename T, size_t N, size_t L>
inline auto mul(T alpha, const SizedMat<T, N, L>& a, const DynMat<T>& b, T beta, DynMat<T>& c) noexcept -> DynMat<T>& {
    auto m = b.shape().second;
    assert(b.shape().first == L);
    assert(a.shape().first == c.shape().first);
    assert(b.shape().second == c.shape().second);
    mul_core(N, m, L, alpha, a.data(), b.data(), beta, c.data());
    return c;
}

template<typename T>
inline auto mul(T alpha, const DynMat<T>& a, const DynMat<T>& b, T beta, DynMat<T>& c) noexcept -> DynMat<T>& {
    auto [n, m] = c.shape();
    auto l = a.shape().second;
    assert(a.shape().first == n);
    assert(b.shape().second == m);
    assert(a.shape().second == b.shape().first);
    mul_core(n, m, l, alpha, a.data(), b.data(), beta, c.data());
    return c;
}


template<typename T, size_t N, size_t M, size_t L>
inline auto operator*(const SizedMat<T, N, L>& a, const SizedMat<T, L, M>& b) noexcept -> SizedMat<T, N, M> {
    auto mr = lalib::SizedMat<T, N, M>::uninit();
    mul(1.0, a, b, 0.0, mr);
    return mr;
}

template<typename T>
inline auto operator*(const DynMat<T>& a, const DynMat<T>& b) noexcept -> DynMat<T> {
    auto mr = lalib::DynMat<T>::uninit(a.shape().first, b.shape().second);
    mul(1.0, a, b, 0.0, mr);
    return mr;
}

}

#endif
