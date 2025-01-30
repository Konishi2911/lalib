#pragma once
#include <type_traits>
#ifndef LALIB_MAT_OPS_HPP
#define LALIB_MAT_OPS_HPP

#include "lalib/mat/sized_mat.hpp"
#include "lalib/mat/dyn_mat.hpp"
#include "lalib/mat/sp_mat.hpp"
#include "vec_ops_core.hpp"

namespace lalib {

// =============== //
//  SCALE          //
// =============== //

template<typename T, size_t N, size_t M>
inline auto scale(const T& alpha, SizedMat<T, N, M>& mat) noexcept -> SizedMat<T, N, M>& {
    auto [n, m] = mat.shape();
    scal_core(alpha, mat.data(), n * m);
    return mat;
}

template<typename T>
inline auto scale(const T& alpha, DynMat<T>& mat) noexcept -> DynMat<T>& {
    auto [n, m] = mat.shape();
    scal_core(alpha, mat.data(), n * m);
    return mat;
}

template<typename T>
inline auto scale(const T& alpha, SpMat<T>& mat) noexcept -> SpMat<T>& {
    scal_core<T>(alpha, mat.data(), mat.nnz());
    return mat;
}


template<typename T, size_t N, size_t M>
inline auto operator*(const T& alpha, const SizedMat<T, N, M>& mat) noexcept -> SizedMat<T, N, M> {
    auto rmat = mat;
    scale(alpha, rmat);
    return rmat;
}

template<typename T>
inline auto operator*(const T& alpha, const DynMat<T>& mat) noexcept -> DynMat<T> {
    auto rmat = mat;
    scale(alpha, rmat);
    return rmat;
}

template<typename T>
inline auto operator*(const T& alpha, const SpMat<T>& mat) noexcept -> SpMat<T> {
    auto rmat = mat;
    scale(alpha, rmat);
    return rmat;
}


// =============== //
//  NEGATION       //
// =============== //

template<typename T, size_t N, size_t M>
requires std::is_integral_v<T> || std::is_floating_point_v<T>
inline auto operator-(const SizedMat<T, N, M>& mat) noexcept -> SizedMat<T, N, M> {
    auto rmat = mat;
    scale(-1.0, rmat);
    return rmat;
}

template<typename T>
requires std::is_integral_v<T> || std::is_floating_point_v<T>
inline auto operator-(const DynMat<T>& mat) noexcept -> DynMat<T> {
    auto rmat = mat;
    scale(-1.0, rmat);
    return rmat;
}

template<typename T>
requires std::is_integral_v<T> || std::is_floating_point_v<T>
inline auto operator-(const SpMat<T>& mat) noexcept -> SpMat<T> {
    auto rmat = mat;
    scale(-1.0, rmat);
    return rmat;
}


}

#endif