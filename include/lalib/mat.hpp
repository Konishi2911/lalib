#pragma once
#ifndef LALIB_MAT_HPP
#define LALIB_MAT_HPP

#include "lalib/mat/sized_mat.hpp"
#include "lalib/mat/dyn_mat.hpp"
#include "lalib/ops/mat_ops.hpp"
#include "lalib/ops/mat_vec_ops.hpp"
#include "lalib/ops/mat_mat_ops.hpp"

#include "lalib/mat/sp_mat.hpp"
#include "lalib/ops/sp_mat_ops.hpp"

namespace lalib {

template<size_t N, size_t M> using MatF = SizedMat<float, N, M>;
template<size_t N, size_t M> using MatD = SizedMat<double, N, M>;
template<size_t N, size_t M> using MatC = SizedMat<std::complex<float>, N, M>;
template<size_t N, size_t M> using MatZ = SizedMat<std::complex<double>, N, M>;

using DynMatF = DynMat<float>;
using DynMatD = DynMat<double>;
using DynMatC = DynMat<std::complex<float>>;
using DynMatZ = DynMat<std::complex<double>>;


// Global Non-member Functions

template<typename T>
inline auto invert(const SizedMat<T, 2, 2>& mat, SizedMat<T, 2, 2>& rmat) -> SizedMat<T, 2, 2> {
    if (&mat != &rmat) {
        auto denom = mat(0, 0) * mat(1, 1) - mat(0, 1) * mat(1, 0);
        rmat(0, 0) = mat(1, 1);
        rmat(1, 1) = mat(0, 0);
        rmat(0, 1) = -mat(0, 1);
        rmat(1, 0) = -mat(1, 0);
        scale(1.0 / denom, rmat);
        return rmat;
    } else {
        auto denom = mat(0, 0) * mat(1, 1) - mat(0, 1) * mat(1, 0);
        std::swap(rmat(0, 0), rmat(1, 1));
        rmat(0, 1) *= -1.0;
        rmat(1, 0) *= -1.0;
        scale(1.0 / denom, rmat);
        return rmat;
    }
}

template<typename T>
inline auto invert(SizedMat<T, 2, 2>& mat) -> SizedMat<T, 2, 2>& {
    invert(mat, mat);
    return mat;
}

template<typename T>
inline auto inverted(const SizedMat<T, 2, 2>& mat) -> SizedMat<T, 2, 2> {
    auto rmat = lalib::SizedMat<T, 2, 2>::uninit();
    auto inv = std::move(invert(mat, rmat));
    return inv;
}

}

#endif