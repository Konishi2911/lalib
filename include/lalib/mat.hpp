#pragma once
#ifndef LALIB_MAT_HPP
#define LALIB_MAT_HPP

#include "lalib/mat/sized_mat.hpp"
#include "lalib/mat/dyn_mat.hpp"
#include "lalib/ops/mat_vec_ops.hpp"
#include "lalib/ops/mat_mat_ops.hpp"

namespace lalib {

template<size_t N, size_t M> using MatF = SizedMat<float, N, M>;
template<size_t N, size_t M> using MatD = SizedMat<double, N, M>;
template<size_t N, size_t M> using MatC = SizedMat<std::complex<float>, N, M>;
template<size_t N, size_t M> using MatZ = SizedMat<std::complex<double>, N, M>;

using DynMatF = DynMat<float>;
using DynMatD = DynMat<double>;
using DynMatC = DynMat<std::complex<float>>;
using DynMatZ = DynMat<std::complex<double>>;

}

#endif