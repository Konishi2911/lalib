#pragma once 
#ifndef LALIB_VEC_HPP
#define LALIB_VEC_HPP

#include "lalib/vec/sized_vec.hpp"
#include "lalib/vec/dyn_vec.hpp"
#include "lalib/ops/vec_ops.hpp"

namespace lalib {

template<size_t N> using VecF = SizedVec<float, N>;
template<size_t N> using VecD = SizedVec<double, N>;
template<size_t N> using VecC = SizedVec<std::complex<float>, N>;
template<size_t N> using VecZ = SizedVec<std::complex<double>, N>;

using DynVecF = DynVec<float>;
using DynVecD = DynVec<double>;
using DynVecC = DynVec<std::complex<float>>;
using DynVecZ = DynVec<std::complex<double>>;

}

#endif