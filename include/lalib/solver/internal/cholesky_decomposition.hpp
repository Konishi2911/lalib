#pragma once 
#ifndef LALIB_SOLVER_INTERNAL_CHOLESKY_HPP
#define LALIB_SOLVER_INTERNAL_CHOLESKY_HPP

#include "lalib/mat.hpp"

namespace lalib::solver::_internal_ {

enum class MemType { Square, Triangle };

template<MemType MType>
inline constexpr auto index(size_t n, size_t i, size_t j) noexcept -> size_t;

template<>
inline constexpr auto index<MemType::Square>(size_t n, size_t i, size_t j) noexcept -> size_t {
	return i * n + j;
}

template<>
inline constexpr auto index<MemType::Triangle>(size_t, size_t i, size_t j) noexcept -> size_t {
	return i * (i + 1) / 2 + j;
}


template<typename T, MemType MType>
auto cholesky_decomposition(size_t n, double* l) -> int32_t {
    for (auto i = 0u; i < n; ++i) {
		for (auto j = 0u; j < i; ++j) {
			for (auto k = 0u; k < j; ++k) { 
				l[index<MType>(n, i, j)] -= l[index<MType>(n, i, k)] * l[index<MType>(n, j, k)]; 
			}
			l[index<MType>(n, i, j)] /= l[index<MType>(n, j, j)];
		}

		for (auto k = 0u; k < i; ++k) { 
			l[index<MType>(n, i, i)] -= std::pow(l[index<MType>(n, i, k)], 2); 
		} 

		if (l[index<MType>(n, i, i)] == 0) { 
            return i;
		} 
		else if (l[index<MType>(n, i, i)] < 0) { 
            return i;
		}

		l[index<MType>(n, i, i)] = std::sqrt( l[index<MType>(n, i, i)] );
	}
    return 0;
}

template<typename T, MemType MType>
auto cholesky_linear(size_t n, size_t nrow, const double* l, const double* rhs, double* rslt) -> double* {
	for (auto k = 0u; k < nrow; ++k) {
		// Forward process
		for (auto i = 0u; i < n; ++i) {
			rslt[k + i * nrow] = rhs[k + i * nrow];
			for (auto j = 0u; j < i; ++j) { 
				rslt[k + i * nrow] -= l[index<MType>(n, i, j)] * rslt[k + j * nrow]; 
			} 
			rslt[k + i * nrow] /= l[index<MType>(n, i, i)];
		}

		// Backward process
		for (int32_t i = n - 1; i >= 0; --i) {
			for (auto j = i + 1; j < static_cast<int32_t>(n); ++j) { 
				rslt[k + i * nrow] -= l[index<MType>(n, j, i)] * rslt[k + j * nrow]; 
			}
			rslt[k + i * nrow] /= l[index<MType>(n, i, i)];
		}
	}
	return rslt; 
}


}
#endif