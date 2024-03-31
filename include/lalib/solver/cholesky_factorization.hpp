#pragma once
#include "lalib/mat/dyn_mat.hpp"
#include "lalib/solver/lapack/potr.hpp"
#ifndef LALIB_SOLVER_CHOLESKY_HPP
#define LALIB_SOLVER_CHOLESKY_HPP

#include "lalib/solver/internal/cholesky_decomposition.hpp"
#include "lalib/type_traits.hpp"
#include "lalib/mat.hpp"

namespace lalib::solver {

template<typename T>
struct DynTriCholeskyFactorization {
    DynTriCholeskyFactorization(lalib::DynHermiteMat<T>&& mat);

	auto lower_matrix() const noexcept -> const lalib::DynLowerTriMat<T>&;

    template<Vector V>
    auto solve_linear(const V& rhs, V& rslt) const -> V&;

    template<Matrix M>
    auto solve_linear(const M& rhs, M& rslt) const -> M&;

private:
    size_t _n;
    lalib::DynLowerTriMat<T> _data;
};

template<typename T>
struct DynCholeskyFactorization {
    DynCholeskyFactorization(lalib::DynMat<T>&& mat);

	auto factor_matrix() const noexcept -> const lalib::DynMat<T>&;

    template<Vector V>
    auto solve_linear(const V& rhs, V& rslt) const -> V&;

    template<Matrix M>
    auto solve_linear(const M& rhs, M& rslt) const -> M&;

private:
    size_t _n;
    lalib::DynMat<T> _data;
};

template<typename T>
inline DynTriCholeskyFactorization<T>::DynTriCholeskyFactorization(lalib::DynHermiteMat<T>&& mat):
    _n(mat.shape().first),
    _data(mat.into_lower_mat())
{ 
    auto rslt = _internal_::cholesky_decomposition<T, _internal_::MemType::Triangle>(this->_n, this->_data.lower_data());

	if (rslt != 0) {
		throw std::runtime_error("[error] fail to decompose the given matrix into LL* matrices (exit with code: " + std::to_string(rslt) + ")");
	}
}


template<typename T>
inline auto DynTriCholeskyFactorization<T>::lower_matrix() const noexcept -> const lalib::DynLowerTriMat<T>& {
	return this->_data;
}


template<typename T>
template<Vector V>
inline auto DynTriCholeskyFactorization<T>::solve_linear(const V& rhs, V& rslt) const -> V& {
   	assert(this->_data.shape().first == rhs.size());

	_internal_::cholesky_linear<double, _internal_::MemType::Triangle>
		(rhs.size(), 1, this->_data.lower_data(), rhs.data(), rslt.data());
	return rslt;
}

template<typename T>
template<Matrix M>
inline auto DynTriCholeskyFactorization<T>::solve_linear(const M& rhs, M& rslt) const -> M& {
   	assert(this->_data.shape().first == rhs.shape().first);
   	assert(rhs.shape().first == rhs.shape().first);
   	assert(rhs.shape().second == rhs.shape().second);

	_internal_::cholesky_linear<double, _internal_::MemType::Triangle>
		(rhs.shape().first, rhs.shape().second, this->_data.lower_data(), rhs.data(), rslt.data());
	return rslt;
}

template<typename T>
inline DynCholeskyFactorization<T>::DynCholeskyFactorization(lalib::DynMat<T>&& mat):
    _n(mat.shape().first),
    _data(mat.into_lower_mat())
{ 
	#if defined LALIB_LAPACK_BACKEND
	auto rslt = _lapack_::potrf(this->_n, this->_data.lower_data(), this->_n);
	#else
    auto rslt = _internal_::cholesky_decomposition<T, _internal_::MemType::Square>(this->_n, this->_data.lower_data());
	#endif

	if (rslt != 0) {
		throw std::runtime_error("[error] fail to decompose the given matrix into LL* matrices (exit with code: " + std::to_string(rslt) + ")");
	}
}


template<typename T>
inline auto DynCholeskyFactorization<T>::factor_matrix() const noexcept -> const lalib::DynMat<T>& {
	return this->_data;
}


template<typename T>
template<Vector V>
inline auto DynCholeskyFactorization<T>::solve_linear(const V& rhs, V& rslt) const -> V& {
   	assert(this->_data.shape().first == rhs.size());

	#if defined LALIB_LAPACK_BACKEND
	rslt = rhs;
	_lapack_::potrs(rhs.size(), 1, this->_data.lower_data(), rhs.size(), rslt.data(), rslt.size());
	#else
	_internal_::cholesky_linear<double, _internal_::MemType::Square>
		(rhs.size(), 1, this->_data.lower_data(), rhs.data(), rslt.data());
	#endif
	return rslt;
}

template<typename T>
template<Matrix M>
inline auto DynCholeskyFactorization<T>::solve_linear(const M& rhs, M& rslt) const -> M& {
   	assert(this->_data.shape().first == rhs.shape().first);
   	assert(rhs.shape().first == rhs.shape().first);
   	assert(rhs.shape().second == rhs.shape().second);

	#if defined LALIB_LAPACK_BACKEND
	rslt = rhs;
	_lapack_::potrs(rhs.shape().first, rhs.shape().second, this->_data.lower_data(), this->_data.shape().first, rslt.data(), rslt.shape().first);
	#else
	_internal_::cholesky_linear<double, _internal_::MemType::Square>
		(rhs.shape().first, rhs.shape().second, this->_data.lower_data(), rhs.data(), rslt.data());
	#endif
	return rslt;
}

}

#endif