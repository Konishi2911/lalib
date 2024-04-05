#pragma once
#ifndef LALIB_SOLVER_CHOLESKY_HPP
#define LALIB_SOLVER_CHOLESKY_HPP

#include "lalib/mat/dyn_mat.hpp"
#include "lalib/solver/internal/cholesky_decomposition.hpp"
#include "lalib/type_traits.hpp"
#include "lalib/mat.hpp"

#if defined(LALIB_LAPACK_BACKEND)
#include "lalib/solver/lapack/potr.hpp"
#endif

namespace lalib::solver {

template<typename T>
struct DynTriCholeskyFactorization {
    DynTriCholeskyFactorization(lalib::DynHermiteMat<T>&& mat);

	auto lower_mat() const noexcept -> const lalib::DynLowerTriMat<T>&;

    template<Vector V>
    auto solve_linear_mut(V& rhs) const -> V&;

    template<Vector V>
    auto solve_linear(const V& rhs) const -> V;

    template<Matrix M>
    auto solve_linear_mut(M& rhs) const -> M&;

    template<Matrix M>
    auto solve_linear(const M& rhs) const -> M;

private:
    size_t _n;
    lalib::DynLowerTriMat<T> _data;
};

template<typename T>
struct DynCholeskyFactorization {
    DynCholeskyFactorization(lalib::DynMat<T>&& mat);

	auto factor_mat() const noexcept -> const lalib::DynMat<T>&;

    template<Vector V>
    auto solve_linear_mut(V& rslt) const -> V&;

    template<Vector V>
    auto solve_linear(const V& rhs) const -> V;

    template<Matrix M>
    auto solve_linear_mut(M& rslt) const -> M&;

    template<Matrix M>
    auto solve_linear(const M& rhs) const -> M;

private:
    size_t _n;
    lalib::DynMat<T> _data;
};


template<typename T>
struct DynModCholeskyFactorization {
	DynModCholeskyFactorization(lalib::DynMat<T>&& mat);

	auto factor_mat() const noexcept -> const lalib::DynMat<T>&;

    template<Vector V>
    auto solve_linear_mut(V& rslt) const -> V&;

    template<Vector V>
    auto solve_linear(const V& rhs) const -> V;

    template<Matrix M>
    auto solve_linear_mut(M& rslt) const -> M&;

    template<Matrix M>
    auto solve_linear(const M& rhs) const -> M;

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
inline auto DynTriCholeskyFactorization<T>::lower_mat() const noexcept -> const lalib::DynLowerTriMat<T>& {
	return this->_data;
}


template<typename T>
template<Vector V>
inline auto DynTriCholeskyFactorization<T>::solve_linear_mut(V& rhs) const -> V& {
   	assert(this->_data.shape().first == rhs.size());

	_internal_::cholesky_linear<double, _internal_::MemType::Triangle>
		(rhs.size(), 1, this->_data.lower_data(), rhs.data(), rhs.data());
	return rhs;
}

template<typename T>
template<Vector V>
inline auto DynTriCholeskyFactorization<T>::solve_linear(const V& rhs) const -> V {
   	assert(this->_data.shape().first == rhs.size());

	auto rslt = rhs;
	_internal_::cholesky_linear<double, _internal_::MemType::Triangle>
		(rhs.size(), 1, this->_data.lower_data(), rhs.data(), rslt.data());
	return rslt;
}

template<typename T>
template<Matrix M>
inline auto DynTriCholeskyFactorization<T>::solve_linear_mut(M& rhs) const -> M& {
   	assert(this->_data.shape().first == rhs.shape().first);
   	assert(rhs.shape().first == rhs.shape().first);
   	assert(rhs.shape().second == rhs.shape().second);

	_internal_::cholesky_linear<double, _internal_::MemType::Triangle>
		(rhs.shape().first, rhs.shape().second, this->_data.lower_data(), rhs.data(), rhs.data());
	return rhs;
}

template<typename T>
template<Matrix M>
inline auto DynTriCholeskyFactorization<T>::solve_linear(const M& rhs) const -> M {
   	assert(this->_data.shape().first == rhs.shape().first);
   	assert(rhs.shape().first == rhs.shape().first);
   	assert(rhs.shape().second == rhs.shape().second);

	auto rslt = rhs;
	_internal_::cholesky_linear<double, _internal_::MemType::Triangle>
		(rhs.shape().first, rhs.shape().second, this->_data.lower_data(), rhs.data(), rslt.data());
	return rslt;
}

template<typename T>
inline DynCholeskyFactorization<T>::DynCholeskyFactorization(lalib::DynMat<T>&& mat):
    _n(mat.shape().first),
    _data(std::move(mat))
{ 
	#if defined LALIB_LAPACK_BACKEND
	auto rslt = _lapack_::potrf(this->_n, this->_data.data(), this->_n);
	#else
    auto rslt = _internal_::cholesky_decomposition<T, _internal_::MemType::Square>(this->_n, this->_data.data());
	#endif

	if (rslt != 0) {
		throw std::runtime_error("[error] fail to decompose the given matrix into LL* matrices (exit with code: " + std::to_string(rslt) + ")");
	}
}


template<typename T>
inline auto DynCholeskyFactorization<T>::factor_mat() const noexcept -> const lalib::DynMat<T>& {
	return this->_data;
}


template<typename T>
template<Vector V>
inline auto DynCholeskyFactorization<T>::solve_linear_mut(V& rhs) const -> V& {
   	assert(this->_data.shape().first == rhs.size());

	#if defined LALIB_LAPACK_BACKEND
	_lapack_::potrs(rhs.size(), 1, this->_data.data(), this->_n, rhs.data(), 1);
	#else
	_internal_::cholesky_linear<double, _internal_::MemType::Square>
		(rhs.size(), 1, this->_data.data(), rhs.data(), rhs.data());
	#endif
	return rhs;
}

template<typename T>
template<Vector V>
inline auto DynCholeskyFactorization<T>::solve_linear(const V& rhs) const -> V {
   	assert(this->_data.shape().first == rhs.size());

	auto rslt = rhs;
	#if defined LALIB_LAPACK_BACKEND
	_lapack_::potrs(rhs.size(), 1, this->_data.data(), this->_n, rslt.data(), 1);
	#else
	_internal_::cholesky_linear<double, _internal_::MemType::Square>
		(rhs.size(), 1, this->_data.data(), rhs.data(), rslt.data());
	#endif
	return rslt;
}

template<typename T>
template<Matrix M>
inline auto DynCholeskyFactorization<T>::solve_linear_mut(M& rhs) const -> M& {
   	assert(this->_data.shape().first == rhs.shape().first);
   	assert(rhs.shape().first == rhs.shape().first);
   	assert(rhs.shape().second == rhs.shape().second);

	#if defined LALIB_LAPACK_BACKEND
	_lapack_::potrs(rhs.shape().first, rhs.shape().second, this->_data.data(), this->_n, rhs.data(), rhs.shape().second);
	#else
	_internal_::cholesky_linear<double, _internal_::MemType::Square>
		(rhs.shape().first, rhs.shape().second, this->_data.data(), rhs.data(), rhs.data());
	#endif
	return rhs;
}

template<typename T>
template<Matrix M>
inline auto DynCholeskyFactorization<T>::solve_linear(const M& rhs) const -> M {
   	assert(this->_data.shape().first == rhs.shape().first);
   	assert(rhs.shape().first == rhs.shape().first);
   	assert(rhs.shape().second == rhs.shape().second);

	auto rslt = rhs;
	#if defined LALIB_LAPACK_BACKEND
	_lapack_::potrs(rhs.shape().first, rhs.shape().second, this->_data.data(), this->_n, rslt.data(), rslt.shape().second);
	#else
	_internal_::cholesky_linear<double, _internal_::MemType::Square>
		(rhs.shape().first, rhs.shape().second, this->_data.data(), rhs.data(), rslt.data());
	#endif
	return rslt;
}


template<typename T>
inline DynModCholeskyFactorization<T>::DynModCholeskyFactorization(lalib::DynMat<T>&& mat):
	_n(mat.shape().first),
	_data(std::move(mat))
{
	_internal_::mod_cholesky_decomposition<T, _internal_::MemType::Square>(this->_n, this->_data.data());
}

template<typename T>
inline auto DynModCholeskyFactorization<T>::factor_mat() const noexcept -> const lalib::DynMat<T>& {
	return this->_data;
}

template<typename T>
template<lalib::Vector V>
inline auto DynModCholeskyFactorization<T>::solve_linear_mut(V& rhs) const -> V& {
   	assert(this->_data.shape().first == rhs.size());

	_internal_::mod_cholesky_linear<T, _internal_::MemType::Square>
		(this->_n, 1, this->_data.data(), rhs.data(), rhs.data());
	return rhs;
}

template<typename T>
template<lalib::Vector V>
inline auto DynModCholeskyFactorization<T>::solve_linear(const V& rhs) const -> V {
   	assert(this->_data.shape().first == rhs.size());

	auto rslt = rhs;
	_internal_::mod_cholesky_linear<T, _internal_::MemType::Square>
		(this->_n, 1, this->_data.data(), rhs.data(), rslt.data());
	return rslt;
}


template<typename T>
template<lalib::Matrix M>
inline auto DynModCholeskyFactorization<T>::solve_linear_mut(M& rhs) const -> M& {
   	assert(this->_data.shape().first == rhs.shape().first);
   	assert(rhs.shape().first == rhs.shape().first);
   	assert(rhs.shape().second == rhs.shape().second);

	_internal_::mod_cholesky_linear<T, _internal_::MemType::Square>
		(this->_n, rhs.shape().second, this->_data.data(), rhs.data(), rhs.data());
	return rhs;
}

template<typename T>
template<lalib::Matrix M>
inline auto DynModCholeskyFactorization<T>::solve_linear(const M& rhs) const -> M {
   	assert(this->_data.shape().first == rhs.shape().first);
   	assert(rhs.shape().first == rhs.shape().first);
   	assert(rhs.shape().second == rhs.shape().second);

	auto rslt = rhs;
	_internal_::mod_cholesky_linear<T, _internal_::MemType::Square>
		(this->_n, rhs.shape().second, this->_data.data(), rhs.data(), rslt.data());
	return rslt;
}

}

#endif