#pragma once
#include "../mat.hpp"
#include "../vec.hpp"
#include "../type_traits.hpp"
#include <concepts>
#include <vector>
#include <utility>

#if defined(LALIB_LAPACK_BACKEND)
#include "lapack/gtsv.hpp"
#else 
#include "internal/tdma.hpp"
#endif

namespace lalib::solver {

template<TriDiagMatrix M>
requires std::floating_point<typename M::ElemType>
struct TriDiag {
public:
    TriDiag(M& mat) noexcept;
    TriDiag(M&& mat) noexcept;

    template<Vector V>
    auto solve_linear(const V& b, V& rslt) noexcept -> V&;

    template<Matrix M1>
    auto solve_linear(const M1& b, M1& rslt) noexcept -> M1&;

private:
    M _mat;
};


template <TriDiagMatrix M>
requires std::floating_point<typename M::ElemType>
inline TriDiag<M>::TriDiag(M &mat) noexcept: _mat(mat)
{ }

template <TriDiagMatrix M>
requires std::floating_point<typename M::ElemType>
inline TriDiag<M>::TriDiag(M &&mat) noexcept: _mat(std::move(mat))
{ }

template <TriDiagMatrix M>
requires std::floating_point<typename M::ElemType>
template <Vector V>
inline auto TriDiag<M>::solve_linear(const V &b, V &rslt) noexcept -> V &
{
    assert(this->_mat.shape().first == b.size());
    auto n = this->_mat.shape().first;

    #if defined(LALIB_LAPACK_BACKEND)
    rslt = b;
    __lapack::gtsv(n, 1, this->_mat.data_dl(), this->_mat.data_d(), this->_mat.data_du(), rslt.data(), 1);
    #else
    __internal::tdma(n, 1, this->_mat.data_dl(), this->_mat.data_d(), this->_mat.data_du(), b.data(), rslt.data());
    #endif

    return rslt;
}

template<TriDiagMatrix M>
requires std::floating_point<typename M::ElemType>
template<Matrix M1>
inline auto lalib::solver::TriDiag<M>::solve_linear(const M1 & b, M1 & rslt) noexcept -> M1 &
{
    assert(this->_mat.shape().first == b.shape().first);
    assert(b.shape() == rslt.shape());
    auto n = this->_mat.shape().first;
    auto nrow = b.shape().second;

    #if defined(LALIB_LAPACK_BACKEND)
    rslt = b;
    __lapack::gtsv(n, nrow, this->_mat.data_dl(), this->_mat.data_d(), this->_mat.data_du(), rslt.data(), rslt.shape().second);
    #else
    __internal::tdma(n, nrow, this->_mat.data_dl(), this->_mat.data_d(), this->_mat.data_du(), b.data(), rslt.data());
    #endif

    return rslt;
}

}