#pragma once
#include "../mat.hpp"
#include "../vec.hpp"
#include "../type_traits.hpp"
#include "internal/tdma.hpp"
#include "lapack/gtsv.hpp"
#include <concepts>
#include <vector>
#include <utility>

namespace lalib::solver {

template<TriDiagMatrix M>
requires std::floating_point<typename M::ElemType>
struct TriDiag {
public:
    TriDiag(M& mat) noexcept;
    TriDiag(M&& mat) noexcept;

    template<Vector V>
    auto solve_linear(const V& b, V& rslt) noexcept -> V&;

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
    __lapack::gtsv(n, 1, this->_mat.data_dl(), this->_mat.data_d(), this->_mat.data_du(), rslt.data(), rslt.size());
    #else
    __internal::tdma(n, this->_mat.data_dl(), this->_mat.data_d(), this->_mat.data_du(), b.data(), rslt.data());
    #endif

    return rslt;
}

}