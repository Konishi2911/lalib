#ifndef LALIB_SOLVER_ILU_HPP
#define LALIB_SOLVER_ILU_HPP

#include <memory>
#include <concepts>
#include "lalib/vec.hpp"

namespace lalib::solver {

/// A structure providing some methods for an incomplete LU(0) factorization of a sparse matrix.
/// @tparam T a floating-point type
/// @tparam M a matrix type
template<std::floating_point T, typename M>
struct Ilu {
    /// Constructs an ILU factorization of a matrix.
    /// @param mat a matrix
    /// @throw std::runtime_error if the matrix is singular
    Ilu(M&& mat);

    /// Gets the decomposed matrix.
    /// @return the decomposed matrix
    auto mat() const noexcept -> const M& {
        return this->_mat;
    }

    /// Solves a linear system.
    /// @param rhs a right-hand side vector
    /// @return a solution vector
    auto solve(const lalib::DynVec<T>& rhs) const noexcept -> lalib::DynVec<T>;

private:
    M _mat;
};

namespace internal {
    template<typename M>
    bool _decomp_lu_inplace(M& mat) {
        for (auto i = 0u; i < mat.shape().first; ++i) {
            for (auto j = 0u; j < i; ++j) {
                if (mat(i, j) != 0) {
                    if (mat(j, j) == 0) { return false; }

                    auto sum = mat(i, j);
                    for (auto k = 0u; k < j; ++k) {
                        sum -= mat(i, k) * mat(k, j);
                    }
                    mat.mut_at(i, j) = sum / mat(j, j);
                }
            }

            for (auto j = i; j < mat.shape().second; ++j) {
                if (mat(i, j) != 0) {
                    auto sum = mat(i, j);
                    for (auto k = 0u; k < i; ++k) {
                        sum -= mat(i, k) * mat(k, j);
                    }
                    mat.mut_at(i, j) = sum;
                }
            }
        }

        return true;
    }
}

template<std::floating_point T, typename M>
Ilu<T, M>::Ilu(M&& mat) : _mat(std::forward<M>(mat)) {
    if (!internal::_decomp_lu_inplace(this->_mat)) {
        throw std::runtime_error("Matrix is singular.");
    }
}

template<std::floating_point T, typename M>
auto Ilu<T, M>::solve(const lalib::DynVec<T>& rhs) const noexcept -> lalib::DynVec<T> {
    auto y = lalib::DynVec<T>::uninit(rhs.size());
    for (auto i = 0u; i < y.size(); ++i) {
        auto sum = rhs[i];
        for (auto j = 0u; j < i; ++j) {
            sum -= this->_mat(i, j) * y[j];
        }
        y[i] = sum;
    }

    auto x = lalib::DynVec<T>::uninit(rhs.size());
    for (auto i = x.size(); i-- > 0;) {
        auto sum = y[i];
        for (auto j = i + 1; j < x.size(); ++j) {
            sum -= this->_mat(i, j) * x[j];
        }
        x[i] = sum / this->_mat(i, i);
    }
    return x;
}

}
#endif // LALIB_SOLVER_ILU_HPP