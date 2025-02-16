#ifndef LALIB_SOLVER_GMRES_HPP
#define LALIB_SOLVER_GMRES_HPP

#include "lalib/mat/dyn_mat.hpp"
#include "lalib/ops/ops_traits.hpp"
#include "lalib/vec/dyn_vec.hpp" 
#include "lalib/ops/vec_ops.hpp"
#include "lalib/ops/mat_vec_ops.hpp"
#include "lalib/solver/ilu.hpp"
#include <ranges>

namespace lalib::solver {

/// @brief      GMRES solver
/// @tparam T   a floating-point type
/// @tparam M   a matrix type
template<typename T, typename M>
struct Gmres {
    /// @brief      Constructs a GMRES solver
    /// @param mat  a matrix
    /// @param tol  a tolerance
    Gmres(M&& mat, T tol): _mat(M(mat)), _lu(std::forward<M>(mat)), _tol(tol) {}

    /// @brief      Solves a linear system
    /// @param rhs  a right-hand side vector
    /// @return     a solution vector
    auto solve(const lalib::DynVec<T>& rhs) const -> lalib::DynVec<T>;

private:
    const M _mat;
    const Ilu<T, M> _lu;
    const T _tol;

    void _arnoldi(std::vector<DynVec<T>>& q, HessenbergMat<T>& hess, T& h) const;
    void _givens_rot(const HessenbergMat<T>& hess, T h, std::vector<T>& s, std::vector<T>& c, DynUpperTriMat<T>& r, std::vector<T>& beta) const;
};


// === Implementation === //

template<typename T, typename M>
auto Gmres<T, M>::solve(const lalib::DynVec<T>& rhs) const -> lalib::DynVec<T> {
    auto n = this->_mat.shape().first;

    // Krylov subspace basis
    auto q = std::vector<lalib::DynVec<T>>();
    q.reserve(n);
    
    // Initial Krylov subspace basis
    auto b_prime = this->_lu.solve(rhs);
    q.emplace_back((1.0 / b_prime.norm2()) * b_prime);

    // Givens rotation components
    auto c = std::vector<T>();
    auto s = std::vector<T>();
    c.reserve(n);
    s.reserve(n);

    // Beta vector ( beta = ||b||| U^T e_1 )
    auto beta = std::vector<T>();
    beta.reserve(n);
    beta.emplace_back(b_prime.norm2());

    auto r = DynUpperTriMat<T>::with_capacity(n);
    auto hess = HessenbergMat<T>::with_capacity(n);
    auto h = Zero<T>::value();

    // Start the GMRES iteration
    for ([[maybe_unused]] auto i: std::views::iota(0u, n)) {
        // Extend the Krylob subspace
        this->_arnoldi(q, hess, h);
        this->_givens_rot(hess, h, s, c, r, beta);

        // Check the convergence
        if (std::abs(beta.back()) < this->_tol) { break; }
    }

    // Solve the uper triangle system
    auto bend = beta.size() - 1;
    r.back_sub(beta | std::views::take(bend));

    // Transform the solution from the Krylov subspace
    auto x = lalib::DynVec<T>::filled(n, Zero<T>::value());
    for (auto i = 0u; i < beta.size() - 1; ++i) {
        axpy(1.0, beta[i] * q[i], x);
    }

    return x;
}


template<typename T, typename M>
void Gmres<T, M>::_arnoldi(std::vector<DynVec<T>>& q, HessenbergMat<T>& hess, T& h) const {
    auto i = q.size() - 1;

    // Extend the Hessenberg matrix
    hess.extend_with_zero();
    if (i > 0) {
        hess(i, i - 1) = h;
    }
    assert(hess.shape().first == i + 1);
    auto rhs = this->_mat * q[i];
    auto v = this->_lu.solve(rhs);
    for (auto j: std::views::iota(0u, i + 1)) {
        hess(j, i) = dot(q[j], v);
    }
    for (auto j: std::views::iota(0u, i + 1)) {
        axpy(-1.0, hess(j, i) * q[j], v);
    }

    h = v.norm2();
    q.emplace_back((1.0 / h) * v);
}

template<typename T, typename M>
void Gmres<T, M>::_givens_rot(const HessenbergMat<T>& hess, T h, std::vector<T>& s, std::vector<T>& c, DynUpperTriMat<T>& r, std::vector<T>& beta) const {
    auto n = hess.shape().first;

    // Extend the upper triangular matrix
    auto col = hess.get_col(n - 1);
    r.extend_with(std::vector(col.begin(), col.end()));
    assert(r.shape().first == n);

    // Extend the beta vector
    beta.emplace_back(Zero<T>::value());
    assert(beta.size() == n+1);

    // Calculate the extended components of the upper triangle matrix
    for (auto i: std::views::iota(0u, n - 1)) {
        auto tmp = c[i] * r(i, n-1) + s[i] * hess(i+1, n-1);
        r(i+1, n-1) = -s[i] * r(i, n-1) + c[i] * hess(i+1, n-1);
        r(i, n-1) = tmp;
    }

    // Calculate the Givens rotation
    auto delta  = std::sqrt(std::pow(r(n-1, n-1), 2) + std::pow(h, 2));
    s.emplace_back(h / delta);
    c.emplace_back(r(n-1, n-1) / delta);
    assert(c.size() == n);
    assert(s.size() == n);

    // Calculate beta = ||b||| U^T e_1
    beta[n] = -s[n-1] * beta[n-1];
    beta[n-1] = c[n-1] * beta[n-1];

    // Apply the Givens rotations to the Hessenberg matrix
    r(n-1, n-1) = c[n-1] * r(n-1, n-1) + s[n-1] * h;
}

}
#endif