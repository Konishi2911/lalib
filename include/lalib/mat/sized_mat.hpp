#pragma once
#include <initializer_list>
#ifndef LALIB_MAT_SIZED_MAT_HPP
#define LALIB_MAT_SIZED_MAT_HPP

#include <array>
#include <vector>
#include <utility>
#include "lalib/ops/ops_traits.hpp"
#include "lalib/type_traits.hpp"

namespace lalib {

/// @brief  Column-major matrix type 
/// @tparam T 
/// @tparam N   number of rows
/// @tparam M   number of columns 
template<typename T, size_t N, size_t M>
struct SizedMat {
public:
    using ElemType = T;
    
    // ==== Initializations ==== //

    /// @brief Create a sized matrix with given array with copy.
    constexpr SizedMat(const std::array<T, M * N>& arr) noexcept: 
        _elems(arr) {}

    /// @brief Create a sized matrix with given vector with copy.
    SizedMat(const std::vector<T>& vec);

    /// @brief Create a sized matrix with given initializer list.
    SizedMat(std::initializer_list<T> init);

    /// @brief A copy constructor
    constexpr SizedMat(const SizedMat<T, N, M>& vec) noexcept = default;

    /// @brief Creates a sized matrix with uninitialized elements.
    static constexpr auto uninit() noexcept -> SizedMat<T, N, M>;

    /// Creates a sized matrix filled with the given value.
    static constexpr auto filled(T value) noexcept -> SizedMat<T, N, M>;

    /// @brief  Creates a diagonal matrix with given elements.
    /// @param value    diagonal elements
    static constexpr auto diag(const std::array<T, N>& value) noexcept -> SizedMat<T, N, M>;

    /// @brief  Creates an uniform diagonal matrix with given value.
    /// @param value    diagonal elements
    static constexpr auto diag(T value) noexcept -> SizedMat<T, N, M>;

    /// @brief  Creates a identity matrix
    static constexpr auto identity() noexcept -> SizedMat<T, N, M>;


    // === Assignment === //

    /// @brief Replaces the elements of the vector.
    /// @param mat the matrix to use as data source
    /// @return a reference of the matrix after modified by the operation.
    constexpr auto operator=(const SizedMat<T, N, M>& mat) noexcept -> SizedMat<T, N, M>& = default;


    // === Inspecting === //

    /// Returns the shape of the matrix (row, column).
    constexpr auto shape() const noexcept -> std::pair<size_t, size_t>;


    // === Accessing, Indexing, and Iterators === //
    using Iter = std::array<T, N>::iterator;
    using ConstIter = std::array<T, N>::const_iterator;

    constexpr auto begin() noexcept -> Iter;
    constexpr auto begin() const noexcept -> ConstIter;
    constexpr auto cbegin() const noexcept -> ConstIter;
    constexpr auto end() noexcept-> Iter;
    constexpr auto end() const noexcept -> ConstIter;
    constexpr auto cend() const noexcept -> ConstIter;

    constexpr auto operator()(size_t i, size_t j) const -> const T&;
    constexpr auto operator()(size_t i, size_t j) -> T&;

    constexpr auto at(size_t i, size_t j) const noexcept -> const T&;
    constexpr auto mut_at(size_t i, size_t j) noexcept -> T&;

    constexpr auto data() noexcept -> T*;
    constexpr auto data() const noexcept -> const T*;

private:
    std::array<T, M * N> _elems;

    /// @brief  Default constructor
    SizedMat() noexcept = default;
};


template<typename T, size_t N, size_t M>
inline SizedMat<T, N, M>::SizedMat(std::initializer_list<T> init):
    _elems()
{
    if (init.size() != N * M) {
        throw std::runtime_error(
            "mismatch size of the given vector. " 
            "expected: " + std::to_string(N * M) + " vec.size(): " + std::to_string(init.size())
        );
    }
    std::copy(init.begin(), init.end(), this->_elems.begin());
}

template<typename T, size_t N, size_t M>
inline SizedMat<T, N, M>::SizedMat(const std::vector<T>& vec):
    _elems()
{
    if (vec.size() != N * M) {
        throw std::runtime_error(
            "mismatch size of the given vector. " 
            "expected: " + std::to_string(N * M) + " vec.size(): " + std::to_string(vec.size())
        );
    }
    std::copy(vec.begin(), vec.end(), this->_elems.begin());
}

template <typename T, size_t N, size_t M>
inline constexpr auto SizedMat<T, N, M>::uninit() noexcept -> SizedMat<T, N, M>
{
    return SizedMat();
}

template <typename T, size_t N, size_t M>
inline constexpr auto SizedMat<T, N, M>::filled(T value) noexcept -> SizedMat<T, N, M>
{
    auto mat = SizedMat<T, N, M>::uninit();
    for (auto i = 0u; i < N; ++i) {
        for (auto j = 0u; j < M; ++j) {
            mat(i, j) = value;
        }
    }
    return mat;
}

template <typename T, size_t N, size_t M>
inline constexpr auto SizedMat<T, N, M>::diag(const std::array<T, N> &value) noexcept -> SizedMat<T, N, M>
{
    static_assert(N == M, "diagonal matrix is only available to a square matrix.");
    auto mat = SizedMat<T, N, M>::uninit();
    for (auto i = 0u; i < N; ++i) {
        mat(i, i) = value[i];
    }
    return mat;
}

template <typename T, size_t N, size_t M>
inline constexpr auto SizedMat<T, N, M>::diag(T value) noexcept -> SizedMat<T, N, M>
{
    static_assert(N == M, "diagonal matrix is only available to a square matrix.");
    auto mat = SizedMat<T, N, M>::uninit();
    for (auto i = 0u; i < N; ++i) {
        mat(i, i) = value;
    }
    return mat;
}

template <typename T, size_t N, size_t M>
inline constexpr auto SizedMat<T, N, M>::identity() noexcept -> SizedMat<T, N, M>
{
    static_assert(N == M, "identity matrix is only available to a square matrix.");
    auto mat = diag(One<T>::value());
    return mat;
}

template <typename T, size_t N, size_t M>
inline constexpr auto SizedMat<T, N, M>::operator()(size_t i, size_t j) const -> const T &
{
    auto& v = this->_elems[i * M + j];
    return v;
}

template <typename T, size_t N, size_t M>
inline constexpr auto SizedMat<T, N, M>::operator()(size_t i, size_t j) -> T &
{
    auto& v = this->_elems[i * M + j];
    return v;
}

template<typename T, size_t N, size_t M>
inline constexpr auto SizedMat<T, N, M>::at(size_t i, size_t j) const noexcept -> const T&
{
    return (*this)(i, j);
}

template<typename T, size_t N, size_t M>
inline constexpr auto SizedMat<T, N, M>::mut_at(size_t i, size_t j) noexcept -> T&
{
    return (*this)(i, j);
}

template <typename T, size_t N, size_t M>
inline constexpr auto SizedMat<T, N, M>::data() noexcept -> T *
{
    return this->_elems.data();
}

template <typename T, size_t N, size_t M>
inline constexpr auto SizedMat<T, N, M>::data() const noexcept -> const T *
{
    return this->_elems.data();
}

template <typename T, size_t N, size_t M>
inline constexpr auto SizedMat<T, N, M>::shape() const noexcept -> std::pair<size_t, size_t>
{
    return std::pair<size_t, size_t>(N, M);
}

template <typename T, size_t N, size_t M>
inline constexpr auto SizedMat<T, N, M>::begin() noexcept -> Iter
{
    return this->_elems.begin();
}
template <typename T, size_t N, size_t M>
inline constexpr auto SizedMat<T, N, M>::begin() const noexcept -> ConstIter
{
    return this->_elems.cbegin();
}
template <typename T, size_t N, size_t M>
inline constexpr auto SizedMat<T, N, M>::cbegin() const noexcept -> ConstIter
{
    return this->_elems.cbegin();
}
template <typename T, size_t N, size_t M>
inline constexpr auto SizedMat<T, N, M>::end() noexcept -> Iter
{
    return this->_elems.end();
}
template <typename T, size_t N, size_t M>
inline constexpr auto SizedMat<T, N, M>::end() const noexcept -> ConstIter
{
    return this->_elems.cend();
}
template <typename T, size_t N, size_t M>
inline constexpr auto SizedMat<T, N, M>::cend() const noexcept -> ConstIter
{
    return this->_elems.cend();
}


/*  ############################  *
    Sized Tri-diagonal Matrix
 *  ############################  */   

template<typename T, size_t N>
struct SizedTriDiagMat {
public:
    using ElemType = T;

    SizedTriDiagMat(const std::array<T, N - 1>& dl, const std::array<T, N>& d, const std::array<T, N - 1>& du) noexcept;

    /// @brief Returns the shape of the tri-diagonal matrix
    /// @return 
    constexpr auto shape() const noexcept -> std::pair<size_t, size_t>;

    constexpr auto operator()(size_t i, size_t j) const -> const T&;
    constexpr auto operator()(size_t i, size_t j) -> T&;

    constexpr auto at(size_t i, size_t j) const -> const T&;
    constexpr auto mut_at(size_t i, size_t j) -> T&;

    /// @brief Returns a pointer to the array of the sub-diagonal elements.
    /// @return 
    auto data_dl() const noexcept -> const T*;
    auto data_dl() noexcept -> T*;

    /// @brief Returns a pointer to the array of the diagonal elements.
    /// @return 
    auto data_d() const noexcept -> const T*;
    auto data_d() noexcept -> T*;

    /// @brief Returns a pointer to the array of the super-diagonal elements.
    /// @return 
    auto data_du() const noexcept -> const T*;
    auto data_du() noexcept -> T*;

private:    
    std::array<T, N - 1>    _dl;
    std::array<T, N>        _d;
    std::array<T, N - 1>    _du;
};

template <typename T, size_t N>
inline SizedTriDiagMat<T, N>::SizedTriDiagMat(const std::array<T, N - 1> &dl, const std::array<T, N> &d, const std::array<T, N - 1> &du) noexcept:
    _dl(dl), _d(d), _du(du)
{ }

template <typename T, size_t N>
inline constexpr auto SizedTriDiagMat<T, N>::shape() const noexcept -> std::pair<size_t, size_t>
{
    return std::pair<size_t, size_t>(N, N);
}

template <typename T, size_t N>
inline constexpr auto SizedTriDiagMat<T, N>::operator()(size_t i, size_t j) const -> const T &
{
    if (i == j)             return this->_d[i];
    else if ( i == j - 1 )  return this->_du[i];
    else if ( i == j + 1 )  return this->_dl[i - 1];
    else                    return Zero<T>::value();
}
template <typename T, size_t N>
inline constexpr auto SizedTriDiagMat<T, N>::operator()(size_t i, size_t j) -> T &
{
    if (i == j)             return this->_d[i];
    else if ( i == j - 1 )  return this->_du[i];
    else if ( i == j + 1 )  return this->_dl[i - 1];
    else                    throw std::out_of_range("Tri-diagonal matrix is zero outside the tri-diagonal.");
}

template<typename T, size_t N>
inline constexpr auto SizedTriDiagMat<T, N>::at(size_t i, size_t j) const -> const T&
{
    return (*this)(i, j);
}

template<typename T, size_t N>
inline constexpr auto SizedTriDiagMat<T, N>::mut_at(size_t i, size_t j) -> T&
{
    return (*this)(i, j);
}

template <typename T, size_t N>
inline auto SizedTriDiagMat<T, N>::data_dl() const noexcept -> const T *
{
    return this->_dl.data();
}
template <typename T, size_t N>
inline auto SizedTriDiagMat<T, N>::data_dl() noexcept -> T *
{
    return this->_dl.data();
}
template <typename T, size_t N>
inline auto SizedTriDiagMat<T, N>::data_d() const noexcept -> const T *
{
    return this->_d.data();
}

template <typename T, size_t N>
inline auto SizedTriDiagMat<T, N>::data_d() noexcept -> T *
{
    return this->_d.data();
}
template <typename T, size_t N>
inline auto SizedTriDiagMat<T, N>::data_du() const noexcept -> const T *
{
    return this->_du.data();
}

template <typename T, size_t N>
inline auto SizedTriDiagMat<T, N>::data_du() noexcept -> T *
{
    return this->_du.data();
}


/*  ##############################  *
    Sized Lower Triangular Matrix
 *  ##############################  */   

template<typename T, size_t N>
struct SizedLowerTriMat {
    SizedLowerTriMat(const std::array<T, N * (N + 1) / 2>& lower_elems) noexcept;


    /// @brief Returns the shape of the tri-diagonal matrix
    constexpr auto shape() const noexcept -> std::pair<size_t, size_t>;

    constexpr auto operator()(size_t i, size_t j) const -> const T&;
    constexpr auto operator()(size_t i, size_t j) -> T&;

    constexpr auto at(size_t i, size_t j) const -> const T&;
    constexpr auto mut_at(size_t i, size_t j) -> T&;

    auto lower_data() const noexcept -> const T*;
    auto lower_data() noexcept -> T*;

private:
    std::array<T, N * (N + 1) / 2> _lower_elems;
};

template<typename T, size_t N>
SizedLowerTriMat<T, N>::SizedLowerTriMat(const std::array<T, N * (N + 1) / 2>& lower_elems) noexcept:
    _lower_elems(lower_elems)
{}


template<typename T, size_t N>
constexpr auto SizedLowerTriMat<T, N>::shape() const noexcept -> std::pair<size_t, size_t> {
    return std::make_pair(N, N);
}

template<typename T, size_t N>
constexpr auto SizedLowerTriMat<T, N>::operator()(size_t i, size_t j) const -> const T& {
    if (i < j) { std::swap(i, j); }
    return this->_lower_elems[i * (i + 1) / 2 + j];
}

template<typename T, size_t N>
constexpr auto SizedLowerTriMat<T, N>::operator()(size_t i, size_t j) -> T& {
    if (i < j) { std::swap(i, j); }
    return this->_lower_elems[i * (i + 1) / 2 + j];
}

template<typename T, size_t N>
constexpr auto SizedLowerTriMat<T, N>::at(size_t i, size_t j) const -> const T& {
    return (*this)(i, j);
}

template<typename T, size_t N>
constexpr auto SizedLowerTriMat<T, N>::mut_at(size_t i, size_t j) -> T& {
    return (*this)(i, j);
}

template<typename T, size_t N>
auto SizedLowerTriMat<T, N>::lower_data() const noexcept -> const T* {
    return this->_lower_elems.data();
}

template<typename T, size_t N>
auto SizedLowerTriMat<T, N>::lower_data() noexcept -> T* {
    return this->_lower_elems.data();
}


/*  ############################  *
    Sized Hermiteian Matrix
 *  ############################  */   

template<typename T, size_t N>
struct SizedHermiteMat {
    SizedHermiteMat(const std::array<T, N * (N + 1) / 2>& lower_elems) noexcept;

    /// @brief  Creates a lower triangular matrix by truncating the upper triangular elements.
    /// @warning    Don't use this instance after calling this function because underlaying element vector will be moved.
    auto into_lower_mat() noexcept -> SizedLowerTriMat<T, N>;

    /// @brief Returns the shape of the tri-diagonal matrix
    constexpr auto shape() const noexcept -> std::pair<size_t, size_t>;

    constexpr auto operator()(size_t i, size_t j) const -> const T&;
    constexpr auto operator()(size_t i, size_t j) -> T&;

    constexpr auto at(size_t i, size_t j) const -> const T&;
    constexpr auto mut_at(size_t i, size_t j) -> T&;

    auto lower_data() const noexcept -> const T*;
    auto lower_data() noexcept -> T*;

private:
    std::array<T, N * (N + 1) / 2> _lower_elems;
};

template<typename T, size_t N>
SizedHermiteMat<T, N>::SizedHermiteMat(const std::array<T, N * (N + 1) / 2>& lower_elems) noexcept:
    _lower_elems(lower_elems)
{}


template<typename T, size_t N>
auto SizedHermiteMat<T, N>::into_lower_mat() noexcept -> SizedLowerTriMat<T, N> {
    auto lm = SizedLowerTriMat<T, N>(std::move(this->_lower_elems));
    return lm;
}

template<typename T, size_t N>
constexpr auto SizedHermiteMat<T, N>::shape() const noexcept -> std::pair<size_t, size_t> {
    return std::make_pair(N, N);
}

template<typename T, size_t N>
constexpr auto SizedHermiteMat<T, N>::operator()(size_t i, size_t j) const -> const T& {
    if (i < j) { std::swap(i, j); }
    return this->_lower_elems[i * (i + 1) / 2 + j];
}

template<typename T, size_t N>
constexpr auto SizedHermiteMat<T, N>::operator()(size_t i, size_t j) -> T& {
    if (i < j) { std::swap(i, j); }
    return this->_lower_elems[i * (i + 1) / 2 + j];
}

template<typename T, size_t N>
constexpr auto SizedHermiteMat<T, N>::at(size_t i, size_t j) const -> const T& {
    return (*this)(i, j);
}

template<typename T, size_t N>
constexpr auto SizedHermiteMat<T, N>::mut_at(size_t i, size_t j) -> T& {
    return (*this)(i, j);
}

template<typename T, size_t N>
auto SizedHermiteMat<T, N>::lower_data() const noexcept -> const T* {
    return this->_lower_elems.data();
}

template<typename T, size_t N>
auto SizedHermiteMat<T, N>::lower_data() noexcept -> T* {
    return this->_lower_elems.data();
}



}

#endif