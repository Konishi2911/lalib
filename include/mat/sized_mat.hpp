#pragma once
#include <array>
#include <utility>
#include "../ops/ops_traits.hpp"

namespace lalib {

/// @brief  Column-major matrix type 
/// @tparam T 
/// @tparam N   number of rows
/// @tparam M   number of columns 
template<typename T, size_t N, size_t M>
struct SizedMat {
public:
    // ==== Initializations ==== //

    /// @brief Create a sized matrix with given array with copy.
    constexpr SizedMat(const std::array<T, M * N>& arr) noexcept: 
        _elems(arr) {}

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
    constexpr auto operator=(const SizedMat<T, N, M>& mat) noexcept -> SizedMat<T, N>& = default;


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

    constexpr auto data() noexcept -> T*;
    constexpr auto data() const noexcept -> const T*;

private:
    std::array<T, M * N> _elems;

    /// @brief  Default constructor
    SizedMat() noexcept = default;
};



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
        for (auto j = 0u; i < M; ++j) {
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
    auto v = this->_elems[i * M + j];
    return v;
}

template <typename T, size_t N, size_t M>
inline constexpr auto SizedMat<T, N, M>::operator()(size_t i, size_t j) -> T &
{
    auto v = this->_elems[i * M + j];
    return v;
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
}