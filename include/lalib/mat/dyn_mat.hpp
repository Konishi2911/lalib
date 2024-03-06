#pragma once
#ifndef LALIB_MAT_DYN_MAT_HPP
#define LALIB_MAT_DYN_MAT_HPP

#include "lalib/ops/ops_traits.hpp"
#include <vector>

namespace lalib {

template<typename T>
struct DynMat {
public:
    using ElemType = T;

    // ==== Initializations ==== //

    /// @brief Create a sized matrix with given array with copy.
    constexpr DynMat(const std::vector<T>& arr, size_t n, size_t m) noexcept: 
        _elems(arr), _shape(std::make_pair(n, m)) 
    {}

    /// @brief A copy constructor
    constexpr DynMat(const DynMat<T>& vec) noexcept = default;

    /// @brief Creates a sized matrix with uninitialized elements.
    static constexpr auto uninit(size_t n, size_t m) noexcept -> DynMat<T>;

    /// Creates a sized matrix filled with the given value.
    static constexpr auto filled(T value, size_t n, size_t m) noexcept -> DynMat<T>;

    /// @brief  Creates a diagonal matrix with given elements.
    /// @param value    diagonal elements
    static constexpr auto diag(const std::vector<T>& value) noexcept -> DynMat<T>;

    /// @brief  Creates an uniform diagonal matrix with given value.
    /// @param value    diagonal elements
    /// @param n        size of the matrix
    static constexpr auto diag(T value, size_t n) noexcept -> DynMat<T>;

    /// @brief  Creates a identity matrix
    /// @param n    size of the matrix
    static constexpr auto identity(size_t n) noexcept -> DynMat<T>;


    // === Assignment === //

    /// @brief Replaces the elements of the vector.
    /// @param mat the matrix to use as data source
    /// @return a reference of the matrix after modified by the operation.
    constexpr auto operator=(const DynMat<T>& mat) noexcept -> DynMat<T>& = default;


    // === Inspecting === //

    /// Returns the shape of the matrix (row, column).
    constexpr auto shape() const noexcept -> std::pair<size_t, size_t>;


    // === Accessing, Indexing, and Iterators === //
    using Iter = std::vector<T>::iterator;
    using ConstIter = std::vector<T>::const_iterator;

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
    std::vector<T> _elems;
    std::pair<size_t, size_t> _shape;

    /// @brief  Default constructor
    DynMat() noexcept = default;
};

template <typename T>
inline constexpr auto DynMat<T>::uninit(size_t n, size_t m) noexcept -> DynMat<T>
{
    return DynMat<T>(std::vector<T>(n * m), n, m);
}

template <typename T>
inline constexpr auto DynMat<T>::filled(T value, size_t n, size_t m) noexcept -> DynMat<T>
{
    auto mat = DynMat<T>::uninit(n, m);
    for (auto i = 0u; i < n; ++i) {
        for (auto j = 0u; j < m; ++j) {
            mat(i, j) = value;
        }
    }
    return mat;
}

template <typename T>
inline constexpr auto DynMat<T>::diag(const std::vector<T> &value) noexcept -> DynMat<T>
{
    auto n = value.size();
    auto mat = DynMat<T>::uninit(n, n);
    for (auto i = 0u; i < n; ++i) {
        mat(i, i) = value[i];
    }
    return mat;
}

template <typename T>
inline constexpr auto DynMat<T>::diag(T value, size_t n) noexcept -> DynMat<T>
{
    auto mat = DynMat<T>::uninit(n, n);
    for (auto i = 0u; i < n; ++i) {
        mat(i, i) = value;
    }
    return mat;
}

template <typename T>
inline constexpr auto DynMat<T>::identity(size_t n) noexcept -> DynMat<T>
{
    auto mat = diag(One<T>::value(), n);
    return mat;
}


template <typename T>
inline constexpr auto DynMat<T>::operator()(size_t i, size_t j) const -> const T &
{
    auto m = this->_shape.second;
    auto& v = this->_elems[i * m + j];
    return v;
}

template <typename T>
inline constexpr auto DynMat<T>::operator()(size_t i, size_t j) -> T &
{
    auto m = this->_shape.second;
    auto& v = this->_elems[i * m + j];
    return v;
}

template <typename T>
inline constexpr auto DynMat<T>::data() noexcept -> T *
{
    return this->_elems.data();
}

template <typename T>
inline constexpr auto DynMat<T>::data() const noexcept -> const T *
{
    return this->_elems.data();
}

template <typename T>
inline constexpr auto DynMat<T>::shape() const noexcept -> std::pair<size_t, size_t>
{
    return this->_shape;
}

template <typename T>
inline constexpr auto DynMat<T>::begin() noexcept -> Iter
{
    return this->_elems.begin();
}
template <typename T>
inline constexpr auto DynMat<T>::begin() const noexcept -> ConstIter
{
    return this->_elems.cbegin();
}
template <typename T>
inline constexpr auto DynMat<T>::cbegin() const noexcept -> ConstIter
{
    return this->_elems.cbegin();
}
template <typename T>
inline constexpr auto DynMat<T>::end() noexcept -> Iter
{
    return this->_elems.end();
}
template <typename T>
inline constexpr auto DynMat<T>::end() const noexcept -> ConstIter
{
    return this->_elems.cend();
}
template <typename T>
inline constexpr auto DynMat<T>::cend() const noexcept -> ConstIter
{
    return this->_elems.cend();
}



/*  ############################  *
    Dynamic Tri-diagonal Matrix
 *  ############################  */   

template<typename T>
struct DynTriDiagMat {
public:
    using ElemType = T;

    DynTriDiagMat(const std::vector<T>& dl, const std::vector<T>& d, const std::vector<T>& du) noexcept;
    DynTriDiagMat(std::vector<T>&& dl, std::vector<T>&& d, std::vector<T>&& du) noexcept;

    /// @brief Returns the shape of the tri-diagonal matrix
    /// @return 
    auto shape() const noexcept -> std::pair<size_t, size_t>;

    constexpr auto operator()(size_t i, size_t j) const -> const T&;
    constexpr auto operator()(size_t i, size_t j) -> T&;

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
    std::vector<T> _dl;
    std::vector<T> _d;
    std::vector<T> _du;
};

template <typename T>
inline DynTriDiagMat<T>::DynTriDiagMat(const std::vector<T> &dl, const std::vector<T> &d, const std::vector<T> &du) noexcept:
    _dl(dl), _d(d), _du(du)
{ }

template <typename T>
inline DynTriDiagMat<T>::DynTriDiagMat(std::vector<T> &&dl, std::vector<T> &&d, std::vector<T> &&du) noexcept:
    _dl(std::move(dl)), _d(std::move(d)), _du(std::move(du))
{ }

template <typename T>
inline auto DynTriDiagMat<T>::shape() const noexcept -> std::pair<size_t, size_t>
{
    auto n = this->_d.size();
    return std::pair<size_t, size_t>(n, n);
}

template <typename T>
inline constexpr auto DynTriDiagMat<T>::operator()(size_t i, size_t j) const -> const T &
{
    if (i == j)             return this->_d[i];
    else if ( i == j - 1 )  return this->_du[i];
    else if ( i == j + 1 )  return this->_dl[i - 1];
    else                    return Zero<T>::value();
}
template <typename T>
inline constexpr auto DynTriDiagMat<T>::operator()(size_t i, size_t j) -> T &
{
    if (i == j)             return this->_d[i];
    else if ( i == j - 1 )  return this->_du[i];
    else if ( i == j + 1 )  return this->_dl[i - 1];
    else                    return Zero<T>::value();
}

template <typename T>
inline auto DynTriDiagMat<T>::data_dl() const noexcept -> const T *
{
    return this->_dl.data();
}
template <typename T>
inline auto DynTriDiagMat<T>::data_dl() noexcept -> T *
{
    return this->_dl.data();
}
template <typename T>
inline auto DynTriDiagMat<T>::data_d() const noexcept -> const T *
{
    return this->_d.data();
}

template <typename T>
inline auto DynTriDiagMat<T>::data_d() noexcept -> T *
{
    return this->_d.data();
}
template <typename T>
inline auto DynTriDiagMat<T>::data_du() const noexcept -> const T *
{
    return this->_du.data();
}

template <typename T>
inline auto DynTriDiagMat<T>::data_du() noexcept -> T *
{
    return this->_du.data();
}
}

#endif