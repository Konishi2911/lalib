#pragma once
#include <initializer_list>
#ifndef LALIB_MAT_DYN_MAT_HPP
#define LALIB_MAT_DYN_MAT_HPP

#include "lalib/ops/ops_traits.hpp"
#include <ranges>
#include <vector>
#include <span>

namespace lalib {

template<typename T>
struct DynMat {
public:
    using ElemType = T;

    // ==== Initializations ==== //

    /// @brief Create a sized matrix with given array with copy.
    constexpr DynMat(size_t n, size_t m, const std::vector<T>& arr) noexcept: 
        _elems(arr), _shape(std::make_pair(n, m)) 
    {}

    /// @brief Create a sized matrix with given array with copy.
    [[deprecated]] constexpr DynMat(const std::vector<T>& arr, size_t n, size_t m) noexcept: 
        _elems(arr), _shape(std::make_pair(n, m)) 
    {}

    /// @brief Create a sized matrix with given array with copy.
    constexpr DynMat(size_t n, size_t m, std::initializer_list<T> init) noexcept: 
        _elems(init), _shape(std::make_pair(n, m)) 
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

    constexpr auto at(size_t i, size_t j) const -> const T&;
    constexpr auto mut_at(size_t i, size_t j) -> T&;

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
    return DynMat<T>(n, m, std::vector<T>(n * m));
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

template<typename T>
inline constexpr auto DynMat<T>::at(size_t i, size_t j) const -> const T&
{
    return (*this)(i, j);
}

template<typename T>
inline constexpr auto DynMat<T>::mut_at(size_t i, size_t j) -> T&
{
    return (*this)(i, j);
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
    else                    throw std::invalid_argument("Zero component cannot be modified.");
}

template <typename T>
inline constexpr auto DynTriDiagMat<T>::at(size_t i, size_t j) const -> const T &
{
    return (*this)(i, j);
}

template <typename T>
inline constexpr auto DynTriDiagMat<T>::mut_at(size_t i, size_t j) -> T &
{
    return (*this)(i, j);
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


/*  ############################  *
    Dynamic Hessenberg Matrix
 *  ############################  */

template<typename T>
struct HessenbergMat {
    /// @brief Creates a Hessenberg matrix with given elements.
    HessenbergMat(std::vector<T>&& hess_val);

    /// @brief Creates an empty Hessenberg matrix by allocating memory for the given capacity.
    static constexpr auto with_capacity(size_t cap) noexcept -> HessenbergMat<T>;


    /// @brief Returns the shape of the tri-diagonal matrix
    constexpr auto shape() const noexcept -> std::pair<size_t, size_t>;

    constexpr auto operator()(size_t i, size_t j) const noexcept -> const T&;
    constexpr auto operator()(size_t i, size_t j) -> T&;

    constexpr auto at(size_t i, size_t j) const noexcept -> const T&;
    constexpr auto mut_at(size_t i, size_t j) -> T&;
    
    /// @brief Extend the Hessenberg matrix with zero elements for one dimension.
    void extend_with_zero() noexcept;;

    /// @brief Extend the Hessenberg matrix with new components for one dimension.
    /// @exception std::invalid_argument if the size of the new elements is not equal to the size of the matrix.
    void extend_with(std::vector<T>&& new_elems);


    /// @brief Returns a slice of the compoenents in the specified column.
    auto get_col(size_t j) const noexcept -> std::span<const T>;


    /// @brief Returns a pointer to the array of the elements.
    auto data() noexcept -> T*;

private:
    size_t _n;
    std::vector<T> _h;
    
    const T _zero = Zero<T>::value();

    static auto _calc_size(size_t n) -> size_t {
        if (n == 0) { return 0; }

        auto ac = 17 + 8 * n;
        auto ac_sq = static_cast<size_t>(std::sqrt(ac));

        if (ac != ac_sq * ac_sq) {
            throw std::invalid_argument("Invalid size for Hessenberg matrix.");
        }

        auto s = (ac_sq - 3) / 2;
        return s;
    }
};

template<typename T>
HessenbergMat<T>::HessenbergMat(std::vector<T>&& hess_val):
    _n(_calc_size(hess_val.size())),
    _h(std::move(hess_val))
{}

template<typename T>
constexpr auto HessenbergMat<T>::with_capacity(size_t cap) noexcept -> HessenbergMat<T> {
    auto v = std::vector<T>();
    v.reserve(cap);
    return HessenbergMat<T>(std::move(v));
}

template<typename T>
constexpr auto HessenbergMat<T>::shape() const noexcept -> std::pair<size_t, size_t> {
    return std::make_pair(this->_n, this->_n);
}

template<typename T>
constexpr auto HessenbergMat<T>::operator()(size_t i, size_t j) const noexcept -> const T& {
    if (i <= j + 1) {
        auto offset = ((j + 1) * j) / 2 + j;
        return this->_h[offset + i];
    } else {
        return this->_zero;
    }
}

template<typename T>
constexpr auto HessenbergMat<T>::operator()(size_t i, size_t j) -> T& {
    if (i <= j + 1) {
        auto offset = ((j + 1) * j) / 2 + j;
        return this->_h[offset + i];
    } else {
        throw std::invalid_argument("Zero component cannot be modified.");
    }
}

template<typename T>
constexpr auto HessenbergMat<T>::at(size_t i, size_t j) const noexcept -> const T& {
    return (*this)(i, j);
}

template<typename T>
constexpr auto HessenbergMat<T>::mut_at(size_t i, size_t j) -> T& {
    return (*this)(i, j);
}

template<typename T>
void HessenbergMat<T>::extend_with_zero() noexcept {
    auto new_elems = this->_n == 0 ? 
        std::vector<T>(1, Zero<T>::value()) : 
        std::vector<T>(this->_n + 2, Zero<T>::value());
    this->_h.insert(this->_h.end(), new_elems.begin(), new_elems.end());
    this->_n += 1;
}

template<typename T>
void HessenbergMat<T>::extend_with(std::vector<T>&& new_elems) {
    if (new_elems.size() != this->_n + 2) {
        throw std::invalid_argument("New components must have length n + 2 = " + std::to_string(this->_n + 2));
    }
    this->_h.insert(this->_h.end(), new_elems.begin(), new_elems.end());
    this->_n += 1;
}

template<typename T>
auto HessenbergMat<T>::get_col(size_t j) const noexcept -> std::span<const T> {
    auto offset = ((j + 1) * j) / 2 + j;
    auto end_offset = std::min(offset + j + 2, this->_h.size());
    return std::span<const T>(this->_h.data() + offset, end_offset - offset);
}

template<typename T>
auto HessenbergMat<T>::data() noexcept -> T* {
    return this->_h.data();
}


/*  ################################  *
    Dynamic Upper Triangular Matrix
 *  ################################  */   

template<typename T>
struct DynUpperTriMat {
    /// @brief Default constructor
    DynUpperTriMat() noexcept = default;

    /// @brief Creates a upper triangular matrix with given elements.
    DynUpperTriMat(std::vector<T>&& upper_elems) noexcept;

    /// @brief Creates an empty upper triangular matrix by allocating memory for the given capacity.
    static constexpr auto with_capacity(size_t cap) noexcept -> DynUpperTriMat<T>;


    /// @brief Returns the shape of the tri-diagonal matrix
    constexpr auto shape() const noexcept -> std::pair<size_t, size_t>;

    constexpr auto operator()(size_t i, size_t j) const -> const T&;
    constexpr auto operator()(size_t i, size_t j) -> T&;

    constexpr auto at(size_t i, size_t j) const -> const T&;
    constexpr auto mut_at(size_t i, size_t j) -> T&;


    /// @brief Extends the upper triangular matrix with zero elements for one dimension.
    void extend_with_zero() noexcept;

    /// @brief Extends the upper triangular matrix with new components for one dimension.
    /// @exception std::invalid_argument if the size of the new elements is not equal to the size of the matrix.
    void extend_with(std::vector<T>&& new_elems);


    /// @brief Perfoms backward substitution to solve the linear system.
    /// @tparam V    a vector type
    /// @param y     a rhs vector, and the solution vector after the operation.
    template<std::ranges::random_access_range V>
    void back_sub(V&& y) const;


    /// @brief Returns a pointer to the array of the upper triangular elements.
    auto data() noexcept -> T*;

private:
    size_t _n;
    std::vector<T> _upper_elems;

    const T _zero = Zero<T>::value();

    static auto _calc_size(size_t n) -> size_t {
        auto ac = 1 + 8 * n;
        auto ac_sq = static_cast<size_t>(std::sqrt(ac));

        if (ac != ac_sq * ac_sq) {
            throw std::invalid_argument("Invalid size for Hessenberg matrix.");
        }

        auto s = (ac_sq - 1) / 2;
        return s;
    }
};

template<typename T>
DynUpperTriMat<T>::DynUpperTriMat(std::vector<T>&& upper_elems) noexcept:
    _n(_calc_size(upper_elems.size())),
    _upper_elems(std::move(upper_elems))
{}

template<typename T>
constexpr auto DynUpperTriMat<T>::with_capacity(size_t cap) noexcept -> DynUpperTriMat<T> {
    auto v = std::vector<T>();
    v.reserve(cap);
    return DynUpperTriMat<T>(std::move(v));
}

template<typename T>
constexpr auto DynUpperTriMat<T>::shape() const noexcept -> std::pair<size_t, size_t> {
    return std::make_pair(this->_n, this->_n);
}

template<typename T>
constexpr auto DynUpperTriMat<T>::operator()(size_t i, size_t j) const -> const T& {
    if (i <= j) {
        auto offset = ((j + 1) * j) / 2;
        return this->_upper_elems[offset + i];
    } else {
        return this->_zero;
    }
}

template<typename T>
constexpr auto DynUpperTriMat<T>::operator()(size_t i, size_t j) -> T& {
    if (i <= j) {
        auto offset = ((j + 1) * j) / 2;
        return this->_upper_elems[offset + i];
    } else {
        throw std::invalid_argument("Zero component cannot be modified.");
    }
}

template<typename T>
constexpr auto DynUpperTriMat<T>::at(size_t i, size_t j) const -> const T& {
    return (*this)(i, j);
}

template<typename T>
constexpr auto DynUpperTriMat<T>::mut_at(size_t i, size_t j) -> T& {
    return (*this)(i, j);
}

template<typename T>
void DynUpperTriMat<T>::extend_with_zero() noexcept {
    auto new_elems = std::vector<T>(this->_n + 1, Zero<T>::value());
    this->_upper_elems.insert(this->_upper_elems.end(), new_elems.begin(), new_elems.end());
    this->_n += 1;
}

template<typename T>
void DynUpperTriMat<T>::extend_with(std::vector<T>&& new_elems) {
    if (new_elems.size() != this->_n + 1) {
        throw std::invalid_argument("New components must have length n + 1 = " + std::to_string(this->_n + 1));
    }
    this->_upper_elems.insert(this->_upper_elems.end(), new_elems.begin(), new_elems.end());
    this->_n += 1;
}

template<typename T>
template<std::ranges::random_access_range V>
void DynUpperTriMat<T>::back_sub(V&& y) const {
    if (y.size() != this->_n) {
        throw std::invalid_argument("The size of the vector must be equal to the size of the matrix.");
    }
    
    auto n = this->_n;
    const auto y_iter = y.begin();
    for (int64_t i = n - 1; i >= 0; --i) {
        for (auto j = static_cast<size_t>(i + 1); j < n; ++j) {
            y_iter[i] -= (*this)(i, j) * y_iter[j];
        }
        y_iter[i] /= (*this)(i, i);
    }
}

template<typename T>
auto DynUpperTriMat<T>::data() noexcept -> T* {
    return this->_upper_elems.data();
}


/*  ################################  *
    Dynamic Lower Triangular Matrix
 *  ################################  */   

template<typename T>
struct DynLowerTriMat {
    DynLowerTriMat(size_t n, std::vector<T>&& lower_elems) noexcept;

    /// @brief Returns the shape of the tri-diagonal matrix
    constexpr auto shape() const noexcept -> std::pair<size_t, size_t>;

    constexpr auto operator()(size_t i, size_t j) const -> const T&;
    constexpr auto operator()(size_t i, size_t j) -> T&;

    constexpr auto at(size_t i, size_t j) const -> const T&;
    constexpr auto mut_at(size_t i, size_t j) -> T&;

    auto lower_data() const noexcept -> const T*;
    auto lower_data() noexcept -> T*;

private:
    size_t _n;
    std::vector<T> _lower_elems;
};

template<typename T>
DynLowerTriMat<T>::DynLowerTriMat(size_t n, std::vector<T>&& lower_elems) noexcept:
    _n(n),
    _lower_elems(std::move(lower_elems))
{}

template<typename T>
constexpr auto DynLowerTriMat<T>::shape() const noexcept -> std::pair<size_t, size_t> {
    return std::make_pair(this->_n, this->_n);
}

template<typename T>
constexpr auto DynLowerTriMat<T>::operator()(size_t i, size_t j) const -> const T& {
    if (i < j) { std::swap(i, j); }
    return this->_lower_elems[i * (i + 1) / 2 + j];
}

template<typename T>
constexpr auto DynLowerTriMat<T>::operator()(size_t i, size_t j) -> T& {
    if (i < j) { std::swap(i, j); }
    return this->_lower_elems[i * (i + 1) / 2 + j];
}

template<typename T>
constexpr auto DynLowerTriMat<T>::at(size_t i, size_t j) const -> const T& {
    return (*this)(i, j);
}

template<typename T>
constexpr auto DynLowerTriMat<T>::mut_at(size_t i, size_t j) -> T& {
    return (*this)(i, j);
}

template<typename T>
auto DynLowerTriMat<T>::lower_data() const noexcept -> const T* {
    return this->_lower_elems.data();
}

template<typename T>
auto DynLowerTriMat<T>::lower_data() noexcept -> T* {
    return this->_lower_elems.data();
}


/*  ############################  *
    Dynamic Hermiteian Matrix
 *  ############################  */   

template<typename T>
struct DynHermiteMat {
    DynHermiteMat(size_t n, std::vector<T>&& lower_elems) noexcept;

    /// @brief  Creates a lower triangular matrix by truncating the upper triangular elements.
    /// @warning    Don't use this instance after calling this function because underlaying element vector will be moved.
    auto into_lower_mat() noexcept -> DynLowerTriMat<T>;

    /// @brief Returns the shape of the tri-diagonal matrix
    constexpr auto shape() const noexcept -> std::pair<size_t, size_t>;

    constexpr auto operator()(size_t i, size_t j) const -> const T&;
    constexpr auto operator()(size_t i, size_t j) -> T&;

    constexpr auto at(size_t i, size_t j) const -> const T&;
    constexpr auto mut_at(size_t i, size_t j) -> T&;

    auto lower_data() const noexcept -> const T*;
    auto lower_data() noexcept -> T*;

private:
    size_t _n;
    std::vector<T> _lower_elems;
};

template<typename T>
DynHermiteMat<T>::DynHermiteMat(size_t n, std::vector<T>&& lower_elems) noexcept:
    _n(n),
    _lower_elems(std::move(lower_elems))
{}


template<typename T>
auto DynHermiteMat<T>::into_lower_mat() noexcept -> DynLowerTriMat<T> {
    auto lm = DynLowerTriMat<T>(this->_n, std::move(this->_lower_elems));
    return lm;
}

template<typename T>
constexpr auto DynHermiteMat<T>::shape() const noexcept -> std::pair<size_t, size_t> {
    return std::make_pair(this->_n, this->_n);
}

template<typename T>
constexpr auto DynHermiteMat<T>::operator()(size_t i, size_t j) const -> const T& {
    if (i < j) { std::swap(i, j); }
    return this->_lower_elems[i * (i + 1) / 2 + j];
}

template<typename T>
constexpr auto DynHermiteMat<T>::operator()(size_t i, size_t j) -> T& {
    if (i < j) { std::swap(i, j); }
    return this->_lower_elems[i * (i + 1) / 2 + j];
}

template<typename T>
constexpr auto DynHermiteMat<T>::at(size_t i, size_t j) const -> const T& {
    return (*this)(i, j);
}

template<typename T>
constexpr auto DynHermiteMat<T>::mut_at(size_t i, size_t j) -> T& {
    return (*this)(i, j);
}

template<typename T>
auto DynHermiteMat<T>::lower_data() const noexcept -> const T* {
    return this->_lower_elems.data();
}

template<typename T>
auto DynHermiteMat<T>::lower_data() noexcept -> T* {
    return this->_lower_elems.data();
}


}

#endif