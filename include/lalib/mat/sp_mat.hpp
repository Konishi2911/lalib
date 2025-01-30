#ifndef LALIB_MAT_SPARSE_MAT_HPP
#define LALIB_MAT_SPARSE_MAT_HPP

#include "lalib/ops/ops_traits.hpp"
#include <algorithm>
#include <vector>
#include <cassert>
#include <ranges>

namespace lalib {

/// @brief  Sparse matrix in coordinate format
template<typename T>
struct SpCooMat {
    using ElemType = T; 

    // ==== Initializations ==== //

    /// @brief Create an empty sparse matrix object.
    SpCooMat() noexcept = default;

    /// @brief Create a sparse matrix with given data.
    SpCooMat(const std::vector<T>& val, const std::vector<size_t>& row_ids, const std::vector<size_t>& col_ids);

    /// @brief Create a sparse matrix with given data.
    SpCooMat(std::vector<T>&& val, std::vector<size_t>&& row_ids, std::vector<size_t>&& col_ids);

    /// @brief Copy constructor
    SpCooMat(const SpCooMat<T>& mat) noexcept = default;

    /// @brief Move constructor
    SpCooMat(SpCooMat<T>&& mat) noexcept = default;


    // === Inspecting === //

    /// @brief Returns the shape of the matrix (row, column).
    /// @return     a pair of size_t representing the shape of the matrix.
    constexpr auto shape() const noexcept -> std::pair<size_t, size_t>;

    /// @brief Returns the number of non-zero elements in the matrix.
    /// @return     the number of non-zero elements.
    constexpr auto nnz() const noexcept -> size_t;

    /// @brief Returns the value of the element at the given position.
    /// @param i    row index
    /// @param j    column index
    /// @return     the value of the element at the given position.
    auto operator()(size_t i, size_t j) const noexcept -> const T&;


    // === Modifying === //

    /// @brief Reserves memory for the matrix.
    /// @param n    the number of elements to reserve
    void reserve(size_t n) noexcept {
        this->_val.reserve(n);
        this->_row_ids.reserve(n);
        this->_col_ids.reserve(n);
    }


    // === Assignment === //

    /// @brief Replaces the elements of the vector.
    /// @param mat the matrix to use as data source
    /// @return a reference of the matrix after modified by the operation.
    constexpr auto operator=(const SpCooMat<T>& mat) noexcept -> SpCooMat<T>&;

    /// @brief Replaces the elements of the vector.
    /// @param mat the matrix to use as data source
    /// @return a reference of the matrix after modified by the operation.
    constexpr auto operator=(SpCooMat<T>&& mat) noexcept -> SpCooMat<T>&;


    // === Accessing === //
    
    /// @brief Returns a pointer to the array of the values.
    /// @return     a pointer to the array of the values.
    auto values() const noexcept -> const std::vector<T>
        { return this->_val; }

    /// @brief Returns a pointer to the array of the row indices.
    /// @return     a pointer to the array of the row indices.
    auto row_indices() const noexcept -> const std::vector<size_t>
        { return this->_row_ids; }

    /// @brief Returns a pointer to the array of the column indices.
    /// @return     a pointer to the array of the column indices.
    auto col_indices() const noexcept -> const std::vector<size_t>
        { return this->_col_ids; }


private:
    std::vector<T> _val;
    std::vector<size_t> _row_ids;
    std::vector<size_t> _col_ids;

    const T _zero = Zero<T>::value();
};


// === Implementations === //

template<typename T>
SpCooMat<T>::SpCooMat(const std::vector<T>& val, const std::vector<size_t>& row_ids, const std::vector<size_t>& col_ids)
    : _val(val), _row_ids(row_ids), _col_ids(col_ids) 
{
    if (this->_val.size() != this->_row_ids.size() || this->_val.size() != this->_col_ids.size()) {
        throw std::runtime_error("The size of the vectors must be the same.");
    }
}

template<typename T>
SpCooMat<T>::SpCooMat(std::vector<T>&& val, std::vector<size_t>&& row_ids, std::vector<size_t>&& col_ids)
    : _val(std::move(val)), _row_ids(std::move(row_ids)), _col_ids(std::move(col_ids))
{
    if (this->_val.size() != this->_row_ids.size() || this->_val.size() != this->_col_ids.size()) {
        throw std::runtime_error("The size of the vectors must be the same.");
    }
}


template<typename T>
constexpr auto SpCooMat<T>::shape() const noexcept -> std::pair<size_t, size_t> {
    size_t nrow = *std::ranges::max_element(this->_row_ids) + 1;
    size_t ncol = *std::ranges::max_element(this->_col_ids) + 1;

    return std::make_pair(nrow, ncol);
}

template<typename T>
constexpr auto SpCooMat<T>::nnz() const noexcept -> size_t {
    return this->_val.size();
}

template<typename T>
auto SpCooMat<T>::operator()(size_t i, size_t j) const noexcept -> const T& {
    for (auto cnt: std::views::iota(0u, this->_val.size())) {
        if (this->_row_ids[cnt] == i && this->_col_ids[cnt] == j) {
            return this->_val[cnt];
        }
    }
    return this->_zero;
}

template<typename T>
constexpr auto SpCooMat<T>::operator=(const SpCooMat<T>& mat) noexcept -> SpCooMat<T>& {
    this->_val = mat._val;
    this->_row_ids = mat._row_ids;
    this->_col_ids = mat._col_ids;

    return *this;
}

template<typename T>
constexpr auto SpCooMat<T>::operator=(SpCooMat<T>&& mat) noexcept -> SpCooMat<T>& {
    this->_val = std::move(mat._val);
    this->_row_ids = std::move(mat._row_ids);
    this->_col_ids = std::move(mat._col_ids);

    return *this;
}

}
#endif