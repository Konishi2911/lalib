#ifndef LALIB_MAT_SPARSE_MAT_HPP
#define LALIB_MAT_SPARSE_MAT_HPP

#include "lalib/ops/ops_traits.hpp"
#include <algorithm>
#include <bits/ranges_algo.h>
#include <cstddef>
#include <vector>
#include <cassert>
#include <ranges>
#include <numeric>
#include <span>

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


    /// @brief Create an unit matrix with given size.
    /// @param n    the size of the matrix
    /// @return     a unit matrix with given size.
    static auto unit(size_t n) -> SpCooMat<T>; 


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

    /// @brief Returns the value of the element at the given position.
    /// @param i    row index
    /// @param j    column index
    /// @return     the value of the element at the given position.
    auto at(size_t i, size_t j) const noexcept -> const T&;

    /// @brief Gets a mutable reference to the element at the given position.
    /// @param i    row index
    /// @param j    column index
    /// @return     a mutable reference to the element at the given position.
    /// @throw      `std::out_of_range` if the index is out of range or does not point to a non-zero element
    auto mut_at(size_t i, size_t j) -> T&;


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
    auto data() noexcept -> T*
        { return this->_val.data(); }
    
    /// @brief Returns a reference to the array of the values.
    /// @return     a pointer to the array of the values.
    auto values() const noexcept -> const std::vector<T>&
        { return this->_val; }

    /// @brief Returns a reference to the array of the row indices.
    /// @return     a pointer to the array of the row indices.
    auto row_indices() const noexcept -> const std::vector<size_t>&
        { return this->_row_ids; }

    /// @brief Returns a reference to the array of the column indices.
    /// @return     a pointer to the array of the column indices.
    auto col_indices() const noexcept -> const std::vector<size_t>&
        { return this->_col_ids; }


    // === Arithmetic Assignment Operators === //

    /// @brief Add and assign the matrix with another matrix.
    /// @param mat the matrix to add
    /// @return a reference of the matrix after modified by the operation.
    auto operator+=(const SpCooMat<T>& mat) -> SpCooMat<T>&;

private:
    std::vector<T> _val;
    std::vector<size_t> _row_ids;
    std::vector<size_t> _col_ids;

    const T _zero = Zero<T>::value();

    void _sort() noexcept;
};


/// @brief Sparse matrix in compressed sparse row format
template<typename T>
struct SpMat {
    using ElemType = T;

    // ==== Initializations ==== //

    /// @brief Create an empty sparse matrix object.
    SpMat() noexcept = default;

    /// @brief Create a sparse matrix from a COO matrix.
    SpMat(const SpCooMat<T>& mat) noexcept;

    /// @brief Create a sparse matrix from a COO matrix.
    SpMat(SpCooMat<T>&& mat) noexcept;

    /// @brief Create a sparse matrix with given data.
    SpMat(const std::vector<T>& val, const std::vector<size_t>& row_ptr, const std::vector<size_t>& col_ids);

    /// @brief Create a sparse matrix with given data.
    SpMat(std::vector<T>&& val, std::vector<size_t>&& row_ptr, std::vector<size_t>&& col_ids);

    /// @brief Copy constructor
    SpMat(const SpMat<T>& mat) noexcept = default;

    /// @brief Move constructor
    SpMat(SpMat<T>&& mat) noexcept = default;

    /// @brief Create an unit matrix with given size.
    /// @param n    the size of the matrix
    /// @return     a unit matrix with given size.
    static auto unit(size_t n) -> SpMat<T>;


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

    /// @brief Returns the value of the element at the given position.
    /// @param i    row index
    /// @param j    column index
    /// @return     the value of the element at the given position.
    auto at(size_t i, size_t j) const noexcept -> const T&;

    /// @brief Gets a mutable reference to the element at the given position.
    /// @param i    row index
    /// @param j    column index
    /// @return     a mutable reference to the element at the given position.
    /// @throw      `std::out_of_range` if the index is out of range or does not point to a non-zero element
    auto mut_at(size_t i, size_t j) -> T&;


    // === Modifying === //

    /// @brief Reserves memory for the matrix.
    /// @param n    the number of rows to reserve
    /// @param nnz  the number of non-zero elements to reserve
    void reserve(size_t n, size_t nnz) noexcept {
        this->_val.reserve(nnz);
        this->_row_ptr.reserve(n + 1);
        this->_col_ids.reserve(nnz);
    }


    // === Assignment === //

    /// @brief Replaces the elements of the vector.
    /// @param mat the matrix to use as data source
    /// @return a reference of the matrix after modified by the operation.
    constexpr auto operator=(const SpMat<T>& mat) noexcept -> SpMat<T>&;

    /// @brief Replaces the elements of the vector.
    /// @param mat the matrix to use as data source
    /// @return a reference of the matrix after modified by the operation.
    constexpr auto operator=(SpMat<T>&& mat) noexcept -> SpMat<T>&;


    // === Accessing === //

    /// @brief Returns a pointer to the array of the values.
    /// @return     a pointer to the array of the values.
    auto data() noexcept -> T*
        { return this->_val.data(); }

    /// @brief Returns a reference to the array of the values.
    /// @return     a pointer to the array of the values.
    auto values() const noexcept -> const std::vector<T>&
        { return this->_val; }
    
    /// @brief Returns a pointer to the array of the row pointers.
    /// @return     a pointer to the array of the row pointers.
    auto row_ptr() const noexcept -> const std::vector<size_t>&
        { return this->_row_ptr; }

    /// @brief Returns a pointer to the array of the column indices.
    /// @return     a pointer to the array of the column indices.
    auto col_indices() const noexcept -> const std::vector<size_t>&
        { return this->_col_ids; }


    // === Arithmetic Assignment Operators === //

    /// @brief Add and assign the matrix with another matrix.
    /// @param mat the matrix to add
    /// @return a reference of the matrix after modified by the operation.
    auto operator+=(const SpMat<T>& mat) -> SpMat<T>&;


private:
    std::vector<T> _val;
    std::vector<size_t> _row_ptr;
    std::vector<size_t> _col_ids;

    const T _zero = Zero<T>::value();

    void _sort() noexcept;
};


// === COO Matrix === //
// === Implementations === //

template<typename T>
SpCooMat<T>::SpCooMat(const std::vector<T>& val, const std::vector<size_t>& row_ids, const std::vector<size_t>& col_ids)
    : _val(val), _row_ids(row_ids), _col_ids(col_ids) 
{
    if (this->_val.size() != this->_row_ids.size() || this->_val.size() != this->_col_ids.size()) {
        throw std::runtime_error("The size of the vectors must be the same.");
    }
    this->_sort();
}

template<typename T>
SpCooMat<T>::SpCooMat(std::vector<T>&& val, std::vector<size_t>&& row_ids, std::vector<size_t>&& col_ids)
    : _val(std::move(val)), _row_ids(std::move(row_ids)), _col_ids(std::move(col_ids))
{
    if (this->_val.size() != this->_row_ids.size() || this->_val.size() != this->_col_ids.size()) {
        throw std::runtime_error("The size of the vectors must be the same.");
    }
    this->_sort();
}

template<typename T>
auto SpCooMat<T>::unit(size_t n) -> SpCooMat<T> {
    std::vector<T> val(n, One<T>::value());
    std::vector<size_t> row_ids(n);
    std::vector<size_t> col_ids(n);

    for (auto i: std::views::iota(0u, n)) {
        row_ids[i] = i;
        col_ids[i] = i;
    }

    return SpCooMat<T>(val, row_ids, col_ids);
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
auto SpCooMat<T>::at(size_t i, size_t j) const noexcept -> const T& {
    return (*this)(i, j);
}

template<typename T>
auto SpCooMat<T>::mut_at(size_t i, size_t j) -> T& {
    for (auto cnt: std::views::iota(0u, this->_val.size())) {
        if (this->_row_ids[cnt] == i && this->_col_ids[cnt] == j) {
            return this->_val[cnt];
        }
    }
    throw std::out_of_range("Index out of range or does not point to a non-zero element.");
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

template<typename T>
auto SpCooMat<T>::operator+=(const SpCooMat<T>& mat) -> SpCooMat<T>& {
    this->_val.reserve(this->_val.size() + mat.nnz());
    this->_row_ids.reserve(this->_row_ids.size() + mat.nnz());
    this->_col_ids.reserve(this->_col_ids.size() + mat.nnz());

    auto nrow_ids = this->_row_ids.size();
    auto nrrow_ids = mat._row_ids.size();
    auto row_iter = this->_row_ids.begin();
    auto rrow_iter = mat._row_ids.begin();
    for (auto i = 0u; i < mat.shape().first; ++i) {
        auto row_iter_end = std::upper_bound(row_iter, this->_row_ids.begin() + nrow_ids, i, std::less<size_t>{});
        auto rrow_iter_end = std::upper_bound(rrow_iter, mat._row_ids.begin() + nrrow_ids, i, std::less<size_t>{});

        auto cursor = std::distance(this->_row_ids.begin(), row_iter);
        auto rcursor = std::distance(mat._row_ids.begin(), rrow_iter);
        auto cursor_end = std::distance(this->_row_ids.begin(), row_iter_end) - 1;
        auto rcursor_end = std::distance(mat._row_ids.begin(), rrow_iter_end) - 1;

        while (true) {
            if (this->_col_ids[cursor] == mat._col_ids[rcursor]) {
                this->_val[cursor] += mat._val[rcursor];

                if (rcursor == rcursor_end) { break; }
                else {
                    if (cursor < cursor_end) { ++cursor; }
                    if (rcursor < rcursor_end) { ++rcursor; }
                }
            }
            else if (this->_col_ids[cursor] < mat._col_ids[rcursor]) {
                if (cursor < cursor_end) { ++cursor; }
                else {
                    this->_val.emplace_back(mat._val[rcursor]);
                    this->_row_ids.emplace_back(i);
                    this->_col_ids.emplace_back(mat._col_ids[rcursor]);

                    if (rcursor == rcursor_end) { break; }
                    else { ++rcursor; }
                }
            }
            else {
                this->_val.emplace_back(mat._val[rcursor]);
                this->_row_ids.emplace_back(i);
                this->_col_ids.emplace_back(mat._col_ids[rcursor]);

                if (rcursor == rcursor_end) { break; }
                else { 
                    if (rcursor < rcursor_end) { ++rcursor; }
                }
            }
        } 

        row_iter = row_iter_end;
        rrow_iter = rrow_iter_end;
    }

    this->_sort();
    return *this;
}


template<typename T>
void SpCooMat<T>::_sort() noexcept {
    std::vector<size_t> ids(this->_val.size());
    std::iota(ids.begin(), ids.end(), 0u);

    std::ranges::sort(ids, [&] (size_t i, size_t j) {
        if (this->_row_ids[i] == this->_row_ids[j]) {
            return this->_col_ids[i] < this->_col_ids[j];
        }
        return this->_row_ids[i] < this->_row_ids[j];
    });

    auto new_val = std::vector<T>(this->_val.size());
    auto new_row_ids = std::vector<size_t>(this->_val.size());
    auto new_col_ids = std::vector<size_t>(this->_val.size());
    for (auto i = 0u; i < this->_val.size(); ++i) {
        new_val[i] = this->_val[ids[i]];
        new_row_ids[i] = this->_row_ids[ids[i]];
        new_col_ids[i] = this->_col_ids[ids[i]];
    }

    this->_val = std::move(new_val);
    this->_row_ids = std::move(new_row_ids);
    this->_col_ids = std::move(new_col_ids);
}


// === CSR Matrix === //
// === Implementations === //

inline void _convert_coo_crs(std::vector<size_t>& row_ptr, const std::vector<size_t>& row_ids) {
    size_t nnz = row_ids.size();

    row_ptr.reserve(nnz);
    if (!row_ptr.empty()) {
        row_ptr.clear();
    }

    row_ptr.emplace_back(0);

    auto it = row_ids.begin();
    for (size_t i = 0; it < row_ids.end(); ++i) {
        it = std::upper_bound(it, row_ids.end(), i, std::less<size_t>{});
        row_ptr.emplace_back(std::distance(row_ids.begin(), it));
    }
}

template<typename T>
SpMat<T>::SpMat(const SpCooMat<T>& mat) noexcept
    : _val(mat.values()), _row_ptr(), _col_ids(mat.col_indices())
{
    _convert_coo_crs(this->_row_ptr, mat.row_indices());
}

template<typename T>
SpMat<T>::SpMat(SpCooMat<T>&& mat) noexcept
    : _val(std::move(mat.values())), _row_ptr(), _col_ids(std::move(mat.col_indices()))
{
    _convert_coo_crs(this->_row_ptr, mat.row_indices());
}

template<typename T>
SpMat<T>::SpMat(const std::vector<T>& val, const std::vector<size_t>& row_ptr, const std::vector<size_t>& col_ids)
    : _val(val), _row_ptr(row_ptr), _col_ids(col_ids) 
{
    if (this->_val.size() != this->_col_ids.size()) {
        throw std::runtime_error("The size of the vectors must be the same.");
    }
}

template<typename T>
SpMat<T>::SpMat(std::vector<T>&& val, std::vector<size_t>&& row_ptr, std::vector<size_t>&& col_ids)
    : _val(std::move(val)), _row_ptr(std::move(row_ptr)), _col_ids(std::move(col_ids))
{
    if (this->_val.size() != this->_col_ids.size()) {
        throw std::runtime_error("The size of the vectors must be the same.");
    }
}

template<typename T>
auto SpMat<T>::unit(size_t n) -> SpMat<T> {
    std::vector<T> val(n, One<T>::value());
    std::vector<size_t> row_ptr(n + 1);
    std::vector<size_t> col_ids(n);

    for (auto i: std::views::iota(0u, n)) {
        row_ptr[i] = i;
        col_ids[i] = i;
    }
    row_ptr[n] = n;

    return SpMat<T>(val, row_ptr, col_ids);
}


template<typename T>
constexpr auto SpMat<T>::shape() const noexcept -> std::pair<size_t, size_t> {
    size_t nrow = this->_row_ptr.size() - 1;
    size_t ncol = *std::ranges::max_element(this->_col_ids) + 1;

    return std::make_pair(nrow, ncol);
}

template<typename T>
constexpr auto SpMat<T>::nnz() const noexcept -> size_t {
    return this->_val.size();
}

template<typename T>
auto SpMat<T>::operator()(size_t i, size_t j) const noexcept -> const T& {
    for (auto cnt: std::views::iota(this->_row_ptr[i], this->_row_ptr[i + 1])) {
        if (this->_col_ids[cnt] == j) {
            return this->_val[cnt];
        }
    }
    return this->_zero;
}

template<typename T>
auto SpMat<T>::at(size_t i, size_t j) const noexcept -> const T& {
    return (*this)(i, j);
}

template<typename T>
auto SpMat<T>::mut_at(size_t i, size_t j) -> T& {
    for (auto cnt: std::views::iota(this->_row_ptr[i], this->_row_ptr[i + 1])) {
        if (this->_col_ids[cnt] == j) {
            return this->_val[cnt];
        }
    }
    throw std::out_of_range("Index out of range or does not point to a non-zero element.");
}

template<typename T>
constexpr auto SpMat<T>::operator=(const SpMat<T>& mat) noexcept -> SpMat<T>& {
    this->_val = mat._val;
    this->_row_ptr = mat._row_ptr;
    this->_col_ids = mat._col_ids;

    return *this;
}

template<typename T>
constexpr auto SpMat<T>::operator=(SpMat<T>&& mat) noexcept -> SpMat<T>& {
    this->_val = std::move(mat._val);
    this->_row_ptr = std::move(mat._row_ptr);
    this->_col_ids = std::move(mat._col_ids);

    return *this;
}

}
#endif