#pragma once
#include <vector>
#include "../ops/ops_traits.hpp"
#include "../ops/vec_ops_core.hpp"
#include "../err/error.hpp"

namespace lalib {

// Declare symbols
template<typename T, size_t N> struct SizedVec;

/// Represents a dynamic-sized vector.
template<typename T>
struct DynVec {
public:
    using ElemType = T;

    // === Initializations === //

    /// @brief Create a sized vector with given array with copy.
    constexpr DynVec(const std::vector<T>& vec) noexcept: 
        _elems(vec) {}

    /// @brief Create a sized vector with given array with move.
    constexpr DynVec(std::vector<T>&& vec) noexcept: 
        _elems( std::move(vec) ) {}

    /// @brief A copy constructor
    constexpr DynVec(const DynVec<T>& vec) noexcept = default;

    /// @brief A move constructor
    constexpr DynVec(DynVec<T>&& vec) noexcept = default;

    /// @brief Destructor
    ~DynVec() noexcept = default;

    /// @brief Creates a sized vector with uninitialized elements.
    /// @param n    the number of elements
    static constexpr auto uninit(size_t n) noexcept -> DynVec<T>;

    /// @brief Creates a sized vector filled with the given value.
    static constexpr auto filled(size_t count, T value) noexcept -> DynVec<T>;


    // === Assignment === //

    /// @brief Replaces the elements of the vector.
    /// @param vec the vector to use as data source
    /// @return a reference of the vector after modified by the operation.
    constexpr auto operator=(const DynVec<T>& vec) noexcept -> DynVec<T>& = default;

    /// @brief Replaces the elements of the vector 
    /// @param vec the vector to use as data source
    /// @return a reference of the vector after modified by the operation.
    constexpr auto operator=(DynVec<T>&& vec) noexcept -> DynVec<T>& = default;


    // === Inspecting === //

    /// @brief Returns the number of elements.
    constexpr auto size() const noexcept -> size_t;


    // === Accessing, Indexing, and Iterators === //
    using Iter = std::vector<T>::iterator;
    using ConstIter = std::vector<T>::const_iterator;

    constexpr auto begin() noexcept -> Iter;
    constexpr auto begin() const noexcept -> ConstIter;
    constexpr auto cbegin() const noexcept -> ConstIter;
    constexpr auto end() noexcept-> Iter;
    constexpr auto end() const noexcept -> ConstIter;
    constexpr auto cend() const noexcept -> ConstIter;

    constexpr auto operator[](size_t i) const -> const T&;
    constexpr auto operator[](size_t i) -> T&;

    constexpr auto data() noexcept -> T*;
    constexpr auto data() const noexcept -> const T*;


    // === Operations === //
    template<size_t N> constexpr auto dot(const SizedVec<T, N>& v) const -> T;
    constexpr auto dot(const DynVec<T>& v) const -> T;

    constexpr auto norm2() const noexcept -> T;

private:
    std::vector<T> _elems;

    void __check_size(size_t, size_t) const;
};

template <typename T>
inline constexpr auto DynVec<T>::uninit(size_t n) noexcept -> DynVec<T>
{
    return DynVec(std::vector<T>(n));
}

template <typename T>
inline constexpr auto DynVec<T>::filled(size_t count, T value) noexcept -> DynVec<T>
{
    return DynVec( std::vector(count, value) );
}

template <typename T>
inline constexpr auto DynVec<T>::size() const noexcept -> size_t {
    return this->_elems.size();
}

template <typename T>
inline constexpr auto DynVec<T>::begin() noexcept -> Iter
{
    return this->_elems.begin();
}

template <typename T>
inline constexpr auto DynVec<T>::begin() const noexcept -> ConstIter
{
    return this->cbegin();
}

template <typename T>
inline constexpr auto DynVec<T>::cbegin() const noexcept -> ConstIter
{
    return this->_elems.cbegin();
}

template <typename T>
inline constexpr auto DynVec<T>::end() noexcept -> Iter
{
    return this->_elems.end();
}

template <typename T>
inline constexpr auto DynVec<T>::end() const noexcept -> ConstIter
{
    return this->cend();
}

template <typename T>
inline constexpr auto DynVec<T>::cend() const noexcept -> ConstIter
{
    return this->_elems.cend();
}

template <typename T>
inline constexpr auto DynVec<T>::operator[](size_t i) const -> const T &
{
    return this->_elems[i];
}

template <typename T>
inline constexpr auto DynVec<T>::operator[](size_t i) -> T& {
    return this->_elems[i];
}

template <typename T>
inline constexpr auto DynVec<T>::data() noexcept -> T *
{
    return this->_elems.data();
}

template <typename T>
inline constexpr auto DynVec<T>::data() const noexcept -> const T *
{
    return this->_elems.data();
}


template <typename T>
inline constexpr auto DynVec<T>::dot(const DynVec<T> &v) const -> T
{
    T d;
    this->__check_size(this->size(), v.size());
    d = dot_core(this->data(), v.data(), this->size());
    return d;
}

template <typename T>
inline constexpr auto DynVec<T>::norm2() const noexcept -> T
{
    T norm;
    norm = norm2_core(this->data(), this->size());
    return norm;
}

// === Specializations of utility templates === //

template <typename T>
template <size_t N>
inline constexpr auto DynVec<T>::dot(const SizedVec<T, N> &v) const -> T
{
    T d;
    this->__check_size(this->size(), v.size());
    d = dot_core(this->data(), v.data(), this->size());
    return d;
}


template<typename T>
inline void DynVec<T>::__check_size(size_t n1, size_t n2) const {
    if (n1 != n2) {
        throw vec_error::SizeMismatched(n1, n2);
    }
}


}