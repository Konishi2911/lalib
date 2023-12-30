#pragma once
#include <array>
#include "../ops/vec_ops_core.hpp"

namespace lalib {

// Declare symbols
template<typename T> struct DynVec;

/// Represents a compile-time fixed size vector.
template<typename T, size_t N>
struct SizedVec {
public:
    // === Initializations === //

    /// @brief Create a sized vector with given array with copy.
    constexpr SizedVec(const std::array<T, N>& arr) noexcept: 
        _elems(arr) {}

    /// @brief A copy constructor
    constexpr SizedVec(const SizedVec& vec) noexcept = default;

    /// @brief Creates a sized vector with uninitialized elements.
    static constexpr auto uninit() noexcept -> SizedVec<T, N>;

    /// Creates a sized vector filled with the given value.
    static constexpr auto filled(T value) noexcept -> SizedVec<T, N>;


    // === Assignment === //

    /// @brief Replaces the elements of the vector.
    /// @param vec the vector to use as data source
    /// @return a reference of the vector after modified by the operation.
    constexpr auto operator=(const SizedVec<T, N>& vec) noexcept -> SizedVec<T, N>& = default;


    // === Inspecting === //

    /// Returns the number of elements.
    constexpr auto size() const noexcept -> size_t;


    // === Accessing, Indexing, and Iterators === //
    using Iter = std::array<T, N>::iterator;
    using ConstIter = std::array<T, N>::const_iterator;

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
    constexpr auto dot(const SizedVec<T, N>& v) const noexcept -> T;
    constexpr auto dot(const DynVec<T>& v) const noexcept -> T;

private:
    std::array<T, N> _elems;

    SizedVec() noexcept = default;
};


template<typename T, size_t N>
constexpr auto generate_std_array_filled_with(T value) -> std::array<T, N> {
    return [value]() constexpr -> std::array<T, N> {
        std::array<T, N> arr;
        arr.fill(value);
        return arr;
    }();
}

template <typename T, size_t N>
inline constexpr auto SizedVec<T, N>::uninit() noexcept -> SizedVec<T, N>
{
    return SizedVec<T, N>();
}

template <typename T, size_t N>
inline constexpr auto SizedVec<T, N>::filled(T value) noexcept -> SizedVec<T, N>
{
    return SizedVec(generate_std_array_filled_with<T, N>(value));
}

template <typename T, size_t N>
inline constexpr auto SizedVec<T, N>::size() const noexcept -> size_t {
    return this->_elems.size();
}

template <typename T, size_t N>
inline constexpr auto SizedVec<T, N>::begin() noexcept -> Iter
{
    return this->_elems.begin();
}

template <typename T, size_t N>
inline constexpr auto SizedVec<T, N>::begin() const noexcept -> ConstIter
{
    return this->cbegin();
}

template <typename T, size_t N>
inline constexpr auto SizedVec<T, N>::cbegin() const noexcept -> ConstIter
{
    return this->_elems.cbegin();
}

template <typename T, size_t N>
inline constexpr auto SizedVec<T, N>::end() noexcept -> Iter
{
    return this->_elems.end();
}

template <typename T, size_t N>
inline constexpr auto SizedVec<T, N>::end() const noexcept -> ConstIter
{
    return this->cend();
}

template <typename T, size_t N>
inline constexpr auto SizedVec<T, N>::cend() const noexcept -> ConstIter
{
    return this->_elems.cend();
}

template <typename T, size_t N>
inline constexpr auto SizedVec<T, N>::operator[](size_t i) const -> const T &
{
    return this->_elems[i];
}

template <typename T, size_t N>
inline constexpr auto SizedVec<T, N>::operator[](size_t i) -> T& {
    return this->_elems[i];
}

template <typename T, size_t N>
inline constexpr auto SizedVec<T, N>::data() noexcept -> T *
{
    return this->_elems.data();
}

template <typename T, size_t N>
inline constexpr auto SizedVec<T, N>::data() const noexcept -> const T *
{
    return this->_elems.data();
}

template <typename T, size_t N>
inline constexpr auto SizedVec<T, N>::dot(const SizedVec<T, N> &v) const noexcept -> T
{
    T d;
    d = dot_core(this->data(), v.data(), this->size());
    return d;
}

template <typename T, size_t N>
inline constexpr auto SizedVec<T, N>::dot(const DynVec<T> &v) const noexcept -> T
{
    T d;
    d = dot_core(this->data(), v.data(), this->size());
    return d;
}

// === Specializations of utility templates === //

}