#pragma once
#include <array>

namespace lalib {

/// Represents a compile-time fixed size vector.
template<typename T, size_t N>
struct SizedVec {
public:
    // === Initializations === //
    /// @brief Create a sized vector with given array with copy.
    constexpr SizedVec(const std::array<T, N>& arr) noexcept: 
        _elems(arr) {}

    /// @brief Create a sized vector with given array with move.
    constexpr SizedVec(std::array<T, N>&& arr) noexcept: 
        _elems( std::move(arr) ) {}

    /// @brief A copy constructor
    constexpr SizedVec(const SizedVec& vec) noexcept = default;

    /// @brief A move constructor
    constexpr SizedVec(SizedVec&& vec) noexcept = default;

    /// Creates a sized vector filled with the given value.
    static constexpr auto filled(T value) noexcept -> SizedVec<T, N>;


    // === Inspecting === //
    /// Returns the number of elements.
    constexpr auto size() const noexcept -> size_t;


    // === Indexing and Iterators === //
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

private:
    std::array<T, N> _elems;
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
inline constexpr auto SizedVec<T, N>::filled(T value) noexcept -> SizedVec<T, N> {
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


// === Specializations of utility templates === //

}