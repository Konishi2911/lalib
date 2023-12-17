#pragma once
#include <vector>

namespace lalib {

/// Represents a dynamic-sized vector.
template<typename T>
struct DynVec {
public:
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

private:
    std::vector<T> _elems;
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

// === Specializations of utility templates === //

}