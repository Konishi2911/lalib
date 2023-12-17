#pragma once
#include "../vec/sized_vec.hpp"
#include "../vec/dyn_vec.hpp"
#include "../err/error.hpp"
#include "vec_ops_core.hpp"


namespace lalib {

/// @brief      Functions to perform element by element addition of vectors.
/// @tparam T   a type of element
/// @tparam N   the number of elements in the vector (for `SizedVec` only).
/// @param v1   a first vector
/// @param v2   a second vector
/// @param vr   a vector to store the result
/// @return     the same as vr after the operation. 
template<typename T, size_t N>
inline auto add(const SizedVec<T, N>& v1, const SizedVec<T, N>& v2, SizedVec<T, N>& vr) noexcept -> SizedVec<T, N>&;

template<typename T, size_t N>
inline auto add(const SizedVec<T, N>& v1, const DynVec<T>& v2, SizedVec<T, N>& vr) -> SizedVec<T, N>&;

template<typename T, size_t N>
inline auto add(const DynVec<T>& v1, const SizedVec<T, N>& v2, SizedVec<T, N>& vr) -> SizedVec<T, N>&;

template<typename T>
inline auto add(const DynVec<T>& v1, const DynVec<T>& v2, DynVec<T>& vr) -> DynVec<T>&;


/// @brief      Overloaded operators to perform element by element addition of vectors.
/// @note       The vector storing the result is newly created internally. Consider using the `add` functions instead of using `+` operator for large-scale vectors to reduce copy cost.
/// @tparam T   a type of element
/// @tparam N   the number of elements in the vector (for `SizedVec` only).
/// @param v1   a first vector
/// @param v2   a second vector
/// @return     a vector storing the result
template<typename T, size_t N>
inline auto operator+(const SizedVec<T, N>& v1, const SizedVec<T, N>& v2) noexcept -> SizedVec<T, N>;

template<typename T>
inline auto operator+(const DynVec<T>& v1, const DynVec<T>& v2) -> DynVec<T>;



// ### Implementations ### //

inline void __check_size(size_t n1, size_t n2) {
    if (n1 != n2) {
        throw vec_error::SizeMismatched(n1, n2);
    }
}

template<typename T, size_t N>
inline auto add(const SizedVec<T, N>& v1, const SizedVec<T, N>& v2, SizedVec<T, N>& vr) noexcept -> SizedVec<T, N>& {
    add_core_sized<T, N>(v1.data(), v2.data(), vr.data());
    return vr;
}

template <typename T, size_t N>
auto add(const SizedVec<T, N> &v1, const DynVec<T> &v2, SizedVec<T, N> &vr) -> SizedVec<T, N> &
{
    __check_size(v1.size(), v2.size());
    add_core_sized<T, N>(v1.data(), v2.data(), vr.data());
    return vr;
}

template <typename T, size_t N>
auto add(const DynVec<T> &v1, const SizedVec<T, N> &v2, SizedVec<T, N> &vr) -> SizedVec<T, N> &
{
    __check_size(v1.size(), v2.size());
    add_core_sized<T, N>(v1.data(), v2.data(), vr.data());
    return vr;
}

template<typename T>
inline auto add(const DynVec<T>& v1, const DynVec<T>& v2, DynVec<T>& vr) -> DynVec<T>& {
    __check_size(v1.size(), v2.size());
    add_core(v1.data(), v2.data(), vr.data(), v1.size());
    return vr;
}


template<typename T, size_t N>
inline auto operator+(const SizedVec<T, N>& v1, const SizedVec<T, N>& v2) noexcept -> SizedVec<T, N> {
    auto vr = SizedVec(v1);
    add(v1, v2, vr);
    return vr;
}

template <typename T>
auto operator+(const DynVec<T> &v1, const DynVec<T> &v2) -> DynVec<T>
{
    auto vr = DynVec(v1);
    add(v1, v2, vr);
    return vr;
}
}
