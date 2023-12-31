#pragma once
#include "../vec/sized_vec.hpp"
#include "../vec/dyn_vec.hpp"
#include "../err/error.hpp"
#include "vec_ops_core.hpp"
#include "ops_traits.hpp"

#include <cstring>


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


/// @brief      Functions to perform element by element subtraction of vectors.
/// @tparam T   a type of element
/// @tparam N   the number of elements in the vector (for `SizedVec` only).
/// @param v1   a first vector
/// @param v2   a second vector
/// @param vr   a vector to store the result
/// @return     the same as vr after the operation. 
template<typename T, size_t N>
inline auto sub(const SizedVec<T, N>& v1, const SizedVec<T, N>& v2, SizedVec<T, N>& vr) noexcept -> SizedVec<T, N>&;

template<typename T, size_t N>
inline auto sub(const SizedVec<T, N>& v1, const DynVec<T>& v2, SizedVec<T, N>& vr) -> SizedVec<T, N>&;

template<typename T, size_t N>
inline auto sub(const DynVec<T>& v1, const SizedVec<T, N>& v2, SizedVec<T, N>& vr) -> SizedVec<T, N>&;

template<typename T>
inline auto sub(const DynVec<T>& v1, const DynVec<T>& v2, DynVec<T>& vr) -> DynVec<T>&;


/// @brief      Overloaded operators to perform element by element subtraction of vectors.
/// @note       The vector storing the result is newly created internally. Consider using the `add` functions instead of using `+` operator for large-scale vectors to reduce copy cost.
/// @tparam T   a type of element
/// @tparam N   the number of elements in the vector (for `SizedVec` only).
/// @param v1   a first vector
/// @param v2   a second vector
/// @return     a vector storing the result
template<typename T, size_t N>
inline auto operator-(const SizedVec<T, N>& v1, const SizedVec<T, N>& v2) noexcept -> SizedVec<T, N>;

template<typename T>
inline auto operator-(const DynVec<T>& v1, const DynVec<T>& v2) -> DynVec<T>;


/// @brief 
/// @tparam T 
/// @tparam N 
/// @param alpha 
/// @param x 
/// @param y 
/// @return 
template<typename T, size_t N>
inline auto axpy(T alpha, const SizedVec<T, N>& x, SizedVec<T, N>& y) noexcept -> SizedVec<T, N>&;

template<typename T>
inline auto axpy(T alpha, const DynVec<T>& x, DynVec<T>& y) noexcept -> DynVec<T>&;


/// @brief      Calculate scalar multipliation of vectors
/// @details    The given vector will be overwritten through this operation.
/// @tparam T 
/// @tparam N 
/// @param alpha 
/// @param x 
/// @return 
template<typename T, size_t N>
inline auto scale(T alpha, SizedVec<T, N>& x) noexcept -> SizedVec<T, N>&;

template<typename T>
inline auto scale(T alpha, DynVec<T>& x) noexcept -> DynVec<T>&;

/// @brief      Calculate scalar multipliation of vectors
/// @details    The result is stored in newly created instance.
/// @tparam T 
/// @tparam N 
/// @param alpha 
/// @param x 
/// @return 
template<typename T, size_t N>
inline auto scaled(T alpha, const SizedVec<T, N>& x) noexcept -> SizedVec<T, N>;

template<typename T>
inline auto scaled(T alpha, const DynVec<T>& x) noexcept -> DynVec<T>;


template<typename T, size_t N>
inline auto operator*(T alpha, const SizedVec<T, N>& x) noexcept -> SizedVec<T, N>;

template<typename T>
inline auto operator*(T alpha, const DynVec<T>& x) noexcept -> DynVec<T>;


/// @brief 
/// @tparam T 
/// @tparam N 
/// @param x 
/// @param y 
/// @return 
template<typename T, size_t N>
inline auto dot(const SizedVec<T, N>& x, SizedVec<T, N>& y) noexcept -> T;

template<typename T>
inline auto dot(const DynVec<T>& x, DynVec<T>& y) noexcept -> T;


// ### Implementations ### //

inline void __check_size(size_t n1, size_t n2) {
    if (n1 != n2) {
        throw vec_error::SizeMismatched(n1, n2);
    }
}


// === Addition ============================================================== //

template<typename T, size_t N>
inline auto add(const SizedVec<T, N>& v1, const SizedVec<T, N>& v2, SizedVec<T, N>& vr) noexcept -> SizedVec<T, N>& {
    add_core(v1.data(), v2.data(), vr.data(), N);
    return vr;
}

template <typename T, size_t N>
auto add(const SizedVec<T, N> &v1, const DynVec<T> &v2, SizedVec<T, N> &vr) -> SizedVec<T, N> &
{
    __check_size(v1.size(), v2.size());
    add_core(v1.data(), v2.data(), vr.data(), N);
    return vr;
}

template <typename T, size_t N>
auto add(const DynVec<T> &v1, const SizedVec<T, N> &v2, SizedVec<T, N> &vr) -> SizedVec<T, N> &
{
    __check_size(v1.size(), v2.size());
    add_core(v1.data(), v2.data(), vr.data(), N);
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

// === Subtraction ============================================================== //

template<typename T, size_t N>
inline auto sub(const SizedVec<T, N>& v1, const SizedVec<T, N>& v2, SizedVec<T, N>& vr) noexcept -> SizedVec<T, N>& {
    sub_core(v1.data(), v2.data(), vr.data(), N);
    return vr;
}

template <typename T, size_t N>
auto sub(const SizedVec<T, N> &v1, const DynVec<T> &v2, SizedVec<T, N> &vr) -> SizedVec<T, N> &
{
    __check_size(v1.size(), v2.size());
    sub_core(v1.data(), v2.data(), vr.data(), N);
    return v1;
}

template <typename T, size_t N>
auto sub(const DynVec<T> &v1, const SizedVec<T, N> &v2, SizedVec<T, N> &vr) -> SizedVec<T, N> &
{
    __check_size(v1.size(), v2.size());
    sub_core(v1.data(), v2.data(), vr.data(), N);
    return vr;
}

template<typename T>
inline auto sub(const DynVec<T>& v1, const DynVec<T>& v2, DynVec<T>& vr) -> DynVec<T>& {
    __check_size(v1.size(), v2.size());
    sub_core(v1.data(), v2.data(), vr.data(), v1.size());
    return vr;
}


template<typename T, size_t N>
inline auto operator-(const SizedVec<T, N>& v1, const SizedVec<T, N>& v2) noexcept -> SizedVec<T, N> {
    auto vr = SizedVec(v1);
    sub(v1, v2, vr);
    return vr;
}

template <typename T>
auto operator-(const DynVec<T> &v1, const DynVec<T> &v2) -> DynVec<T>
{
    auto vr = DynVec(v1);
    sub(v1, v2, vr);
    return vr;
}


// === AXPY ============================================================== //

template <typename T, size_t N>
auto axpy(T alpha, const SizedVec<T, N> &x, SizedVec<T, N> &y) noexcept -> SizedVec<T, N> &
{
    axpy_core(alpha, x.data(), y.data(), N);
    return y;
}
template <typename T>
auto axpy(T alpha, const DynVec<T> &x, DynVec<T> &y) noexcept -> DynVec<T> &
{
    __check_size(x.size(), y.size());
    axpy_core(alpha, x.data(), y.data(), x.size());
    return y;
}


// === SCALE ============================================================== //


template<typename T, size_t N>
auto scale(T alpha, SizedVec<T, N>& x) noexcept -> SizedVec<T, N>& {
    scal_core(alpha, x.data(), x.size());
    return x;
}
template<typename T>
auto scale(T alpha, DynVec<T>& x) noexcept -> DynVec<T>& {
    scal_core(alpha, x.data(), x.size());
    return x;
}


template<typename T, size_t N>
auto scaled(T alpha, const SizedVec<T, N>& x) noexcept -> SizedVec<T, N> {
    return alpha * x;
}
template<typename T>
auto scaled(T alpha, const DynVec<T>& x) noexcept -> DynVec<T> {
    return alpha * x;
}


template<typename T, size_t N>
auto operator*(T alpha, const SizedVec<T, N>& x) noexcept -> SizedVec<T, N> {
    auto r = x;
    scal_core(alpha, r.data(), r.size());
    return r;
}

template<typename T>
auto operator*(T alpha, const DynVec<T>& x) noexcept -> DynVec<T> {
    auto r = x;
    scal_core(alpha, r.data(), r.size());
    return r;
}

template<typename T, size_t N>
auto operator*(const SizedVec<T, N>& x, T alpha) noexcept -> SizedVec<T, N> {
    auto r = x;
    scal_core(alpha, r.data(), r.size());
    return r;
}

template<typename T>
auto operator*(const DynVec<T>& x, T alpha) noexcept -> DynVec<T> {
    auto r = x;
    scal_core(alpha, r.data(), r.size());
    return r;
}

template<typename T, size_t N>
auto operator/(const SizedVec<T, N>& x, T alpha) noexcept -> SizedVec<T, N> {
    auto r = x;
    scal_core(reciprocal(alpha), r.data(), r.size());
    return r;
}

template<typename T>
auto operator/(const DynVec<T>& x, T alpha) noexcept -> DynVec<T> {
    auto r = x;
    scal_core(reciprocal(alpha), r.data(), r.size());
    return r;
}


// === DOT ============================================================== //

template<typename T, size_t N>
auto dot(const SizedVec<T, N>& x, const SizedVec<T, N>& y) noexcept -> T {
    T d;
    d = dot_core(x.data(), y.data(), x.size());
    return d;
}
template<typename T>
auto dot(const DynVec<T>& x, const DynVec<T>& y) noexcept -> T {
    T d;
    __check_size(x.size(), y.size());
    d = dot_core(x.data(), y.data(), x.size());
    return d;
}

}