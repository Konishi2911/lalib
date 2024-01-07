#pragma once
#include <concepts>
#include <complex>

namespace lalib {

template<typename T>
requires std::floating_point<T>
constexpr auto reciprocal(const T& val) noexcept {
    return 1.0 / val;
}

template<typename T>
requires std::floating_point<T>
constexpr auto reciprocal(const std::complex<T>& val) noexcept {
    return 1.0 / val;
}

}