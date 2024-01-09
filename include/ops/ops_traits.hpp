#pragma once
#include <concepts>
#include <complex>

namespace lalib {

// === Zero === //

template<typename T>
requires std::floating_point<T>
struct Zero {
    static auto value() noexcept -> T { return 0.0; }
};

template<typename T>
requires std::floating_point<T>
struct Zero<std::complex<T>> {
    static auto value() noexcept -> T { return std::complex<T>(0.0, 0.0); }
};

// === One === //

template<typename T>
requires std::floating_point<T>
struct One {
    static auto value() noexcept -> T { return 1.0; }
};

template<typename T>
requires std::floating_point<T>
struct One<std::complex<T>> {
    static auto value() noexcept -> T { return std::complex<T>(1.0, 0.0); }
};


// === Reciprocal === //

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