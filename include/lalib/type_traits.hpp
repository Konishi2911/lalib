#pragma once
#ifndef LALIB_TYPE_TRAITS_HPP
#define LALIB_TYPE_TRAITS_HPP

#include <concepts>
#include <type_traits>

namespace lalib {

template<typename T>
concept Vector = 
    requires(const T& v) {
        typename T::ElemType;

        // Inspection
        { v.size() } -> std::convertible_to<size_t>;

        // Indexing
        { v[std::declval<size_t>()] } -> std::convertible_to<typename T::ElemType>;

        // Basic vector operations
        { v.norm2() } -> std::convertible_to<typename T::ElemType>;
    } && 
    requires(T& v) {
        { v.data() } -> std::convertible_to<typename T::ElemType*>;
    };

template<typename T>
concept Matrix = requires(const T& v) {
    typename T::ElemType;

    // Inspection
    { v.shape() } -> std::convertible_to<std::pair<size_t, size_t>>;

    // Indexing
    { v(std::declval<size_t>(), std::declval<size_t>()) } -> std::convertible_to<typename T::ElemType>;
};

template<typename T>
concept TriDiagMatrix = Matrix<T> &&
requires(T& v) {
    // accessors to the diagonal elements
    { v.data_dl() } -> std::convertible_to<typename T::ElemType*>;
    { v.data_d() } -> std::convertible_to<typename T::ElemType*>;
    { v.data_du() } -> std::convertible_to<typename T::ElemType*>;
};

}

#endif