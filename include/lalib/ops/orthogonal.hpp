#pragma once
#ifndef LALIB_ORTHOGONAL_HPP
#define LALIB_ORTHOGONAL_HPP

#include "lalib/type_traits.hpp"
#include "lalib/ops/vec_ops.hpp"
#include <ranges>
#include <concepts>

namespace lalib::orth {

/// @brief  Preforms the conventional Gram Schmidt (CGS) orthogonalization.
template<std::ranges::random_access_range C>
requires Vector<std::ranges::range_value_t<C>>
inline void cgs(C& vecs) {
    auto n = vecs[0].size();
    auto m = vecs.size();

    for (auto j = 1u; j < m; ++j) {
        assert(n == vecs[1].size());
        auto vj = vecs[j];
        for (auto k = 0u; k < j; ++k) {
            lalib::axpy(-lalib::dot(vecs[k], vj) / vecs[k].dot(vecs[k]), vecs[k], vecs[j]);
        }
    }
}

}

#endif