#pragma once 
#ifndef LALIB_SOLVER_INTERNAL_TDMA_HPP
#define LALIB_SOLVER_INTERNAL_TDMA_HPP

#include <concepts>
#include <vector>

namespace lalib::solver::__internal {

template<typename T>
requires ::std::floating_point<T>
auto tdma(size_t n, size_t nrow, const T* dl, const T* d, const T* du, const T* b, T* x) -> T* {
    for (auto k = 0u; k < nrow; ++k) {
        std::vector<T> p, q;
        p.reserve(n);
        q.reserve(n);

        p.emplace_back(- du[0] / d[0]);
        q.emplace_back(b[k] / d[0]);
        for (auto i = 1u; i < n - 1; ++i) {
            auto denom = (d[i] + dl[i - 1] * p[i - 1]);
            p.emplace_back(- du[i] / denom);
            q.emplace_back((b[k + i * nrow] - dl[i - 1] * q[i - 1]) / denom);
        }

        auto tail = n - 1;
        x[k + tail * nrow] = (b[k + tail * nrow] - dl[tail - 1] * q[tail - 1]) / (d[tail] + dl[tail - 1] * p[tail - 1]);
        for (int32_t i = tail - 1; i >= 0; --i) {
            x[k + i * nrow] = p[i] * x[k + (i + 1) * nrow] + q[i];
        }
    }
    return x;
}

}

#endif