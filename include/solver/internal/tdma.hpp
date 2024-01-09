#pragma once 
#include <concepts>
#include <vector>

namespace lalib::solver::__internal {

template<typename T>
requires std::floating_point<T>
auto tdma(size_t n, const T* dl, const T* d, const T* du, const T* b, T* x) -> T* {
    std::vector<T> p, q;
    p.reserve(n);
    q.reserve(n);

    p.emplace_back(- du[0] / d[0]);
    q.emplace_back(b[0] / d[0]);
    for (auto i = 1u; i < n - 1; ++i) {
        auto denom = (d[i] + dl[i - 1] * p[i - 1]);
        p.emplace_back(- du[i] / denom);
        q.emplace_back((b[i] - dl[i - 1] * q[i - 1]) / denom);
    }

    x[n - 1] = (b[n - 1] - dl[n - 2] * q[n - 2]) / (d[n - 1] + dl[n - 2] * p[n - 2]);
    for (int32_t i = n - 2; i >= 0; --i) {
        x[i] = p[i] * x[i + 1] + q[i];
    }
    return x;
}

}