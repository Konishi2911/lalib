#ifndef LALIB_SP_MAT_OPS_HPP
#define LALIB_SP_MAT_OPS_HPP

#include "lalib/mat/sp_mat.hpp"
#include "lalib/ops/mat_ops.hpp"

namespace lalib {

template<typename T>
inline auto operator+(const SpCooMat<T>& m1, const SpCooMat<T>& m2) noexcept -> SpCooMat<T> {
    auto mr = SpCooMat(m1);
    mr += m2;
    return mr;
}

template<typename T>
inline auto operator-(const SpCooMat<T>& m1, const SpCooMat<T>& m2) noexcept -> SpCooMat<T> {
    auto mr = SpCooMat(m1);
    mr += -m2;
    return mr;
}

}

#endif