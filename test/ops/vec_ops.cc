#include "../..//include/ops/vec_ops.hpp"
#include <gtest/gtest.h>

TEST(VecOpsTests, SizedVecSizedVecAddOpsDoubleTest) {
    auto v1 = lalib::SizedVec<double, 3>({1.0, 2.0, 3.0});
    auto v2 = lalib::SizedVec<double, 3>({2.0, 3.0, 4.0});
    auto vr = v1 + v2;

    ASSERT_DOUBLE_EQ(3.0, vr[0]);
    ASSERT_DOUBLE_EQ(5.0, vr[1]);
    ASSERT_DOUBLE_EQ(7.0, vr[2]);
}

TEST(VecOpsTests, SizedVecSizedVecAddOpsDoubleComplexTest) {
    auto v1 = lalib::SizedVec<std::complex<double>, 3>({std::complex<double>{1.0, 3.0}, {2.0, 2.0}, {3.0, 1.0}});
    auto v2 = lalib::SizedVec<std::complex<double>, 3>({std::complex<double>{2.0, 4.0}, {3.0, 3.0}, {4.0, 2.0}});
    auto vr = v1 + v2;

    ASSERT_DOUBLE_EQ(3.0, vr[0].real());
    ASSERT_DOUBLE_EQ(7.0, vr[0].imag());
    ASSERT_DOUBLE_EQ(5.0, vr[1].real());
    ASSERT_DOUBLE_EQ(5.0, vr[1].imag());
    ASSERT_DOUBLE_EQ(7.0, vr[2].real());
    ASSERT_DOUBLE_EQ(3.0, vr[2].imag());
}

TEST(VecOpsTests, DynVecDynVecAddOpsDoubleTest) {
    auto v1 = lalib::DynVec<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
    auto v2 = lalib::DynVec<double>({2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0});
    auto vr = v1 + v2;

    ASSERT_DOUBLE_EQ(3.0, vr[0]);
    ASSERT_DOUBLE_EQ(5.0, vr[1]);
    ASSERT_DOUBLE_EQ(7.0, vr[2]);
    ASSERT_DOUBLE_EQ(9.0, vr[3]);
    ASSERT_DOUBLE_EQ(11.0, vr[4]);
    ASSERT_DOUBLE_EQ(13.0, vr[5]);
    ASSERT_DOUBLE_EQ(15.0, vr[6]);
    ASSERT_DOUBLE_EQ(17.0, vr[7]);
    ASSERT_DOUBLE_EQ(19.0, vr[8]);
}

TEST(VecOpsTests, DynVecDynVecAddOpsDoubleSizeMismatchedTest) {
    auto v1 = lalib::DynVec<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
    auto v2 = lalib::DynVec<double>({2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
    ASSERT_THROW(
        {
            auto vr = v1 + v2;
        },
        vec_error::SizeMismatched
    );
}