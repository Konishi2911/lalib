#include <gtest/gtest.h>
#include "lalib/mat.hpp"

TEST(CommonMatTests, InvertTest) {
    auto mat = lalib::SizedMat<double, 2, 2>({1.0, 2.0, 3.0, 4.0});
    lalib::invert(mat);

    EXPECT_DOUBLE_EQ(-2.0, mat(0, 0));
    EXPECT_DOUBLE_EQ(1.0, mat(0, 1));
    EXPECT_DOUBLE_EQ(1.5, mat(1, 0));
    EXPECT_DOUBLE_EQ(-0.5, mat(1, 1));

    const auto mat2 = lalib::SizedMat<double, 2, 2>({1.0, 2.0, 3.0, 4.0});
    auto rmat = lalib::inverted(mat2);

    EXPECT_DOUBLE_EQ(-2.0, rmat(0, 0));
    EXPECT_DOUBLE_EQ(1.0, rmat(0, 1));
    EXPECT_DOUBLE_EQ(1.5, rmat(1, 0));
    EXPECT_DOUBLE_EQ(-0.5, rmat(1, 1));
}