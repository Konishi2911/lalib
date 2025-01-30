#include <gtest/gtest.h>
#include "lalib/ops/mat_ops.hpp"

TEST(MatOpsTest, ScaleTest) {
    auto mat = lalib::SizedMat<double, 2, 3>({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
    scale(2.0, mat);

    EXPECT_DOUBLE_EQ(2.0, mat(0, 0));
    EXPECT_DOUBLE_EQ(4.0, mat(0, 1));
    EXPECT_DOUBLE_EQ(6.0, mat(0, 2));
    EXPECT_DOUBLE_EQ(8.0, mat(1, 0));
    EXPECT_DOUBLE_EQ(10.0, mat(1, 1));
    EXPECT_DOUBLE_EQ(12.0, mat(1, 2));

    auto mat2 = lalib::SizedMat<double, 2, 3>({ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
    auto rmat = 2.0 * mat2;

    EXPECT_DOUBLE_EQ(2.0, rmat(0, 0));
    EXPECT_DOUBLE_EQ(4.0, rmat(0, 1));
    EXPECT_DOUBLE_EQ(6.0, rmat(0, 2));
    EXPECT_DOUBLE_EQ(8.0, rmat(1, 0));
    EXPECT_DOUBLE_EQ(10.0, rmat(1, 1));
    EXPECT_DOUBLE_EQ(12.0, rmat(1, 2));
}

TEST(MatOpsTests, SpScaleTest) {
    auto mat = lalib::SpMat<double>(
        { 1.0, 2.0, 3.0, 4.0 },
        { 0, 2, 4 },
        { 0, 1, 0, 1 }
    );
    scale(2.0, mat);

    EXPECT_DOUBLE_EQ(2.0, mat(0, 0));
    EXPECT_DOUBLE_EQ(4.0, mat(0, 1));
    EXPECT_DOUBLE_EQ(6.0, mat(1, 0));
    EXPECT_DOUBLE_EQ(8.0, mat(1, 1));

    auto mat2 = lalib::SpMat<double>(
        { 1.0, 2.0, 3.0, 4.0 },
        { 0, 2, 4 },
        { 0, 1, 0, 1 }
    );
    auto rmat = 2.0 * mat2;

    EXPECT_DOUBLE_EQ(2.0, rmat(0, 0));
    EXPECT_DOUBLE_EQ(4.0, rmat(0, 1));
    EXPECT_DOUBLE_EQ(6.0, rmat(1, 0));
    EXPECT_DOUBLE_EQ(8.0, rmat(1, 1));
}