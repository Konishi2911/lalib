#include <gtest/gtest.h>
#include "lalib/ops/sp_mat_ops.hpp"


TEST(MatMatOpsTests, SpCooMatAdditionTest) {
    auto m1 = lalib::SpCooMat<double>({
        1.0, 3.0, 5.0, 7.0
    }, { 0, 0, 1, 1 }, { 0, 1, 0, 1 });

    auto m2 = lalib::SpCooMat<double>({
        1.0, 3.0, 5.0, 7.0
    }, { 0, 0, 1, 1 }, { 0, 1, 0, 1 });

    auto m3 = m1 + m2;

    ASSERT_DOUBLE_EQ(2.0, m3(0, 0));
    ASSERT_DOUBLE_EQ(6.0, m3(0, 1));
    ASSERT_DOUBLE_EQ(10.0, m3(1, 0));
    ASSERT_DOUBLE_EQ(14.0, m3(1, 1));
}

TEST(MatMatOpsTests, SpCooMatSubtractionTest) {
    auto m1 = lalib::SpCooMat<double>({
        1.0, 3.0, 5.0, 7.0
    }, { 0, 0, 1, 1 }, { 0, 1, 0, 1 });

    auto m2 = lalib::SpCooMat<double>({
        1.0, 3.0, 5.0, 7.0
    }, { 0, 0, 1, 1 }, { 0, 1, 0, 1 });

    auto m3 = m1 - m2;

    ASSERT_DOUBLE_EQ(0.0, m3(0, 0));
    ASSERT_DOUBLE_EQ(0.0, m3(0, 1));
    ASSERT_DOUBLE_EQ(0.0, m3(1, 0));
    ASSERT_DOUBLE_EQ(0.0, m3(1, 1));
}