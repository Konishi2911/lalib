#include <gtest/gtest.h>
#include "lalib/ops/mat_mat_ops.hpp"
#include <iostream>

TEST(MatMatOpsTests, SizedMatSizedMatAddTest) {
    auto m1 = lalib::SizedMat<double, 2, 3>({
        1.0, 3.0, 5.0,
        2.0, 4.0, 6.0
    });
    auto m2 = m1 + m1;

    ASSERT_DOUBLE_EQ(2 * m1(0, 0), m2(0, 0));
    ASSERT_DOUBLE_EQ(2 * m1(0, 1), m2(0, 1));
    ASSERT_DOUBLE_EQ(2 * m1(0, 2), m2(0, 2));
    ASSERT_DOUBLE_EQ(2 * m1(1, 0), m2(1, 0));
    ASSERT_DOUBLE_EQ(2 * m1(1, 1), m2(1, 1));
    ASSERT_DOUBLE_EQ(2 * m1(1, 2), m2(1, 2));
}

TEST(MatMatOpsTests, SizedMatSizedMatSubTest) {
    auto m1 = lalib::SizedMat<double, 2, 3>({
        1.0, 3.0, 5.0,
        2.0, 4.0, 6.0
    });
    auto m2 = m1 - m1;

    ASSERT_DOUBLE_EQ(0.0, m2(0, 0));
    ASSERT_DOUBLE_EQ(0.0, m2(0, 1));
    ASSERT_DOUBLE_EQ(0.0, m2(0, 2));
    ASSERT_DOUBLE_EQ(0.0, m2(1, 0));
    ASSERT_DOUBLE_EQ(0.0, m2(1, 1));
    ASSERT_DOUBLE_EQ(0.0, m2(1, 2));
}

TEST(MatMatOpsTests, SizedMatSizedMatMulTest) {
    auto m1 = lalib::SizedMat<double, 2, 3>({
        1.0, 3.0, 5.0,
        2.0, 4.0, 6.0
    });
    auto m2 = lalib::SizedMat<double, 3, 2>({ 
        1.0, 2.0,
        2.0, 3.0,
        3.0, 4.0
    });
    auto m3 = lalib::SizedMat<double, 2, 2>::filled(1.0);
    auto alpha = 2.0;
    auto beta = 3.0;

    lalib::mul(2.0, m1, m2, 3.0, m3);
    auto m4 = m1 * m2;

    ASSERT_DOUBLE_EQ(alpha * 22.0 + beta, m3(0, 0));
    ASSERT_DOUBLE_EQ(alpha * 31.0 + beta, m3(0, 1));
    ASSERT_DOUBLE_EQ(alpha * 28.0 + beta, m3(1, 0));
    ASSERT_DOUBLE_EQ(alpha * 40.0 + beta, m3(1, 1));

    ASSERT_DOUBLE_EQ(22.0, m4(0, 0));
    ASSERT_DOUBLE_EQ(31.0, m4(0, 1));
    ASSERT_DOUBLE_EQ(28.0, m4(1, 0));
    ASSERT_DOUBLE_EQ(40.0, m4(1, 1));
}


TEST(MatMatOpsTests, DynMatDynMatMulTest) {
    auto m1 = lalib::DynMat<double>(2, 3, {
        1.0, 3.0, 5.0,
        2.0, 4.0, 6.0
    });
    auto m2 = lalib::DynMat<double>(3, 2, { 
        1.0, 2.0,
        2.0, 3.0,
        3.0, 4.0
    });
    auto m3 = lalib::DynMat<double>::filled(1.0, 2, 2);
    auto alpha = 2.0;
    auto beta = 3.0;

    lalib::mul(2.0, m1, m2, 3.0, m3);
    auto m4 = m1 * m2;

    ASSERT_DOUBLE_EQ(alpha * 22.0 + beta, m3(0, 0));
    ASSERT_DOUBLE_EQ(alpha * 31.0 + beta, m3(0, 1));
    ASSERT_DOUBLE_EQ(alpha * 28.0 + beta, m3(1, 0));
    ASSERT_DOUBLE_EQ(alpha * 40.0 + beta, m3(1, 1));

    ASSERT_DOUBLE_EQ(22.0, m4(0, 0));
    ASSERT_DOUBLE_EQ(31.0, m4(0, 1));
    ASSERT_DOUBLE_EQ(28.0, m4(1, 0));
    ASSERT_DOUBLE_EQ(40.0, m4(1, 1));
}

TEST(MatMatOpsTests, SizedMatSizedMatMulAssignTest) {
    auto m1 = lalib::SizedMat<double, 2, 2>({
        1.0, 3.0,
        2.0, 4.0
    });
    auto m2 = lalib::SizedMat<double, 2, 2>({ 
        1.0, 2.0,
        2.0, 3.0
    });
    auto alpha = 2.0;
    auto beta = 3.0;

    lalib::mul(2.0, m1, m2, 3.0, m1);

    ASSERT_DOUBLE_EQ(alpha * 7.0 + beta * 1.0, m1(0, 0));
    ASSERT_DOUBLE_EQ(alpha * 11.0 + beta * 3.0, m1(0, 1));
    ASSERT_DOUBLE_EQ(alpha * 10.0 + beta * 2.0, m1(1, 0));
    ASSERT_DOUBLE_EQ(alpha * 16.0 + beta * 4.0, m1(1, 1));
}