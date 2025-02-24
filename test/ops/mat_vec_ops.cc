#include "lalib/ops/mat_vec_ops.hpp"
#include <iostream>
#include <gtest/gtest.h>

// ### Matrix-Vector Multiplication ### //
TEST(MatVecOpsTests, SizedMatSizedVecMulTest) {
    auto alpha = 2.0;
    auto beta = 3.0;
    auto m = lalib::SizedMat<double, 2 ,4>({
        1.0, 2.0, 3.0, 4.0,
        2.0, 4.0, 1.0, 3.0
    });
    auto v = lalib::SizedVec<double, 4>({2.0, 3.0, 4.0, 1.0});
    auto dynv = lalib::DynVec<double>({2.0, 3.0, 4.0, 1.0});

    auto vr1 = lalib::SizedVec<double, 2>::filled(1.0);
    lalib::mul(alpha, m, v, beta, vr1);

    auto vr2 = lalib::SizedVec<double, 2>::filled(1.0);
    lalib::mul(alpha, m, dynv, beta, vr2);

    auto vr3 = m * v;
    auto vr4 = m * dynv;

    EXPECT_DOUBLE_EQ(48.0 + 3.0, vr1[0]);
    EXPECT_DOUBLE_EQ(46.0 + 3.0, vr1[1]);
    
    EXPECT_DOUBLE_EQ(48.0 + 3.0, vr2[0]);
    EXPECT_DOUBLE_EQ(46.0 + 3.0, vr2[1]);

    EXPECT_DOUBLE_EQ(24.0, vr3[0]);
    EXPECT_DOUBLE_EQ(23.0, vr3[1]);
    
    EXPECT_DOUBLE_EQ(24.0, vr4[0]);
    EXPECT_DOUBLE_EQ(23.0, vr4[1]);
}

TEST(MatVecOpsTests, SizedMatDynVecMulTest) {
    auto alpha = 2.0;
    auto beta = 3.0;
    auto m = lalib::DynMat<double>(2, 4, {
        1.0, 2.0, 3.0, 4.0,
        2.0, 4.0, 1.0, 3.0
    });
    auto v = lalib::SizedVec<double, 4>({2.0, 3.0, 4.0, 1.0});
    auto dynv = lalib::DynVec<double>({2.0, 3.0, 4.0, 1.0});

    auto vr1 = lalib::DynVec<double>::filled(2, 1.0);
    lalib::mul(alpha, m, v, beta, vr1);

    auto vr2 = lalib::DynVec<double>::filled(2, 1.0);
    lalib::mul(alpha, m, dynv, beta, vr2);

    auto vr3 = m * v;
    auto vr4 = m * dynv;

    EXPECT_DOUBLE_EQ(48.0 + 3.0, vr1[0]);
    EXPECT_DOUBLE_EQ(46.0 + 3.0, vr1[1]);
    
    EXPECT_DOUBLE_EQ(48.0 + 3.0, vr2[0]);
    EXPECT_DOUBLE_EQ(46.0 + 3.0, vr2[1]);

    EXPECT_DOUBLE_EQ(24.0, vr3[0]);
    EXPECT_DOUBLE_EQ(23.0, vr3[1]);
    
    EXPECT_DOUBLE_EQ(24.0, vr4[0]);
    EXPECT_DOUBLE_EQ(23.0, vr4[1]);
}

TEST(MatVecOpsTests, SizedMatSizedVecMulAssignTest) {
    auto m = lalib::SizedMat<double, 2 ,2>({
        1.0, 2.0,
        2.0, 4.0
    });
    auto v = lalib::SizedVec<double, 2>({2.0, 3.0});

    lalib::mul(1.0, m, v, 0.0, v);

    EXPECT_DOUBLE_EQ(8.0, v[0]);
    EXPECT_DOUBLE_EQ(16.0, v[1]);
}

TEST(MatVecOpsTests, SpMatDynVecMulTest) {
    auto alpha = 2.0;
    auto beta = 3.0;

    /*
    1.0, 2.0, 0.0,
    0.0, 0.0, 3.0, 
    4.0, 1.0, 2.0
    */
    auto m = lalib::SpMat<double>(
        {1.0, 2.0, 3.0, 4.0, 1.0, 2.0},
        {0, 2, 3, 6},
        {0, 1, 2, 0, 1, 2}
    );
    auto v = lalib::DynVec<double>({2.0, 3.0, 4.0});

    auto vr = lalib::DynVec<double>::filled(3, 1.0);
    lalib::mul(alpha, m, v, beta, vr);

    EXPECT_DOUBLE_EQ(alpha * 8.0 + beta, vr[0]);
    EXPECT_DOUBLE_EQ(alpha * 12.0 + beta, vr[1]);
    EXPECT_DOUBLE_EQ(alpha * 19.0 + beta, vr[2]);
}