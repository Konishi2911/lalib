#include "lalib/ops/vec_ops.hpp"
#include <gtest/gtest.h>

// ### Negation ### //
TEST(VecOpsTests, SizedVecNegOpsDoubleTest) {
    auto v1 = lalib::SizedVec<double, 3>({1.0, 2.0, 3.0});
    auto vr = -v1;

    ASSERT_DOUBLE_EQ(-1.0, vr[0]);
    ASSERT_DOUBLE_EQ(-2.0, vr[1]);
    ASSERT_DOUBLE_EQ(-3.0, vr[2]);
}

TEST(VecOpsTests, DynVecNegOpsDoubleTest) {
    auto v1 = lalib::DynVec<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
    auto vr = -v1;

    ASSERT_DOUBLE_EQ(-1.0, vr[0]);
    ASSERT_DOUBLE_EQ(-2.0, vr[1]);
    ASSERT_DOUBLE_EQ(-3.0, vr[2]);
    ASSERT_DOUBLE_EQ(-4.0, vr[3]);
    ASSERT_DOUBLE_EQ(-5.0, vr[4]);
    ASSERT_DOUBLE_EQ(-6.0, vr[5]);
    ASSERT_DOUBLE_EQ(-7.0, vr[6]);
    ASSERT_DOUBLE_EQ(-8.0, vr[7]);
    ASSERT_DOUBLE_EQ(-9.0, vr[8]);
    ASSERT_DOUBLE_EQ(-10.0, vr[9]);
}


// ### Addition ### //
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
        lalib::vec_error::SizeMismatched
    );
}

// ### Subtraction ### //
TEST(VecOpsTests, SizedVecSizedVecSubOpsDoubleTest) {
    auto v1 = lalib::SizedVec<double, 3>({1.0, 2.0, 3.0});
    auto v2 = lalib::SizedVec<double, 3>({2.0, 3.0, 4.0});
    auto vr = v1 - v2;

    ASSERT_DOUBLE_EQ(-1.0, vr[0]);
    ASSERT_DOUBLE_EQ(-1.0, vr[1]);
    ASSERT_DOUBLE_EQ(-1.0, vr[2]);
}

TEST(VecOpsTests, SizedVecSizedVecSubOpsDoubleComplexTest) {
    auto v1 = lalib::SizedVec<std::complex<double>, 3>({std::complex<double>{1.0, 3.0}, {2.0, 2.0}, {3.0, 1.0}});
    auto v2 = lalib::SizedVec<std::complex<double>, 3>({std::complex<double>{2.0, 4.0}, {3.0, 3.0}, {4.0, 2.0}});
    auto vr = v1 - v2;

    ASSERT_DOUBLE_EQ(-1.0, vr[0].real());
    ASSERT_DOUBLE_EQ(-1.0, vr[0].imag());
    ASSERT_DOUBLE_EQ(-1.0, vr[1].real());
    ASSERT_DOUBLE_EQ(-1.0, vr[1].imag());
    ASSERT_DOUBLE_EQ(-1.0, vr[2].real());
    ASSERT_DOUBLE_EQ(-1.0, vr[2].imag());
}

TEST(VecOpsTests, DynVecDynVecSubOpsDoubleTest) {
    auto v1 = lalib::DynVec<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
    auto v2 = lalib::DynVec<double>({2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0});
    auto vr = v1 - v2;

    ASSERT_DOUBLE_EQ(-1.0, vr[0]);
    ASSERT_DOUBLE_EQ(-1.0, vr[1]);
    ASSERT_DOUBLE_EQ(-1.0, vr[2]);
    ASSERT_DOUBLE_EQ(-1.0, vr[3]);
    ASSERT_DOUBLE_EQ(-1.0, vr[4]);
    ASSERT_DOUBLE_EQ(-1.0, vr[5]);
    ASSERT_DOUBLE_EQ(-1.0, vr[6]);
    ASSERT_DOUBLE_EQ(-1.0, vr[7]);
    ASSERT_DOUBLE_EQ(-1.0, vr[8]);
}

TEST(VecOpsTests, DynVecDynVecSubOpsDoubleSizeMismatchedTest) {
    auto v1 = lalib::DynVec<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
    auto v2 = lalib::DynVec<double>({2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
    ASSERT_THROW(
        {
            auto vr = v1 - v2;
        },
        lalib::vec_error::SizeMismatched
    );
}

TEST(VecOpsTests, SizedVecAxpyTest) {
    double alpha = 2.0;
    auto v1 = lalib::SizedVec<double, 3>({1.0, 2.0, 3.0});
    auto vr = lalib::SizedVec<double, 3>::uninit();

    size_t n = 4u;
    for (auto i = 0u; i < n; ++i) {
        axpy(alpha, v1, vr);
    }

    ASSERT_DOUBLE_EQ(8.0, vr[0]);
    ASSERT_DOUBLE_EQ(16.0, vr[1]);
    ASSERT_DOUBLE_EQ(24.0, vr[2]);
}

TEST(VecOpsTests, SizedVecScalarScaleTest) {
    double alpha = 2.0;
    auto v1 = lalib::SizedVec<double, 3>({1.0, 2.0, 3.0});

    scale(alpha, v1);

    ASSERT_DOUBLE_EQ(2.0, v1[0]);
    ASSERT_DOUBLE_EQ(4.0, v1[1]);
    ASSERT_DOUBLE_EQ(6.0, v1[2]);
}

TEST(VecOpsTests, SizedVecScalarScaleOpTest) {
    double alpha = 2.0;
    auto v1 = lalib::SizedVec<double, 3>({1.0, 2.0, 3.0});

    auto vr = alpha * v1;
    auto vr1 = v1 * alpha;

    ASSERT_DOUBLE_EQ(2.0, vr[0]);
    ASSERT_DOUBLE_EQ(4.0, vr[1]);
    ASSERT_DOUBLE_EQ(6.0, vr[2]);

    ASSERT_DOUBLE_EQ(2.0, vr1[0]);
    ASSERT_DOUBLE_EQ(4.0, vr1[1]);
    ASSERT_DOUBLE_EQ(6.0, vr1[2]);
}

TEST(VecOpsTests, SizedVecScalarDivOpTest) {
    double alpha = 2.0;
    auto v1 = lalib::SizedVec<double, 3>({1.0, 2.0, 3.0});

    auto vr = v1 / alpha;

    ASSERT_DOUBLE_EQ(0.5, vr[0]);
    ASSERT_DOUBLE_EQ(1.0, vr[1]);
    ASSERT_DOUBLE_EQ(1.5, vr[2]);
}

TEST(VecOpsTests, DynVecAxpyTest) {
    double alpha = 2.0;
    auto v1 = lalib::DynVec<double>({1.0, 2.0, 3.0, 2.0, 4.0});
    auto vr = lalib::DynVec<double>::uninit(5);

    size_t n = 4u;
    for (auto i = 0u; i < n; ++i) {
        axpy(alpha, v1, vr);
    }

    ASSERT_DOUBLE_EQ(8.0, vr[0]);
    ASSERT_DOUBLE_EQ(16.0, vr[1]);
    ASSERT_DOUBLE_EQ(24.0, vr[2]);
    ASSERT_DOUBLE_EQ(16.0, vr[3]);
    ASSERT_DOUBLE_EQ(32.0, vr[4]);
}

TEST(VecOpsTests, DynVecScalarScaleTest) {
    double alpha = 3.0;
    auto v1 = lalib::DynVec<double>({1.0, 2.0, 3.0, 2.0, 4.0});

    scale(alpha, v1);

    ASSERT_DOUBLE_EQ(3.0, v1[0]);
    ASSERT_DOUBLE_EQ(6.0, v1[1]);
    ASSERT_DOUBLE_EQ(9.0, v1[2]);
    ASSERT_DOUBLE_EQ(6.0, v1[3]);
    ASSERT_DOUBLE_EQ(12.0, v1[4]);
}

TEST(VecOpsTests, DynVecScalarScaleOpTest) {
    double alpha = 3.0;
    auto v1 = lalib::DynVec<double>({1.0, 2.0, 3.0, 2.0, 4.0});

    auto vr = alpha * v1;
    auto vr1 = v1 * alpha;

    ASSERT_DOUBLE_EQ(3.0, vr[0]);
    ASSERT_DOUBLE_EQ(6.0, vr[1]);
    ASSERT_DOUBLE_EQ(9.0, vr[2]);
    ASSERT_DOUBLE_EQ(6.0, vr[3]);
    ASSERT_DOUBLE_EQ(12.0, vr[4]);

    ASSERT_DOUBLE_EQ(3.0, vr1[0]);
    ASSERT_DOUBLE_EQ(6.0, vr1[1]);
    ASSERT_DOUBLE_EQ(9.0, vr1[2]);
    ASSERT_DOUBLE_EQ(6.0, vr1[3]);
    ASSERT_DOUBLE_EQ(12.0, vr1[4]);
}

TEST(VecOpsTests, DynVecScalarDivOpTest) {
    double alpha = 3.0;
    auto v1 = lalib::DynVec<double>({1.0, 2.0, 3.0, 2.0, 4.0});

    auto vr = v1 / alpha;

    ASSERT_DOUBLE_EQ(1.0 / 3.0, vr[0]);
    ASSERT_DOUBLE_EQ(2.0 / 3.0, vr[1]);
    ASSERT_DOUBLE_EQ(1.0, vr[2]);
    ASSERT_DOUBLE_EQ(2.0 / 3.0, vr[3]);
    ASSERT_DOUBLE_EQ(4.0 / 3.0, vr[4]);
}

// # Cross product tests
TEST(VecOpsTests, SizedVecCross2ProductTest) {
    auto v1 = lalib::SizedVec<double, 2>({1.0, 0.0});
    auto v2 = lalib::SizedVec<double, 2>({0.0, 1.0});

    ASSERT_DOUBLE_EQ(1.0, lalib::cross(v1, v2));
    ASSERT_DOUBLE_EQ(-1.0, lalib::cross(v2, v1));
}

TEST(VecOpsTests, SizedVecCorss3ProductTest) {
    auto v1 = lalib::SizedVec<double, 3>({1.0, 0.0, 0.0});
    auto v2 = lalib::SizedVec<double, 3>({0.0, 1.0, 0.0});
    auto v3 = lalib::SizedVec<double, 3>::filled(1.0);
    auto vr = lalib::SizedVec<double, 3>({1.0, 1.0, 1.0});

    lalib::cross(v1, v2, vr);
    ASSERT_DOUBLE_EQ(0.0, vr[0]);
    ASSERT_DOUBLE_EQ(0.0, vr[1]);
    ASSERT_DOUBLE_EQ(1.0, vr[2]);

    lalib::cross(v2, v1, vr);
    ASSERT_DOUBLE_EQ(0.0, vr[0]);
    ASSERT_DOUBLE_EQ(0.0, vr[1]);
    ASSERT_DOUBLE_EQ(-1.0, vr[2]);

    lalib::cross(v1, v3, v3);
    ASSERT_DOUBLE_EQ(0.0, v3[0]);
    ASSERT_DOUBLE_EQ(-1.0, v3[1]);
    ASSERT_DOUBLE_EQ(1.0, v3[2]);

    auto v_tmp = lalib::cross(v2, v1);
    ASSERT_DOUBLE_EQ(0.0, v_tmp[0]);
    ASSERT_DOUBLE_EQ(0.0, v_tmp[1]);
    ASSERT_DOUBLE_EQ(-1.0, v_tmp[2]);
}

TEST(VecOpsTests, DynVecCross3ProductFailureTest) {
    auto v = lalib::DynVec<double>::filled(4, 1.0);
    auto vr = lalib::DynVec<double>({1.0, 1.0, 1.0});

    ASSERT_DEATH({ lalib::cross(v, vr, vr); }, "");
}