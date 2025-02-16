#include "lalib/solver/ilu.hpp"
#include "lalib/mat.hpp"
#include <gtest/gtest.h>

TEST(ILUTests, DenseDecompositionTest) {
    auto mat = lalib::DynMat<double>(3, 3, {
        2, -1, -2,
        -4, 6, 3,
        -4, -2, 8
    });

    lalib::solver::Ilu<double, lalib::DynMat<double>> ilu(std::move(mat));

    ASSERT_NEAR(ilu.mat()(0, 0), 2.0, 1e-6);
    ASSERT_NEAR(ilu.mat()(0, 1), -1.0, 1e-6);
    ASSERT_NEAR(ilu.mat()(0, 2), -2.0, 1e-6);
    ASSERT_NEAR(ilu.mat()(1, 0), -2.0, 1e-6);
    ASSERT_NEAR(ilu.mat()(1, 1), 4.0, 1e-6);
    ASSERT_NEAR(ilu.mat()(1, 2), -1.0, 1e-6);
    ASSERT_NEAR(ilu.mat()(2, 0), -2.0, 1e-6);
    ASSERT_NEAR(ilu.mat()(2, 1), -1.0, 1e-6);
    ASSERT_NEAR(ilu.mat()(2, 2), 3.0, 1e-6);

    auto rhs = lalib::DynVec<double>({-1, 5, 2});
    auto x = ilu.solve(rhs);

    ASSERT_NEAR(x[0], 1.0, 1e-6);
    ASSERT_NEAR(x[1], 1.0, 1e-6);
    ASSERT_NEAR(x[2], 1.0, 1e-6);
}

TEST(ILUTests, SparseDecompositionTest) {
    auto mat = lalib::SpCooMat<double>(
        std::vector<double>{ 2, -1, -2, -4, 4, -4, 11},
        std::vector<size_t>{ 0, 0, 0, 1, 1, 2, 2},
        std::vector<size_t>{ 0, 1, 2, 0, 1, 0, 2}
    );

    lalib::solver::Ilu<double, lalib::SpCooMat<double>> ilu(std::move(mat));

    ASSERT_NEAR(ilu.mat()(1, 2), 0.0, 1e-6);
    ASSERT_NEAR(ilu.mat()(2, 1), 0.0, 1e-6);
}