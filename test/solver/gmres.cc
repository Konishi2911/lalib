#include "lalib/solver/gmres.hpp"
#include <gtest/gtest.h>

TEST(GmresTests, GmresTest) {
    auto mat = lalib::DynMat<double>(3, 3, {
        4.0, 2.0, 6.0,
        2.0, 5.0, 5.0,
        6.0, 5.0, 14.0
    });
    auto b = lalib::DynVec<double>({2.0, 5.0, 1.0});
    auto gmres = lalib::solver::Gmres<double, lalib::DynMat<double>>(mat, 1e-6);
    auto sol = gmres.solve(b);

    ASSERT_NEAR(1.25, sol[0], 1e-6);
    ASSERT_NEAR(1.5, sol[1], 1e-6);
    ASSERT_NEAR(-1.0, sol[2], 1e-6);
}

TEST(GmresTests, SpGmresTest) {
    auto mat = lalib::SpMat<double>(
        {4.0, 2.0, 6.0, 2.0, 5.0, 5.0, 6.0, 5.0, 14.0}, 
        {0, 3, 6, 9}, 
        {0, 1, 2, 0, 1, 2, 0, 1, 2}
    );
    auto b = lalib::DynVec<double>({2.0, 5.0, 1.0});
    auto gmres = lalib::solver::Gmres(mat, 1e-6);
    auto sol = gmres.solve(b);

    ASSERT_NEAR(1.25, sol[0], 1e-6);
    ASSERT_NEAR(1.5, sol[1], 1e-6);
    ASSERT_NEAR(-1.0, sol[2], 1e-6);
}