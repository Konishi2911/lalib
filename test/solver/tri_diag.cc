#include "../../include/solver/tri_diag.hpp"
#include <gtest/gtest.h>

TEST(TriDiagSolverTests, SizedLinearSolverTest) {
    auto mat = lalib::SizedTriDiagMat<double, 4>(
        { 1.0, 1.0, 1.0 },
        { 2.0, 2.0, 2.0, 2.0 },
        { 1.0, 1.0, 1.0 }
    );
    auto rhs = lalib::SizedVec<double, 4>(
        { 4.0, 8.0, 12.0, 11.0 }
    );
    auto solver = lalib::solver::TriDiag(mat);

    solver.solve_linear(rhs, rhs);

    EXPECT_DOUBLE_EQ(1.0, rhs[0]);
    EXPECT_DOUBLE_EQ(2.0, rhs[1]);
    EXPECT_DOUBLE_EQ(3.0, rhs[2]);
    EXPECT_DOUBLE_EQ(4.0, rhs[3]);
}

TEST(TriDiagSolverTests, DynLinearSolverTest) {
    auto mat = lalib::DynTriDiagMat<double>(
        { 1.0, 1.0, 1.0 },
        { 2.0, 2.0, 2.0, 2.0 },
        { 1.0, 1.0, 1.0 }
    );
    auto rhs = lalib::DynVec<double>(
        { 4.0, 8.0, 12.0, 11.0 }
    );
    auto solver = lalib::solver::TriDiag(mat);

    solver.solve_linear(rhs, rhs);

    EXPECT_DOUBLE_EQ(1.0, rhs[0]);
    EXPECT_DOUBLE_EQ(2.0, rhs[1]);
    EXPECT_DOUBLE_EQ(3.0, rhs[2]);
    EXPECT_DOUBLE_EQ(4.0, rhs[3]);
}