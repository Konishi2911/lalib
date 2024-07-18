#include <gtest/gtest.h>
#include "lalib/solver/tri_diag.hpp"

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

TEST(TriDiagSolverTests, SizedMatLinearSolverTest) {
    auto mat = lalib::SizedTriDiagMat<double, 4>(
        { 1.0, 1.0, 1.0 },
        { 2.0, 2.0, 2.0, 2.0 },
        { 1.0, 1.0, 1.0 }
    );
    auto rhs = lalib::SizedMat<double, 4, 3>({ 
        4.0, 2.0, 8.0,
        8.0, 4.0, 16.0,
        12.0, 6.0, 24.0,
        11.0, 5.5, 22.0
    });
    auto solver = lalib::solver::TriDiag(mat);

    solver.solve_linear(rhs, rhs);

    EXPECT_DOUBLE_EQ(1.0, rhs(0, 0));
    EXPECT_DOUBLE_EQ(2.0, rhs(1, 0));
    EXPECT_DOUBLE_EQ(3.0, rhs(2, 0));
    EXPECT_DOUBLE_EQ(4.0, rhs(3, 0));
    EXPECT_DOUBLE_EQ(0.5, rhs(0, 1));
    EXPECT_DOUBLE_EQ(1.0, rhs(1, 1));
    EXPECT_DOUBLE_EQ(1.5, rhs(2, 1));
    EXPECT_DOUBLE_EQ(2.0, rhs(3, 1));
    EXPECT_DOUBLE_EQ(2.0, rhs(0, 2));
    EXPECT_DOUBLE_EQ(4.0, rhs(1, 2));
    EXPECT_DOUBLE_EQ(6.0, rhs(2, 2));
    EXPECT_DOUBLE_EQ(8.0, rhs(3, 2));
}

TEST(TriDiagSolverTests, DynMatLinearSolverTest) {
    auto mat = lalib::DynTriDiagMat<double>(
        { 1.0, 1.0, 1.0 },
        { 2.0, 2.0, 2.0, 2.0 },
        { 1.0, 1.0, 1.0 }
    );
    auto rhs = lalib::DynMat<double>(4, 3, { 
        4.0, 2.0, 8.0,
        8.0, 4.0, 16.0,
        12.0, 6.0, 24.0,
        11.0, 5.5, 22.0
    });
    auto solver = lalib::solver::TriDiag(mat);

    solver.solve_linear(rhs, rhs);

    EXPECT_DOUBLE_EQ(1.0, rhs(0, 0));
    EXPECT_DOUBLE_EQ(2.0, rhs(1, 0));
    EXPECT_DOUBLE_EQ(3.0, rhs(2, 0));
    EXPECT_DOUBLE_EQ(4.0, rhs(3, 0));
    EXPECT_DOUBLE_EQ(0.5, rhs(0, 1));
    EXPECT_DOUBLE_EQ(1.0, rhs(1, 1));
    EXPECT_DOUBLE_EQ(1.5, rhs(2, 1));
    EXPECT_DOUBLE_EQ(2.0, rhs(3, 1));
    EXPECT_DOUBLE_EQ(2.0, rhs(0, 2));
    EXPECT_DOUBLE_EQ(4.0, rhs(1, 2));
    EXPECT_DOUBLE_EQ(6.0, rhs(2, 2));
    EXPECT_DOUBLE_EQ(8.0, rhs(3, 2));
}