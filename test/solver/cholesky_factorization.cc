#include <gtest/gtest.h>
#include "lalib/solver/cholesky_factorization.hpp"
#include "lalib/mat.hpp"
#include "lalib/vec.hpp"

#if defined LALIB_LAPACK_BACKEND
#include "lalib/solver/lapack/potr.hpp"
TEST(CholeskyDecompositionTests, LapackCholeskyDecompositionTest) {
    auto mat = lalib::DynMat<double>(std::vector{
		4.0, 2.0, 6.0,
		2.0, 5.0, 5.0,
		6.0, 5.0, 14.0
	}, 3, 3);
    auto rslt = lalib::solver::_lapack_::potrf(mat.shape().first, mat.data(), mat.shape().second);

    ASSERT_EQ(0, rslt);
	EXPECT_DOUBLE_EQ(2.0, mat(0, 0));
	EXPECT_DOUBLE_EQ(1.0, mat(1, 0));
	EXPECT_DOUBLE_EQ(2.0, mat(1, 1));
	EXPECT_DOUBLE_EQ(3.0, mat(2, 0));
	EXPECT_DOUBLE_EQ(1.0, mat(2, 1));
	EXPECT_DOUBLE_EQ(2.0, mat(2, 2));
}
#endif

TEST(CholeskyDecompositionTests, CholeskyDecompositionTest) {
    auto mat = lalib::DynHermiteMat<double>(3, std::vector{
		4.0, 
		2.0, 5.0, 
		6.0, 5.0, 14.0
	});
    auto mat_sq = lalib::DynMat<double>(std::vector{
		4.0, 2.0, 6.0,
		2.0, 5.0, 5.0,
		6.0, 5.0, 14.0
	}, 3, 3);

	{
		auto cholesky = lalib::solver::DynTriCholeskyFactorization<double>(std::move(mat));
		auto l = cholesky.lower_mat();

		EXPECT_DOUBLE_EQ(2.0, l(0, 0));
		EXPECT_DOUBLE_EQ(1.0, l(1, 0));
		EXPECT_DOUBLE_EQ(2.0, l(1, 1));
		EXPECT_DOUBLE_EQ(3.0, l(2, 0));
		EXPECT_DOUBLE_EQ(1.0, l(2, 1));
		EXPECT_DOUBLE_EQ(2.0, l(2, 2));
	}
	{
		auto cholesky = lalib::solver::DynCholeskyFactorization<double>(std::move(mat_sq));
		auto l = cholesky.factor_mat();

		EXPECT_DOUBLE_EQ(2.0, l(0, 0));
		EXPECT_DOUBLE_EQ(1.0, l(1, 0));
		EXPECT_DOUBLE_EQ(2.0, l(1, 1));
		EXPECT_DOUBLE_EQ(3.0, l(2, 0));
		EXPECT_DOUBLE_EQ(1.0, l(2, 1));
		EXPECT_DOUBLE_EQ(2.0, l(2, 2));
	}
}

TEST(CholeskyDecompositionTests, LinearSolverTest) {
    auto mat = lalib::DynHermiteMat<double>(3, std::vector{
		4.0, 
		2.0, 5.0, 
		6.0, 5.0, 14.0
	});
    auto mat_sq = lalib::DynMat<double>(std::vector{
		4.0, 2.0, 6.0,
		2.0, 5.0, 5.0,
		6.0, 5.0, 14.0
	}, 3, 3);

	const auto b = lalib::DynVecD({ 2.0, 5.0, 1.0 });
    auto rslt = b;

	{
		auto cholesky = lalib::solver::DynTriCholeskyFactorization(std::move(mat));
		cholesky.solve_linear_mut(rslt);
		
		EXPECT_DOUBLE_EQ(1.25, rslt[0]);
		EXPECT_DOUBLE_EQ(1.5, rslt[1]);
		EXPECT_DOUBLE_EQ(-1.0, rslt[2]);
	}
	{
		auto cholesky = lalib::solver::DynCholeskyFactorization(std::move(mat_sq));
		auto rslt = cholesky.solve_linear(b);
		
		EXPECT_DOUBLE_EQ(1.25, rslt[0]);
		EXPECT_DOUBLE_EQ(1.5, rslt[1]);
		EXPECT_DOUBLE_EQ(-1.0, rslt[2]);
	}
}

TEST(CholeskyDecompositionTests, MultiLinearSolverTest) {
    auto mat = lalib::DynHermiteMat<double>(3, std::vector{
		4.0, 
		2.0, 5.0, 
		6.0, 5.0, 14.0
	});
    auto mat_sq = lalib::DynMat<double>(std::vector{
		4.0, 2.0, 6.0,
		2.0, 5.0, 5.0,
		6.0, 5.0, 14.0
	}, 3, 3);

	const auto b = lalib::DynMatD({ 
        2.0, 0.0,
        5.0, 2.0, 
        1.0, -1.0
    }, 3, 2);

	{
		auto rslt = b;
		auto cholesky = lalib::solver::DynTriCholeskyFactorization(std::move(mat));
		cholesky.solve_linear_mut(rslt);
		
		EXPECT_DOUBLE_EQ(1.25, rslt(0, 0));
		EXPECT_DOUBLE_EQ(1.5, rslt(1, 0));
		EXPECT_DOUBLE_EQ(-1.0, rslt(2, 0));

		EXPECT_DOUBLE_EQ(0.375, rslt(0, 1));
		EXPECT_DOUBLE_EQ(0.75, rslt(1, 1));
		EXPECT_DOUBLE_EQ(-0.5, rslt(2, 1));
	}
	{
		auto cholesky = lalib::solver::DynCholeskyFactorization(std::move(mat_sq));
		auto rslt = cholesky.solve_linear(b);
		
		EXPECT_DOUBLE_EQ(1.25, rslt(0, 0));
		EXPECT_DOUBLE_EQ(1.5, rslt(1, 0));
		EXPECT_DOUBLE_EQ(-1.0, rslt(2, 0));

		EXPECT_DOUBLE_EQ(0.375, rslt(0, 1));
		EXPECT_DOUBLE_EQ(0.75, rslt(1, 1));
		EXPECT_DOUBLE_EQ(-0.5, rslt(2, 1));
	}
}

TEST(ModCholeskyDecompositionTests, DecompositionTest) {
    auto mat_sq = lalib::DynMat<double>(std::vector{
		4.0, 2.0, 6.0,
		2.0, 5.0, 5.0,
		6.0, 5.0, 14.0
	}, 3, 3);

	{
		auto cholesky = lalib::solver::DynModCholeskyFactorization<double>(std::move(mat_sq));
		auto l = cholesky.factor_mat();

		EXPECT_DOUBLE_EQ(4.0, l(0, 0));
		EXPECT_DOUBLE_EQ(2.0, l(1, 0));
		EXPECT_DOUBLE_EQ(4.0, l(1, 1));
		EXPECT_DOUBLE_EQ(6.0, l(2, 0));
		EXPECT_DOUBLE_EQ(2.0, l(2, 1));
		EXPECT_DOUBLE_EQ(4.0, l(2, 2));
	}
}

TEST(ModCholeskyDecompositionTests, LinearSolverTest) {
    auto mat_sq = lalib::DynMat<double>(std::vector{
		4.0, 2.0, 6.0,
		2.0, 5.0, 5.0,
		6.0, 5.0, 14.0
	}, 3, 3);

	const auto b = lalib::DynVecD({ 2.0, 5.0, 1.0 });
	{
		auto cholesky = lalib::solver::DynModCholeskyFactorization(std::move(mat_sq));
		auto rslt = cholesky.solve_linear(b);
		
		EXPECT_DOUBLE_EQ(1.25, rslt[0]);
		EXPECT_DOUBLE_EQ(1.5, rslt[1]);
		EXPECT_DOUBLE_EQ(-1.0, rslt[2]);
	}
}

TEST(ModCholeskyDecompositionTests, MultiLinearSolverTest) {
    auto mat_sq = lalib::DynMat<double>(std::vector{
		4.0, 2.0, 6.0,
		2.0, 5.0, 5.0,
		6.0, 5.0, 14.0
	}, 3, 3);

	const auto b = lalib::DynMatD({ 
        2.0, 0.0,
        5.0, 2.0, 
        1.0, -1.0
    }, 3, 2);

	{
		auto cholesky = lalib::solver::DynModCholeskyFactorization(std::move(mat_sq));
		auto rslt = cholesky.solve_linear(b);
		
		EXPECT_DOUBLE_EQ(1.25, rslt(0, 0));
		EXPECT_DOUBLE_EQ(1.5, rslt(1, 0));
		EXPECT_DOUBLE_EQ(-1.0, rslt(2, 0));

		EXPECT_DOUBLE_EQ(0.375, rslt(0, 1));
		EXPECT_DOUBLE_EQ(0.75, rslt(1, 1));
		EXPECT_DOUBLE_EQ(-0.5, rslt(2, 1));
	}
}