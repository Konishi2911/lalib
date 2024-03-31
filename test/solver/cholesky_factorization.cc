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

	auto cholesky = lalib::solver::DynTriCholeskyFactorization<double>(std::move(mat));
    auto l = cholesky.lower_matrix();

	EXPECT_DOUBLE_EQ(2.0, l(0, 0));
	EXPECT_DOUBLE_EQ(1.0, l(1, 0));
	EXPECT_DOUBLE_EQ(2.0, l(1, 1));
	EXPECT_DOUBLE_EQ(3.0, l(2, 0));
	EXPECT_DOUBLE_EQ(1.0, l(2, 1));
	EXPECT_DOUBLE_EQ(2.0, l(2, 2));
}

TEST(CholeskyDecompositionTests, LinearSolverTest) {
    auto mat = lalib::DynHermiteMat<double>(3, std::vector{
		4.0, 
		2.0, 5.0, 
		6.0, 5.0, 14.0
	});
	auto b = lalib::DynVecD({ 2.0, 5.0, 1.0 });
    auto rslt = lalib::DynVecD::filled(3, 0.0);

	auto cholesky = lalib::solver::DynTriCholeskyFactorization(std::move(mat));
    cholesky.solve_linear(b, rslt);
	
	EXPECT_DOUBLE_EQ(1.25, rslt[0]);
	EXPECT_DOUBLE_EQ(1.5, rslt[1]);
	EXPECT_DOUBLE_EQ(-1.0, rslt[2]);
}

TEST(CholeskyDecompositionTests, MultiLinearSolverTest) {
    auto mat = lalib::DynHermiteMat<double>(3, std::vector{
		4.0, 
		2.0, 5.0, 
		6.0, 5.0, 14.0
	});
	auto b = lalib::DynMatD({ 
        2.0, 0.0,
        5.0, 2.0, 
        1.0, -1.0
    }, 3, 2);
    auto rslt = lalib::DynMatD::filled(0.0, 3, 2);

	auto cholesky = lalib::solver::DynTriCholeskyFactorization(std::move(mat));
    cholesky.solve_linear(b, rslt);
	
	EXPECT_DOUBLE_EQ(1.25, rslt(0, 0));
	EXPECT_DOUBLE_EQ(1.5, rslt(1, 0));
	EXPECT_DOUBLE_EQ(-1.0, rslt(2, 0));

	EXPECT_DOUBLE_EQ(0.375, rslt(0, 1));
	EXPECT_DOUBLE_EQ(0.75, rslt(1, 1));
	EXPECT_DOUBLE_EQ(-0.5, rslt(2, 1));
}