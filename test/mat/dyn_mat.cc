#include "lalib/mat/dyn_mat.hpp"
#include <gtest/gtest.h>

TEST(DynMatTests, CopyConstTest) {
    auto mat = lalib::DynMat<double>({
        1.0, 2.0, 3.0, 4.0,
        2.0, 1.0, 5.0, 3.0
    }, 2, 4);

    auto mat_copy = mat;

    ASSERT_EQ(mat_copy(0, 0), mat(0, 0));
    ASSERT_EQ(mat_copy(0, 1), mat(0, 1));
    ASSERT_EQ(mat_copy(0, 2), mat(0, 2));
    ASSERT_EQ(mat_copy(0, 3), mat(0, 3));
    ASSERT_EQ(mat_copy(1, 0), mat(1, 0));
    ASSERT_EQ(mat_copy(1, 1), mat(1, 1));
    ASSERT_EQ(mat_copy(1, 2), mat(1, 2));
    ASSERT_EQ(mat_copy(1, 3), mat(1, 3));
}

TEST(DynMatTests, CopyAssignTest) {
    auto mat = lalib::DynMat<double>({
        1.0, 2.0, 3.0, 4.0,
        2.0, 1.0, 5.0, 3.0
    }, 2, 4);
    auto mat_copy = lalib::DynMat<double>::uninit(2, 4);

    mat_copy = mat;

    ASSERT_EQ(mat_copy(0, 0), mat(0, 0));
    ASSERT_EQ(mat_copy(0, 1), mat(0, 1));
    ASSERT_EQ(mat_copy(0, 2), mat(0, 2));
    ASSERT_EQ(mat_copy(0, 3), mat(0, 3));
    ASSERT_EQ(mat_copy(1, 0), mat(1, 0));
    ASSERT_EQ(mat_copy(1, 1), mat(1, 1));
    ASSERT_EQ(mat_copy(1, 2), mat(1, 2));
    ASSERT_EQ(mat_copy(1, 3), mat(1, 3));
}

TEST(DynMatTests, ShapeTest) {
    auto mat = lalib::DynMat<double>({
        1.0, 2.0, 3.0, 4.0,
        2.0, 1.0, 5.0, 3.0
    }, 2, 4);

    ASSERT_EQ(2, mat.shape().first);
    ASSERT_EQ(4, mat.shape().second);
}

TEST(DynMatTests, FilledTest) {
    auto val = 3.0;
    auto mat = lalib::DynMat<double>::filled(val, 3, 4);

    for (const auto& c: mat) {
        ASSERT_EQ(val, c);
    }
}

TEST(SizedMatTests, DiagTest) {
    auto n_rows = 5u;
    auto mat1 = lalib::DynMat<double>::diag(2.0, n_rows);
    auto mat2 = lalib::DynMat<double>::diag({1.0, 2.0 ,3.0, 4.0, 5.0});

    // Uniform Diagonal Matrix
    for (auto i = 0u; i < n_rows; ++i) {
        for (auto j = 0u; j < n_rows; ++j) {
            if (i == j) ASSERT_DOUBLE_EQ(2.0, mat1(i, j));
            else        ASSERT_DOUBLE_EQ(0.0, mat1(i, j));
        }
    }
    // Arbitral Diagonal Matrix
    for (auto i = 0u; i < n_rows; ++i) {
        for (auto j = 0u; j < n_rows; ++j) {
            if (i == j) ASSERT_DOUBLE_EQ(static_cast<double>(i + 1), mat2(i, j));
            else        ASSERT_DOUBLE_EQ(0.0, mat2(i, j));
        }
    }
}