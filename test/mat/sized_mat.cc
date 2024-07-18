#include "lalib/mat/sized_mat.hpp"
#include <gtest/gtest.h>
#include <stdexcept>

TEST(SizedMatTests, VecConstTest) {
    using MatD24 = lalib::SizedMat<double, 2, 4>;
    using MatD23 = lalib::SizedMat<double, 2, 3>;
    EXPECT_NO_THROW({
        MatD24(std::vector{
            1.0, 2.0, 3.0, 4.0,
            2.0, 1.0, 5.0, 3.0
        });
    });
    EXPECT_THROW({
        MatD23(std::vector{
            1.0, 2.0, 3.0, 4.0,
            2.0, 1.0, 5.0, 3.0
        });
    }, std::runtime_error);
}

TEST(SizedMatTests, InitListConstTest) {
    using MatD24 = lalib::SizedMat<double, 2, 4>;
    using MatD23 = lalib::SizedMat<double, 2, 3>;
    EXPECT_NO_THROW({
        MatD24 ({
            1.0, 2.0, 3.0, 4.0,
            2.0, 1.0, 5.0, 3.0
        });
    });
    EXPECT_THROW({
        MatD23({
            1.0, 2.0, 3.0, 4.0,
            2.0, 1.0, 5.0, 3.0
        });
    }, std::runtime_error);
}

TEST(SizedMatTests, CopyConstTest) {
    auto mat = lalib::SizedMat<double, 2, 4> {
        1.0, 2.0, 3.0, 4.0,
        2.0, 1.0, 5.0, 3.0
    };

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

TEST(SizedMatTests, CopyAssignTest) {
    auto mat = lalib::SizedMat<double, 2, 4> {
        1.0, 2.0, 3.0, 4.0,
        2.0, 1.0, 5.0, 3.0
    };
    auto mat_copy = lalib::SizedMat<double, 2, 4>::uninit();

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

TEST(SizedMatTests, ShapeTest) {
    auto mat = lalib::SizedMat<double, 2, 4> {
        1.0, 2.0, 3.0, 4.0,
        2.0, 1.0, 5.0, 3.0
    };

    ASSERT_EQ(2, mat.shape().first);
    ASSERT_EQ(4, mat.shape().second);
}

TEST(SizedMatTests, FilledTest) {
    auto val = 3.0;
    auto mat = lalib::SizedMat<double, 3, 4>::filled(val);

    for (const auto& c: mat) {
        ASSERT_EQ(val, c);
    }
}

TEST(SizedMatTests, DiagTest) {
    auto mat1 = lalib::SizedMat<double, 3, 3>::diag(2.0);
    auto mat2 = lalib::SizedMat<double, 3, 3>::diag({1.0, 2.0 ,3.0});

    // Uniform Diagonal Matrix
    for (auto i = 0u; i < 3; ++i) {
        for (auto j = 0u; j < 3; ++j) {
            if (i == j) ASSERT_DOUBLE_EQ(2.0, mat1(i, j));
            else        ASSERT_DOUBLE_EQ(0.0, mat1(i, j));
        }
    }
    // Arbitral Diagonal Matrix
    for (auto i = 0u; i < 3; ++i) {
        for (auto j = 0u; j < 3; ++j) {
            if (i == j) ASSERT_DOUBLE_EQ(static_cast<double>(i + 1), mat2(i, j));
            else        ASSERT_DOUBLE_EQ(0.0, mat2(i, j));
        }
    }
}