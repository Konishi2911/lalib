#include "lalib/mat/sp_mat.hpp"
#include <gtest/gtest.h>

TEST(SpMatTests, SpCooMatCreationTest) {
    using SpCooMatD = lalib::SpCooMat<double>;

    EXPECT_NO_THROW({
        SpCooMatD(
            { 1.0, 2.0, 3.0, 4.0 },
            { 0, 0, 1, 1 },
            { 0, 1, 0, 1 }
        );
    });

    EXPECT_THROW({
        SpCooMatD(
            { 1.0, 2.0, 3.0, 4.0 },
            { 0, 0, 1, 1 },
            { 0, 1, 0 }
        );
    }, std::runtime_error);

    EXPECT_THROW({
        SpCooMatD(
            { 1.0, 2.0, 3.0, 4.0 },
            { 0, 0, 1, 1 },
            { 0, 1, 0, 1, 2 }
        );
    }, std::runtime_error);

    EXPECT_THROW({
        SpCooMatD(
            { 1.0, 2.0, 3.0, 4.0 },
            { 0, 0, 1, 1, 2 },
            { 0, 1, 0, 1 }
        );
    }, std::runtime_error);

    EXPECT_THROW({
        SpCooMatD(
            { 1.0, 2.0, 3.0, 4.0, 5.0 },
            { 0, 0, 1, 1 },
            { 0, 1, 0, 1 }
        );
    }, std::runtime_error);
}

TEST(SpMatTests, SpCooMatCopyTest) {
    auto mat = lalib::SpCooMat<double> {
        { 1.0, 2.0, 3.0, 4.0 },
        { 0, 0, 1, 1 },
        { 0, 1, 0, 1 }
    };

    auto mat_copy = mat;

    ASSERT_EQ(mat_copy(0, 0), mat(0, 0));
    ASSERT_EQ(mat_copy(0, 1), mat(0, 1));
    ASSERT_EQ(mat_copy(1, 0), mat(1, 0));
    ASSERT_EQ(mat_copy(1, 1), mat(1, 1));
}

TEST(SpMatTests, SpCooMatCopyAssignTest) {
    auto mat = lalib::SpCooMat<double> {
        { 1.0, 2.0, 3.0, 4.0 },
        { 0, 0, 1, 1 },
        { 0, 1, 0, 1 }
    };
    auto mat_copy = lalib::SpCooMat<double> {};

    mat_copy = mat;

    ASSERT_EQ(mat_copy(0, 0), mat(0, 0));
    ASSERT_EQ(mat_copy(0, 1), mat(0, 1));
    ASSERT_EQ(mat_copy(1, 0), mat(1, 0));
    ASSERT_EQ(mat_copy(1, 1), mat(1, 1));
}

TEST(SpMatTests, SpCooMatShapeTest) {
    auto mat = lalib::SpCooMat<double> {
        { 1.0, 2.0, 3.0, 4.0 },
        { 0, 0, 1, 1 },
        { 0, 1, 0, 1 }
    };

    auto shape = mat.shape();

    ASSERT_EQ(shape.first, 2);
    ASSERT_EQ(shape.second, 2);
}

TEST(SpMatTests, SpCooMatNnzTest) {
    auto mat = lalib::SpCooMat<double> {
        { 1.0, 2.0, 3.0, 4.0 },
        { 0, 0, 1, 1 },
        { 0, 1, 0, 1 }
    };

    auto nnz = mat.nnz();

    ASSERT_EQ(nnz, 4);
}