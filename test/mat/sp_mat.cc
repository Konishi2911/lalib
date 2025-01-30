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


// === CSR Matrix === //

TEST(SpMatTests, SpMatCreationTest) {
    using SpMatD = lalib::SpMat<double>;

    EXPECT_NO_THROW({
        SpMatD(
            { 1.0, 2.0, 3.0, 4.0 },
            { 0, 2, 4 },
            { 0, 1, 0, 1 }
        );
    });

    /* ------------- */
    /* 1.0  2.0  0.0 */
    /* 3.0  0.0  0.0 */
    /* 0.0  4.0  5.0 */
    /* ------------- */
    EXPECT_NO_THROW({
        SpMatD(
            { 1.0, 2.0, 3.0, 4.0, 5.0 },
            { 0, 2, 3, 5 },
            { 0, 1, 0, 1, 2 }
        );
    });

    EXPECT_THROW({
        SpMatD(
            { 1.0, 2.0, 3.0, 4.0 },
            { 0, 2, 4 },
            { 0, 1, 0 }
        );
    }, std::runtime_error);
}

TEST(SpMatTests, SpMatCopyTest) {
    auto mat = lalib::SpMat<double> {
        { 1.0, 2.0, 3.0, 4.0 },
        { 0, 2, 4 },
        { 0, 1, 0, 1 }
    };

    auto mat_copy = mat;

    ASSERT_EQ(mat_copy(0, 0), mat(0, 0));
    ASSERT_EQ(mat_copy(0, 1), mat(0, 1));
    ASSERT_EQ(mat_copy(1, 0), mat(1, 0));
    ASSERT_EQ(mat_copy(1, 1), mat(1, 1));
}

TEST(SpMatTests, SpMatCopyAssignTest) {
    auto mat = lalib::SpMat<double> {
        { 1.0, 2.0, 3.0, 4.0 },
        { 0, 2, 4 },
        { 0, 1, 0, 1 }
    };
    auto mat_copy = lalib::SpMat<double> {};

    mat_copy = mat;

    ASSERT_EQ(mat_copy(0, 0), mat(0, 0));
    ASSERT_EQ(mat_copy(0, 1), mat(0, 1));
    ASSERT_EQ(mat_copy(1, 0), mat(1, 0));
    ASSERT_EQ(mat_copy(1, 1), mat(1, 1));
}

TEST(SpMatTests, SpMatShapeTest) {
    auto mat = lalib::SpMat<double>(
        { 1.0, 2.0, 3.0, 4.0, 5.0 },
        { 0, 2, 3, 5 },
        { 0, 1, 0, 1, 2 }
    );

    auto shape = mat.shape();

    ASSERT_EQ(shape.first, 3);
    ASSERT_EQ(shape.second, 3);
}

TEST(SpMatTests, SpMatNnzTest) {
    auto mat = lalib::SpMat<double> {
        { 1.0, 2.0, 3.0, 4.0 },
        { 0, 2, 4 },
        { 0, 1, 0, 1 }
    };

    auto nnz = mat.nnz();

    ASSERT_EQ(nnz, 4);
}

TEST(SpMatTests, CooCrsConversionTest) {
    auto coo_mat = lalib::SpCooMat<double> {
        { 1.0, 2.0, 3.0, 4.0, 5.0 },
        { 0, 0, 1, 2, 2 },
        { 0, 1, 0, 1, 2 }
    };

    auto crs_mat = lalib::SpMat<double>(coo_mat);
    auto crs_mat_moved = lalib::SpMat<double>(std::move(coo_mat));

    ASSERT_EQ(crs_mat(0, 0), 1.0);
    ASSERT_EQ(crs_mat(0, 1), 2.0);
    ASSERT_EQ(crs_mat(0, 2), 0.0);
    ASSERT_EQ(crs_mat(1, 0), 3.0);
    ASSERT_EQ(crs_mat(1, 1), 0.0);
    ASSERT_EQ(crs_mat(1, 2), 0.0);
    ASSERT_EQ(crs_mat(2, 0), 0.0);
    ASSERT_EQ(crs_mat(2, 1), 4.0);
    ASSERT_EQ(crs_mat(2, 2), 5.0);

    ASSERT_EQ(crs_mat_moved(0, 0), 1.0);
    ASSERT_EQ(crs_mat_moved(0, 1), 2.0);
    ASSERT_EQ(crs_mat_moved(0, 2), 0.0);
    ASSERT_EQ(crs_mat_moved(1, 0), 3.0);
    ASSERT_EQ(crs_mat_moved(1, 1), 0.0);
    ASSERT_EQ(crs_mat_moved(1, 2), 0.0);
    ASSERT_EQ(crs_mat_moved(2, 0), 0.0);
    ASSERT_EQ(crs_mat_moved(2, 1), 4.0);
    ASSERT_EQ(crs_mat_moved(2, 2), 5.0);
}