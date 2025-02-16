#include "lalib/mat/sp_mat.hpp"
#include <gtest/gtest.h>
#include <gtest/gtest_pred_impl.h>

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

TEST(SpMatTests, CooUnitMatrixTest) {
    auto unit_mat = lalib::SpMat<double>::unit(3);

    ASSERT_EQ(unit_mat(0, 0), 1.0);
    ASSERT_EQ(unit_mat(0, 1), 0.0);
    ASSERT_EQ(unit_mat(0, 2), 0.0);
    ASSERT_EQ(unit_mat(1, 0), 0.0);
    ASSERT_EQ(unit_mat(1, 1), 1.0);
    ASSERT_EQ(unit_mat(1, 2), 0.0);
    ASSERT_EQ(unit_mat(2, 0), 0.0);
    ASSERT_EQ(unit_mat(2, 1), 0.0);
    ASSERT_EQ(unit_mat(2, 2), 1.0);
}

TEST(SpMatTests, CrsUnitMatrixTest) {
    auto unit_mat = lalib::SpMat<double>::unit(3);

    ASSERT_EQ(unit_mat(0, 0), 1.0);
    ASSERT_EQ(unit_mat(0, 1), 0.0);
    ASSERT_EQ(unit_mat(0, 2), 0.0);
    ASSERT_EQ(unit_mat(1, 0), 0.0);
    ASSERT_EQ(unit_mat(1, 1), 1.0);
    ASSERT_EQ(unit_mat(1, 2), 0.0);
    ASSERT_EQ(unit_mat(2, 0), 0.0);
    ASSERT_EQ(unit_mat(2, 1), 0.0);
    ASSERT_EQ(unit_mat(2, 2), 1.0);
}

TEST(SpMatTests, CooMatrixAddAssignTest) {
    auto mat1 = lalib::SpCooMat<double>::unit(2);
    auto mat2 = lalib::SpCooMat<double> {
        { 1.0, 2.0, 3.0, 4.0 },
        { 0, 0, 1, 1 },
        { 0, 1, 0, 1 }
    };

    mat1 += mat2;

    ASSERT_EQ(mat1(0, 0), 2.0);
    ASSERT_EQ(mat1(0, 1), 2.0);
    ASSERT_EQ(mat1(1, 0), 3.0);
    ASSERT_EQ(mat1(1, 1), 5.0);


    auto mat3 = lalib::SpCooMat<double>::unit(5);
    // mat4 = 
    //  1.0, 2.0, 0.0, 0.0, 0.0
    //  3.0, 0.0, 0.0, 0.0, 0.0
    //  0.0, 4.0, 0.0, 0.0, 0.0
    //  2.0, 0.0, 0.0, 5.0, 0.0
    //  6.0, 0.0, 0.0, 0.0, 0.0
    auto mat4 = lalib::SpCooMat<double> {
        { 1.0, 2.0, 3.0, 4.0, 2.0, 5.0, 6.0 },
        { 0, 0, 1, 2, 3, 3, 4 },
        { 0, 1, 0 , 1, 0, 3, 0 }
    };

    mat3 += mat4;

    ASSERT_EQ(mat3(0, 0), 2.0);
    ASSERT_EQ(mat3(0, 1), 2.0);
    ASSERT_EQ(mat3(0, 2), 0.0);
    ASSERT_EQ(mat3(0, 3), 0.0);
    ASSERT_EQ(mat3(1, 0), 3.0);
    ASSERT_EQ(mat3(1, 1), 1.0);
    ASSERT_EQ(mat3(1, 2), 0.0);
    ASSERT_EQ(mat3(1, 3), 0.0);
    ASSERT_EQ(mat3(2, 0), 0.0);
    ASSERT_EQ(mat3(2, 1), 4.0);
    ASSERT_EQ(mat3(2, 2), 1.0);
    ASSERT_EQ(mat3(2, 3), 0.0);
    ASSERT_EQ(mat3(3, 0), 2.0);
    ASSERT_EQ(mat3(3, 1), 0.0);
    ASSERT_EQ(mat3(3, 2), 0.0);
    ASSERT_EQ(mat3(3, 3), 6.0);
    ASSERT_EQ(mat3(4, 0), 6.0);
    ASSERT_EQ(mat3(4, 1), 0.0);
    ASSERT_EQ(mat3(4, 2), 0.0);
    ASSERT_EQ(mat3(4, 3), 0.0);
    ASSERT_EQ(mat3(4, 4), 1.0);
}