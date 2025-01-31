#include "lalib/mat/dyn_mat.hpp"
#include <gtest/gtest.h>

TEST(DynMatTests, CopyConstTest) {
    auto mat = lalib::DynMat<double>(2, 4, {
        1.0, 2.0, 3.0, 4.0,
        2.0, 1.0, 5.0, 3.0
    });

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
    auto mat = lalib::DynMat<double>(2, 4, {
        1.0, 2.0, 3.0, 4.0,
        2.0, 1.0, 5.0, 3.0
    });
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
    auto mat = lalib::DynMat<double>(2, 4, {
        1.0, 2.0, 3.0, 4.0,
        2.0, 1.0, 5.0, 3.0
    });

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

TEST(DynMatTests, DiagTest) {
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

TEST(DynMatTests, HessenbergMatTest) {
    /*
        * 1.0 3.0 6.0
        * 2.0 4.0 7.0
        * 0.0 5.0 8.0 
    */
    const auto mat = lalib::HessenbergMat<double>(
        std::vector<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0})
    );

    ASSERT_DOUBLE_EQ(1.0, mat(0, 0));
    ASSERT_DOUBLE_EQ(3.0, mat(0, 1));
    ASSERT_DOUBLE_EQ(6.0, mat(0, 2));
    ASSERT_DOUBLE_EQ(2.0, mat(1, 0));
    ASSERT_DOUBLE_EQ(4.0, mat(1, 1));
    ASSERT_DOUBLE_EQ(7.0, mat(1, 2));
    ASSERT_DOUBLE_EQ(0.0, mat(2, 0));
    ASSERT_DOUBLE_EQ(5.0, mat(2, 1));
    ASSERT_DOUBLE_EQ(8.0, mat(2, 2));
}

TEST(DynMatTests, HessenbergMatExtendTest) {
    auto hess1 = lalib::HessenbergMat<double>({0.0});

    hess1.extend_with_zero();

    ASSERT_DOUBLE_EQ(2, hess1.shape().first);
    ASSERT_DOUBLE_EQ(2, hess1.shape().second);

    hess1.extend_with({1.0, 2.0, 3.0, 4.0});
    ASSERT_DOUBLE_EQ(3, hess1.shape().first);
    ASSERT_DOUBLE_EQ(3, hess1.shape().second);


    const auto hess = hess1;

    ASSERT_DOUBLE_EQ(0.0, hess(0, 0));
    ASSERT_DOUBLE_EQ(0.0, hess(0, 1));
    ASSERT_DOUBLE_EQ(2.0, hess(0, 2));
    ASSERT_DOUBLE_EQ(0.0, hess(1, 0));
    ASSERT_DOUBLE_EQ(0.0, hess(1, 1));
    ASSERT_DOUBLE_EQ(3.0, hess(1, 2));
    ASSERT_DOUBLE_EQ(0.0, hess(2, 0));
    ASSERT_DOUBLE_EQ(1.0, hess(2, 1));
    ASSERT_DOUBLE_EQ(4.0, hess(2, 2));
}

TEST(DynMatTests, HessenbergMatGetColTest) {
    auto hess = lalib::HessenbergMat<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});

    auto col0 = hess.get_col(0);
    ASSERT_EQ(2, col0.size());
    ASSERT_DOUBLE_EQ(1.0, col0[0]);
    ASSERT_DOUBLE_EQ(2.0, col0[1]);

    auto col1 = hess.get_col(1);
    ASSERT_EQ(3, col1.size());
    ASSERT_DOUBLE_EQ(3.0, col1[0]);
    ASSERT_DOUBLE_EQ(4.0, col1[1]);
    ASSERT_DOUBLE_EQ(5.0, col1[2]);

    auto col2 = hess.get_col(2);
    ASSERT_EQ(3, col2.size());
    ASSERT_DOUBLE_EQ(6.0, col2[0]);
    ASSERT_DOUBLE_EQ(7.0, col2[1]);
    ASSERT_DOUBLE_EQ(8.0, col2[2]);
}

TEST(DynMatTests, DynUpperTriMatTest) {
    /*
        * 1.0 2.0 4.0
        * 0.0 3.0 5.0
        * 0.0 0.0 6.0 
    */
    const auto mat = lalib::DynUpperTriMat<double>(
        std::vector<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
    );

    ASSERT_DOUBLE_EQ(1.0, mat(0, 0));
    ASSERT_DOUBLE_EQ(2.0, mat(0, 1));
    ASSERT_DOUBLE_EQ(4.0, mat(0, 2));
    ASSERT_DOUBLE_EQ(0.0, mat(1, 0));
    ASSERT_DOUBLE_EQ(3.0, mat(1, 1));
    ASSERT_DOUBLE_EQ(5.0, mat(1, 2));
    ASSERT_DOUBLE_EQ(0.0, mat(2, 0));
    ASSERT_DOUBLE_EQ(0.0, mat(2, 1));
    ASSERT_DOUBLE_EQ(6.0, mat(2, 2));
}

TEST(DynMatTests, DynUpperTriMatExtendTest) {
    auto tri1 = lalib::DynUpperTriMat<double>({0.0});

    tri1.extend_with_zero();

    ASSERT_DOUBLE_EQ(2, tri1.shape().first);
    ASSERT_DOUBLE_EQ(2, tri1.shape().second);

    tri1.extend_with(std::vector<double>({1.0, 2.0, 3.0}));
    ASSERT_DOUBLE_EQ(3, tri1.shape().first);
    ASSERT_DOUBLE_EQ(3, tri1.shape().second);


    const auto tri = tri1;
    ASSERT_DOUBLE_EQ(0.0, tri(0, 0));
    ASSERT_DOUBLE_EQ(0.0, tri(0, 1));
    ASSERT_DOUBLE_EQ(1.0, tri(0, 2));
    ASSERT_DOUBLE_EQ(0.0, tri(1, 0));
    ASSERT_DOUBLE_EQ(0.0, tri(1, 1));
    ASSERT_DOUBLE_EQ(2.0, tri(1, 2));
    ASSERT_DOUBLE_EQ(0.0, tri(2, 0));
    ASSERT_DOUBLE_EQ(0.0, tri(2, 1));
    ASSERT_DOUBLE_EQ(3.0, tri(2, 2));
}

TEST(DynMatTests, DynUpperTriMatBackSubTest) {
    auto tri = lalib::DynUpperTriMat<double>({1.0, 0.5, 1.0, 0.5, 2.0, 1.0});
    auto y = std::vector<double>({4.5, 4.0, 1.0});

    tri.back_sub(y);

    ASSERT_DOUBLE_EQ(3.0, y[0]);
    ASSERT_DOUBLE_EQ(2.0, y[1]);
    ASSERT_DOUBLE_EQ(1.0, y[2]);
}