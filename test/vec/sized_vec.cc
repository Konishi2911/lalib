#include "../../include/vec/sized_vec.hpp"
#include <iterator>
#include <ranges>
#include <gtest/gtest.h>

//static_assert(std::random_access_iterator<typename lalib::SizedVec<uint32_t, 3>::Iter>);
//static_assert(std::random_access_iterator<typename lalib::SizedVec<uint32_t, 3>::ConstIter>);

TEST(SizedVecTests, CopyConstTest) {
    auto vec = lalib::SizedVec<uint32_t, 3>({0, 1, 2});
    auto vec_copy = vec;

    ASSERT_EQ(vec_copy[0], vec[0]);
    ASSERT_EQ(vec_copy[1], vec[1]);
    ASSERT_EQ(vec_copy[2], vec[2]);
}

TEST(SizedVecTests, CopyAssignTest) {
    auto vec = lalib::SizedVec<uint32_t, 3>({0, 1, 2});
    auto vec_copy = lalib::SizedVec<uint32_t, 3>({2, 1, 0});
    vec_copy = vec;

    ASSERT_EQ(vec_copy[0], vec[0]);
    ASSERT_EQ(vec_copy[1], vec[1]);
    ASSERT_EQ(vec_copy[2], vec[2]);
}

TEST(SizedVecTests, SizeTest) {
    auto vec = lalib::SizedVec<uint32_t, 3>::filled(0);
    ASSERT_EQ(3, vec.size());
}

TEST(SizedVecTests, FilledTest) {
    auto vec = lalib::SizedVec<uint32_t, 3>::filled(1);
    ASSERT_EQ(1, vec[0]);
    ASSERT_EQ(1, vec[1]);
    ASSERT_EQ(1, vec[2]);

    auto vec4 = lalib::SizedVec<uint32_t, 4>::filled(1);
    ASSERT_EQ(1, vec4[0]);
    ASSERT_EQ(1, vec4[1]);
    ASSERT_EQ(1, vec4[2]);
    ASSERT_EQ(1, vec4[3]);
}

TEST(SizedVecTets, DotTest) {
    auto v1 = lalib::SizedVec<double, 4>::filled(1);
    auto v2 = lalib::SizedVec<double, 4> ({ 1.0, 2.0, 3.0, 4.0 });

    ASSERT_DOUBLE_EQ(10.0, v1.dot(v2));
}

TEST(SizedVecTets, Norm2Test) {
    auto v = lalib::SizedVec<double, 4> ({ 1.0, 2.0, 3.0, 4.0 });

    ASSERT_DOUBLE_EQ(std::sqrt(30), v.norm2());
}