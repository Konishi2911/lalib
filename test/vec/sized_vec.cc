#include "../../include/vec/sized_vec.hpp"
#include <iterator>
#include <gtest/gtest.h>

//static_assert(std::random_access_iterator<typename lalib::SizedVec<uint32_t, 3>::Iter>);
//static_assert(std::random_access_iterator<typename lalib::SizedVec<uint32_t, 3>::ConstIter>);

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