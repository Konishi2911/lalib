#include "../../include/vec/dyn_vec.hpp"
#include <iterator>
#include <ranges>
#include <algorithm>
#include <gtest/gtest.h>

//static_assert(std::random_access_iterator<typename lalib::SizedVec<uint32_t, 3>::Iter>);
//static_assert(std::random_access_iterator<typename lalib::SizedVec<uint32_t, 3>::ConstIter>);

TEST(DynVecTests, CopyConstTest) {
    auto vec = lalib::DynVec<uint32_t>({0, 1, 2});
    auto vec_copy = vec;

    ASSERT_EQ(vec_copy[0], vec[0]);
    ASSERT_EQ(vec_copy[1], vec[1]);
    ASSERT_EQ(vec_copy[2], vec[2]);
}

TEST(DynVecTests, MoveConstTest) {
    auto vec = lalib::DynVec<uint32_t>({0, 1, 2});
    auto vec_addr = std::vector<const uint32_t*>();
    std::ranges::transform(vec, std::back_inserter(vec_addr), [](const uint32_t& e) { return &e; });
    auto vec_move = std::move(vec);

    ASSERT_EQ(vec_addr[0], &vec_move[0]);
    ASSERT_EQ(vec_addr[1], &vec_move[1]);
    ASSERT_EQ(vec_addr[2], &vec_move[2]);
}

TEST(DynVecTests, CopyAssignTest) {
    auto vec = lalib::DynVec<uint32_t>({0, 1, 2});
    auto vec_copy = lalib::DynVec<uint32_t>({2, 1, 0});
    vec_copy = vec;

    ASSERT_EQ(vec_copy[0], vec[0]);
    ASSERT_EQ(vec_copy[1], vec[1]);
    ASSERT_EQ(vec_copy[2], vec[2]);
}

TEST(DynVecTests, MoveAssignTest) {
    auto vec = lalib::DynVec<uint32_t>({0, 1, 2});
    auto vec_move = lalib::DynVec<uint32_t>({2, 1, 0});
    auto vec_addr = std::vector<const uint32_t*>();
    std::ranges::transform(vec, std::back_inserter(vec_addr), [](const uint32_t& e) { return &e; });
    vec_move = std::move(vec);

    ASSERT_EQ(vec_addr[0], &vec_move[0]);
    ASSERT_EQ(vec_addr[1], &vec_move[1]);
    ASSERT_EQ(vec_addr[2], &vec_move[2]);
}

TEST(DynVecTests, SizeTest) {
    auto vec = lalib::DynVec<uint32_t>::filled(3, 0);
    ASSERT_EQ(3, vec.size());
}

TEST(DynVecTests, FilledTest) {
    auto vec = lalib::DynVec<uint32_t>::filled(3, 1);
    ASSERT_EQ(1, vec[0]);
    ASSERT_EQ(1, vec[1]);
    ASSERT_EQ(1, vec[2]);

    auto vec4 = lalib::DynVec<uint32_t>::filled(4, 1);
    ASSERT_EQ(1, vec4[0]);
    ASSERT_EQ(1, vec4[1]);
    ASSERT_EQ(1, vec4[2]);
    ASSERT_EQ(1, vec4[3]);
}

TEST(DynVecTests, DotTest) {
    auto v1 = lalib::DynVec<double>::filled(4, 1);
    auto v2 = lalib::DynVec<double> ({ 1.1, 2.1, 3.0, 4.0 });

    ASSERT_DOUBLE_EQ(10.2, v1.dot(v2));
}

TEST(DynVecTests, DotTestSizeMismatched) {
    auto v1 = lalib::DynVec<double>::filled(3, 1);
    auto v2 = lalib::DynVec<double> ({ 1.0, 2.0, 3.0, 4.0 });
    ASSERT_THROW(
        {
            v1.dot(v2); 
        }, 
        lalib::vec_error::SizeMismatched
    );
}

TEST(DynVecTests, Norm2Test) {
    auto v = lalib::DynVec<double> ({ 1.0, 2.0, 3.0, 4.0 });

    ASSERT_DOUBLE_EQ(std::sqrt(30.0), v.norm2());
}