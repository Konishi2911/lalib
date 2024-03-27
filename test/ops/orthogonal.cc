#include <gtest/gtest.h>
#include <random>
#include "lalib/ops/orthogonal.hpp"
#include "lalib/vec.hpp"

TEST(OrthogonalizationTests, CGSTest) {
    auto vecs = std::vector {
        lalib::VecD<3>({ 1.0, 0.0, 0.0 }),
        lalib::VecD<3>({ 1.0, 1.0, 0.0 }),
        lalib::VecD<3>({ 0.0, 1.0, 1.0 })
    };

    lalib::orth::cgs(vecs);
    
    ASSERT_DOUBLE_EQ(0.0, vecs[0].dot(vecs[1]));
    ASSERT_DOUBLE_EQ(0.0, vecs[0].dot(vecs[2]));
    ASSERT_DOUBLE_EQ(0.0, vecs[1].dot(vecs[2]));
}

TEST(OrthogonalizationTests, CGSRandomTest) {
    auto mt = std::mt19937(std::random_device()());
    auto rng = std::uniform_real_distribution<double>(-0.4, 0.4);
    auto vecs = std::vector {
        lalib::VecD<3>({ 1.0 + rng(mt), rng(mt), rng(mt) }),
        lalib::VecD<3>({ rng(mt), 1.0 + rng(mt), rng(mt) }),
        lalib::VecD<3>({ rng(mt), rng(mt), 1.0 + rng(mt) })
    };

    lalib::orth::cgs(vecs);
    
    ASSERT_NEAR(0.0, vecs[0].dot(vecs[1]), 1e-10);
    ASSERT_NEAR(0.0, vecs[0].dot(vecs[2]), 1e-10);
    ASSERT_NEAR(0.0, vecs[1].dot(vecs[2]), 1e-10);
}