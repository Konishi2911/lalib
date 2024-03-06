#include "lalib/err/error.hpp"
#include <gtest/gtest.h>

TEST(ErrorTests, VecSizeMismatchedErrorTest) {
    EXPECT_THROW(
        { 
            throw lalib::vec_error::SizeMismatched(2, 3, "Additional Messages"); 
        }, 
        lalib::vec_error::SizeMismatched
    );
}