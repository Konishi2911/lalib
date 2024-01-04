#include "../../include/err/error.hpp"
#include <gtest/gtest.h>

TEST(ErrorTests, VecSizeMismatchedErrorTest) {
    EXPECT_THROW(
        { 
            throw vec_error::SizeMismatched(2, 3, "Additional Messages"); 
        }, 
        vec_error::SizeMismatched
    );
}