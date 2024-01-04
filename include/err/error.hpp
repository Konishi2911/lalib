#pragma once
#include <cstddef>
#include <stdexcept>
#include <string>

namespace vec_error {

/// @brief Represents an erorr due to the size mismatch of two vectors. 
struct SizeMismatched: public std::runtime_error {
    SizeMismatched(size_t n1, size_t n2, std::string message = "") noexcept;
};

SizeMismatched::SizeMismatched(size_t n1, size_t n2, std::string message) noexcept:
    runtime_error::runtime_error(
        std::string("VecError::SizeMismatched: The number of elements in two vectors does not matched.\n") + 
        std::string("|- number of elements in vec1: ") + std::to_string(n1) + "\n" +
        std::string("|- number of elements in vec2: ") + std::to_string(n2) + "\n" +
        message
    )
{ }

}
