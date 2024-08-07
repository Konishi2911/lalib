cmake_minimum_required(VERSION 3.20)
project(lalib VERSION 0.6.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS True)


if (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang") 
	add_compile_options(-Wall -Wextra -Rpass=vec)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
	add_compile_options(-Wall -Wextra)
else()
	add_compile_options(-Wall -Wextra)
endif()


find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")


include(cmake/backend.cmake)

include_directories(
	${CMAKE_SOURCE_DIR}/include
)

find_package(GTest)
if (${GTest_FOUND}) 
	enable_testing()
	add_subdirectory(test)
endif()

add_subdirectory(benchmark)