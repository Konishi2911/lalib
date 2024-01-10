cmake_minimum_required(VERSION 3.20)
project(lalib VERSION 0.2.3 LANGUAGES C CXX)

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


option(LALIB_BACKEND "Backend" "Internal")
if (LALIB_BACKEND MATCHES "Internal")
	add_compile_definitions(LALIB_INTERNAL_BACKEND)
elseif (LALIB_BACKEND MATCHES "BLAS")
	find_package(BLAS REQUIRED)
	add_compile_definitions(LALIB_BLAS_BACKEND)
elseif (LALIB_BACKEND MATCHES "LAPACK")
	find_package(LAPACK REQUIRED)
	add_compile_definitions(LALIB_BLAS_BACKEND LALIB_LAPACK_BACKEND)
endif()

option(LALIB_USE_ACCELERATOR "Accelerator" "FALSE")
if (LALIB_USE_ACCELERATOR)
	if (${OpenMP_CXX_VERSION} VERSION_LESS 4.5)
		message(FATAL_ERROR "OpenMP version 4.5 is required to enable hardware acceleration.")
	endif()
	add_compile_definitions(LALIB_USE_ACCELERATOR)
endif()


include_directories(
	. 
)

find_package(GTest)
if (${GTest_FOUND}) 
	enable_testing()
	add_subdirectory(test)
endif()

add_subdirectory(benchmark)