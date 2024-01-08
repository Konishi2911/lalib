## Vectors
add_executable(sized_vec_test vec/sized_vec.cc)
target_link_libraries(sized_vec_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(sized_vec_test)

add_executable(dyn_vec_test vec/dyn_vec.cc)
target_link_libraries(dyn_vec_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(dyn_vec_test)

## Matrices
add_executable(sized_mat_test mat/sized_mat.cc)
target_link_libraries(sized_mat_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(sized_mat_test)

add_executable(dyn_mat_test mat/dyn_mat.cc)
target_link_libraries(dyn_mat_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(dyn_mat_test)



## Vector Operations
add_executable(vec_ops_test ops/vec_ops.cc)
target_link_libraries(vec_ops_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(vec_ops_test)

## Matrix-Vector Operations
add_executable(mat_vec_ops_test ops/mat_vec_ops.cc)
target_link_libraries(mat_vec_ops_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(mat_vec_ops_test)


## Errors
add_executable(error_test err/error.cc)
target_link_libraries(error_test PRIVATE GTest::GTest GTest::Main)
gtest_discover_tests(error_test)