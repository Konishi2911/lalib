add_executable(sized_vec_test vec/sized_vec.cc)
target_link_libraries(sized_vec_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(sized_vec_test)

add_executable(dyn_vec_test vec/dyn_vec.cc)
target_link_libraries(dyn_vec_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(dyn_vec_test)

## Vector Operations
add_executable(vec_ops_test ops/vec_ops.cc)
target_link_libraries(vec_ops_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(vec_ops_test)


## Errors
add_executable(error_test err/error.cc)
target_link_libraries(error_test PRIVATE GTest::GTest GTest::Main)
gtest_discover_tests(error_test)