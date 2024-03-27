## Vectors
add_executable(lalib_sized_vec_test vec/sized_vec.cc)
target_link_libraries(lalib_sized_vec_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(lalib_sized_vec_test)

add_executable(lalib_dyn_vec_test vec/dyn_vec.cc)
target_link_libraries(lalib_dyn_vec_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(lalib_dyn_vec_test)

## Matrices
add_executable(lalib_sized_mat_test mat/sized_mat.cc)
target_link_libraries(lalib_sized_mat_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(lalib_sized_mat_test)

add_executable(lalib_dyn_mat_test mat/dyn_mat.cc)
target_link_libraries(lalib_dyn_mat_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(lalib_dyn_mat_test)



## Vector Operations
add_executable(lalib_vec_ops_test ops/vec_ops.cc)
target_link_libraries(lalib_vec_ops_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(lalib_vec_ops_test)

## Matrix-Vector Operations
add_executable(lalib_mat_vec_ops_test ops/mat_vec_ops.cc)
target_link_libraries(lalib_mat_vec_ops_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(lalib_mat_vec_ops_test)

## Matrix-Matrix Operations
add_executable(lalib_mat_mat_ops_test ops/mat_mat_ops.cc)
target_link_libraries(lalib_mat_mat_ops_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(lalib_mat_mat_ops_test)

## Vector Orthogonalization
add_executable(lalib_orth_test ops/orthogonal.cc)
target_link_libraries(lalib_orth_test PRIVATE ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} GTest::GTest GTest::Main)
gtest_discover_tests(lalib_orth_test)


## Solvers
add_executable(lalib_tri_diag_test solver/tri_diag.cc)
target_link_libraries(lalib_tri_diag_test PRIVATE 
    $<$<BOOL:${LAPACK_FOUND}>:LAPACK::LAPACK> 
    $<$<BOOL:${LAPACK_FOUND}>:-llapacke>
    ${BLAS_LIBRARIES} ${OpenMP_CXX_LIBRARIES} 
    GTest::GTest GTest::Main
)
gtest_discover_tests(lalib_tri_diag_test)


## Errors
add_executable(lalib_error_test err/error.cc)
target_link_libraries(lalib_error_test PRIVATE GTest::GTest GTest::Main)
gtest_discover_tests(lalib_error_test)