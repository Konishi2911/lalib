add_executable(sized_vec_test vec/sized_vec.cc)
target_link_libraries(sized_vec_test PRIVATE GTest::GTest GTest::Main)
gtest_discover_tests(sized_vec_test)

add_executable(dyn_vec_test vec/dyn_vec.cc)
target_link_libraries(dyn_vec_test PRIVATE GTest::GTest GTest::Main)
gtest_discover_tests(dyn_vec_test)