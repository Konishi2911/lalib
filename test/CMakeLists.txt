add_executable(sized_vec_test vec/sized_vec.cc)
target_link_libraries(sized_vec_test PRIVATE GTest::GTest GTest::Main)
gtest_discover_tests(sized_vec_test)