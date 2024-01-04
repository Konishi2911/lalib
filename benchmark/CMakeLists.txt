add_executable(vec_bench vec.cc)
target_link_libraries(vec_bench PRIVATE ${OpenMP_CXX_LIBRARIES})