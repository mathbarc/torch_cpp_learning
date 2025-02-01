cmake_minimum_required(VERSION 3.25.1)
project(torch_cpp_learning VERSION 0.0.1)


find_package(Torch REQUIRED)

add_executable(linear_regressor sources/linear_regressor.cpp)
target_link_libraries(linear_regressor ${TORCH_LIBRARIES})

add_executable(load_model sources/load_model.cpp)
target_link_libraries(load_model ${TORCH_LIBRARIES})
