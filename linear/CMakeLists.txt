find_package(Torch REQUIRED)

add_executable(linear_regressor
               ${CMAKE_CURRENT_SOURCE_DIR}/linear_regressor.cpp)
target_link_libraries(linear_regressor ${TORCH_LIBRARIES})

add_executable(load_model ${CMAKE_CURRENT_SOURCE_DIR}/load_model.cpp)
target_link_libraries(load_model ${TORCH_LIBRARIES})
