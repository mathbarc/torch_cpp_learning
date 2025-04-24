find_package(Torch REQUIRED)
include_directories(${UTILS_INCLUDE_DIR})
add_executable(fc_classifier ${CMAKE_CURRENT_SOURCE_DIR}/fc_classifier.cpp)
target_link_libraries(fc_classifier ${TORCH_LIBRARIES} utils)
