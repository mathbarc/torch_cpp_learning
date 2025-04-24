find_package(Torch REQUIRED)
set(UTILS_INCLUDE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    CACHE PATH "Utils lib path")
include_directories(${UTILS_INCLUDE_DIR})
file(GLOB UTILS_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)
file(GLOB UTILS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)

add_library(utils STATIC ${UTILS_SOURCES} ${UTILS_HEADERS})
target_link_libraries(utils ${TORCH_LIBRARIES})
