cmake_minimum_required(VERSION 3.20)
project(braiNNlet VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use FetchContent to automatically download Eigen3
include(FetchContent)

FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)

FetchContent_MakeAvailable(Eigen3)

# Collect library source files (exclude test files)
file(GLOB_RECURSE LIB_SOURCES "src/**/*.cpp" "src/**/*.hpp")

# Create the library
add_library(braiNNlet STATIC ${LIB_SOURCES})

# Link Eigen3 to the library
target_link_libraries(braiNNlet PUBLIC Eigen3::Eigen)

# Set include directories
target_include_directories(braiNNlet PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Collect all test source files
file(GLOB TEST_SOURCES 
    "tests/test_main.cpp"
    "tests/test_tensor.cpp" 
    "tests/test_activations.cpp"
    "tests/test_loss.cpp"
    "tests/test_dense_layer.cpp"
    "tests/test_backpropagation.cpp"
)

# Create the test executable
add_executable(test ${TEST_SOURCES})
target_link_libraries(test braiNNlet)

# Create the main executable
add_executable(main main.cpp)

target_link_libraries(main braiNNlet)
