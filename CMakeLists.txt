cmake_minimum_required(VERSION 3.16)
project(braiNNlet)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Library sources
set(LIB_SOURCES
    src/core/nn/tensor.cpp
    src/core/nn/dense_layer.cpp
    src/core/nn/activations.cpp
    src/core/nn/loss.cpp
    src/core/nn/layer.cpp
    src/core/nn/network.cpp
    src/core/data/Dataset.cpp
    src/core/data/MnistLoader.cpp
    src/core/training/Trainer.cpp
)

# Create the neural network library
add_library(braiNNlet_lib ${LIB_SOURCES})
target_include_directories(braiNNlet_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(braiNNlet_lib Eigen3::Eigen)

# CLI application sources
set(CLI_SOURCES
    src/cli/main.cpp
    src/cli/utils/cli_utils.cpp
    src/cli/utils/config_manager.cpp
    src/cli/commands/create_command.cpp
    src/cli/commands/train_command.cpp
    src/cli/commands/eval_command.cpp
    src/cli/commands/data_command.cpp
    src/cli/interfaces/interactive_mode.cpp
)

# Create the CLI executable
add_executable(braiNNlet-cli ${CLI_SOURCES})
target_include_directories(braiNNlet-cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(braiNNlet-cli braiNNlet_lib Eigen3::Eigen)

# Main demo executable
add_executable(braiNNlet main.cpp)
target_include_directories(braiNNlet PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(braiNNlet braiNNlet_lib Eigen3::Eigen)

# Debug MNIST executable
add_executable(debug_mnist debug_mnist.cpp)
target_include_directories(debug_mnist PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(debug_mnist braiNNlet_lib Eigen3::Eigen)

# Test sources
set(TEST_SOURCES
    tests/main.test.cpp
    tests/tensor.test.cpp
    tests/activations.test.cpp
    tests/loss.test.cpp
    tests/dense_layer.test.cpp
    tests/backpropagation.test.cpp
)

# Create test executable
add_executable(test_braiNNlet ${TEST_SOURCES})
target_include_directories(test_braiNNlet PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_braiNNlet braiNNlet_lib Eigen3::Eigen)
