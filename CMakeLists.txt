<<<<<<< HEAD
cmake_minimum_required(VERSION 3.16)
project(braiNNlet)
=======
cmake_minimum_required(VERSION 3.25)
project(Brainnlet VERSION 1.0.0 LANGUAGES CXX)
>>>>>>> 0b019000c513a33b3290daba0447d9c38b4ade2d

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

<<<<<<< HEAD
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
=======
# Set executable suffix to .exe for all platforms
set(CMAKE_EXECUTABLE_SUFFIX ".exe")

# Find required packages
find_package(Qt6 REQUIRED COMPONENTS Widgets Charts Concurrent)
find_package(Eigen3 CONFIG REQUIRED)

# Enable automatic MOC, UIC, and RCC for Qt
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Core library (no Qt dependencies)
add_library(core OBJECT
    src/core/nn/Tensor.cpp
    src/core/nn/Layer.cpp
    src/core/nn/DenseLayer.cpp
    src/core/nn/Activations.cpp
    src/core/nn/Loss.cpp
    src/core/nn/Network.cpp
>>>>>>> 0b019000c513a33b3290daba0447d9c38b4ade2d
    src/core/data/Dataset.cpp
    src/core/data/MnistLoader.cpp
    src/core/training/Trainer.cpp
)

<<<<<<< HEAD
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
=======
target_include_directories(core PUBLIC src)
target_link_libraries(core PUBLIC Eigen3::Eigen)

# GUI library
add_library(gui OBJECT
    src/gui/MainWindow.cpp
    src/gui/PlotWidget.cpp
)

target_include_directories(gui PUBLIC src)
target_link_libraries(gui PUBLIC Qt6::Widgets Qt6::Charts Qt6::Concurrent core)

# Console demo executable
add_executable(brainnlet_demo
    src/app/main.cpp
    $<TARGET_OBJECTS:core>
)

target_link_libraries(brainnlet_demo PRIVATE Eigen3::Eigen)

# GUI executable
add_executable(brainnlet
    src/app/gui_main.cpp
    $<TARGET_OBJECTS:core>
    $<TARGET_OBJECTS:gui>
)

target_link_libraries(brainnlet PRIVATE Qt6::Widgets Qt6::Charts Qt6::Concurrent Eigen3::Eigen)

# Tests
enable_testing()
add_subdirectory(tests)
>>>>>>> 0b019000c513a33b3290daba0447d9c38b4ade2d
