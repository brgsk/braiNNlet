cmake_minimum_required(VERSION 3.25)
project(Brainnlet VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
    src/core/data/Dataset.cpp
    src/core/data/MnistLoader.cpp
    src/core/training/Trainer.cpp
)

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