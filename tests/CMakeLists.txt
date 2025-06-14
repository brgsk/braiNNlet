# Tests configuration
find_package(Qt6 REQUIRED COMPONENTS Test)

# Test executable with all current implementation tests
add_executable(test_core
    main.test.cpp
    tensor.test.cpp
    activations.test.cpp
    loss.test.cpp
    dense_layer.test.cpp
    integration.test.cpp
)

target_link_libraries(test_core PRIVATE 
    core 
    Qt6::Test 
    Eigen3::Eigen
)

target_include_directories(test_core PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Add test
add_test(NAME CoreTests COMMAND test_core) 