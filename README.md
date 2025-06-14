# braiNNlet - Interactive Neural Network Explorer

An interactive desktop application for building, training, and visualizing deep neural networks built with Qt6, C++20, and Eigen3.

## Project Overview

braiNNlet provides an intuitive GUI interface for creating neural networks layer-by-layer, training them on datasets like MNIST, and visualizing the training process in real-time.

### Key Features

- **Interactive Network Builder**: Add/remove layers with different activation functions
- **Real-time Training Visualization**: Live plots of loss and accuracy metrics
- **Dataset Support**: MNIST integration
- **Modern C++ Implementation**: Built with C++20 features and best practices

## Project Structure

```
braiNNlet/
├── CMakeLists.txt              # Main CMake configuration
├── README.md                   # This file
├── src/
│   ├── core/                   # Core neural network library (no Qt dependencies)
│   │   ├── nn/                 # Neural network components
│   │   │   ├── Tensor.hpp/cpp  # Matrix operations with Eigen
│   │   │   ├── Layer.hpp/cpp   # Abstract layer base class
│   │   │   ├── DenseLayer.hpp/cpp # Fully connected layer
│   │   │   ├── Activations.hpp/cpp # Activation functions
│   │   │   ├── Loss.hpp/cpp    # Loss functions
│   │   │   └── Network.hpp/cpp # Complete network class
│   │   ├── data/               # Dataset loading
│   │   │   ├── Dataset.hpp/cpp # Base dataset class
│   │   │   └── MnistLoader.hpp/cpp # MNIST dataset
│   │   └── training/           # Training infrastructure
│   │       └── Trainer.hpp/cpp # Training loop with callbacks
│   ├── gui/                    # Qt GUI components
│   │   ├── MainWindow.hpp/cpp  # Main application window
│   │   └── PlotWidget.hpp/cpp  # Training plots
│   └── app/
│       ├── main.cpp            # Application entry point
│       └── gui_main.cpp        # GUI application entry point
├── tests/                      # Comprehensive test suite
│   ├── CMakeLists.txt          # Test configuration
│   ├── main.test.cpp           # Test runner and orchestration
│   ├── tensor.test.cpp         # Tensor class tests
│   ├── activations.test.cpp    # Activation function tests
│   ├── loss.test.cpp           # Loss function tests
│   ├── dense_layer.test.cpp    # Dense layer tests
│   └── integration.test.cpp    # End-to-end integration tests
└── build/                      # Build directory (generated)
```

## Architecture

### Core Design Principles

1. **Separation of Concerns**: Core neural network logic is independent of GUI
2. **Modern C++**: Uses C++20 features, RAII, smart pointers
3. **Type Safety**: Strong typing with STL containers (`std::vector<std::string>`)
4. **Performance**: Eigen3 for optimized matrix operations
5. **Testability**: Unit tests for core functionality

### Neural Network Components

- **Tensor**: Wrapper around Eigen matrices with NN-specific operations
- **Layer**: Abstract base class for all layer types
- **DenseLayer**: Fully connected layer with configurable activation
- **Activations**: ReLU, Sigmoid, Tanh, Linear functions
- **Loss Functions**: MSE, CrossEntropy, BinaryCrossEntropy
- **Network**: Container for layers with forward/backward passes
- **Trainer**: Training loop with metrics and callbacks

## Dependencies

### Required

- **CMake** ≥ 3.25
- **C++20** compatible compiler (GCC 10+, Clang 12+, MSVC 2019+)
- **Qt6** (Widgets, Charts, Concurrent)
- **Eigen3** (Linear algebra library)

## Building

Ensure Qt6 and Eigen3 are installed on your system:

```bash
# Ubuntu/Debian
sudo apt install qt6-base-dev libqt6charts6-dev libeigen3-dev

# macOS with Homebrew
brew install qt6 eigen

# Build
cmake -B build -S .
cmake --build build --config Release
```

## Usage

### Console Demo

The current implementation includes a console demo application:

```bash
./build/brainnlet_demo.exe
```

which showcases the core functionality of the library.

### GUI Application

Run the application with:

```bash
./build/brainnlet.exe
```

which showcases the core functionality of the library.

## Development Status

### ✅ Completed

- Core neural network library (Tensor, Layer, Network classes)
- Training infrastructure with callbacks
- Dataset loading framework
- Basic console application
- Qt GUI implementation
- Real-time loss and accuracy plotting
- Project structure and build system
- **Comprehensive test suite** with test cases covering all core functionality

### 🚧 In Progress

- Network visualization

### 📋 Planned

- Add more datasets
- More layer types (Convolutional, LSTM)
- Advanced optimizers (Adam, RMSprop)
- Model serialization/loading
- Batch normalization
- Dropout layers

## Testing

braiNNlet includes a comprehensive test suite that validates all core functionality with individual test cases across 5 test categories.

### Test Suite Overview

The test suite is designed to thoroughly validate the neural network library:

- **Tensor Tests** (10 categories): Matrix operations, element access, mathematical operations, serialization
- **Activation Function Tests** (7 categories): ReLU, Sigmoid, Tanh, Linear functions with edge cases
- **Loss Function Tests** (7 categories): MSE, Binary/Multi-class Cross Entropy with numerical stability
- **Dense Layer Tests** (9 categories): Forward/backward passes, gradient computation, parameter updates
- **Integration Tests** (8 categories): End-to-end training, batch processing, gradient flow

### Running Tests

```bash
# Build and run all tests
cd build
cmake --build . --target test_core
./build/test_core.exe
```

### Test Coverage

The test suite validates:

#### Core Functionality

- **Matrix Operations**: Addition, subtraction, multiplication, transpose
- **Element Access**: Const/non-const operators, bounds checking
- **Memory Management**: Object lifecycle, large tensor handling
- **Serialization**: Vector conversion with proper storage order

#### Neural Network Components

- **Activation Functions**: Forward/backward passes, gradient computation
- **Loss Functions**: All loss types with batch processing and edge cases
- **Dense Layers**: Parameter initialization, gradient flow, weight updates
- **Training Loop**: Multi-layer networks, convergence validation

#### Edge Cases & Robustness

- **Numerical Stability**: Extreme values, overflow/underflow handling
- **Error Handling**: Dimension mismatches, invalid inputs
- **Batch Processing**: Multiple samples, gradient accumulation
- **Empty/Single Element**: Boundary conditions

### Adding New Tests

When extending the library, add corresponding tests:

```cpp
// Example test structure
void testNewFeature() {
    printf("\n=== Testing New Feature ===\n");
    
    // Setup test data
    // Execute functionality
    // Validate results with assertions
    
    printf("✓ New feature test passed\n");
}
```

Tests should be added to the appropriate test file and included in the main test runner.

## Contributing

This is an academic project. Key areas for enhancement:

1. **GUI Development**: Complete the Qt interface implementation
2. **Visualization**: Add network topology and weight visualization
3. **Performance**: Optimize matrix operations and memory usage
4. **Features**: Add more layer types and training algorithms
5. **Testing**: Extend test coverage for new features and GUI components

## Implementation Notes

### Code Style

- Use modern C++ features (auto, range-based loops, smart pointers)
- Prefer explicit over implicit (clear variable names, no magic numbers)
- RAII for resource management
- Early returns to reduce nesting

### Error Handling

- Input validation at function boundaries
- Specific exception types with context
- Graceful degradation where possible

### Performance Considerations

- Eigen3 for vectorized operations
- Move semantics for large objects
- Minimal copying of tensor data
- Efficient batch processing

## License

This project is developed as part of Programming II coursework at Silesian University of Technology.

## Troubleshooting

### Common Build Issues

- **Qt6 not found**: Ensure Qt6 is in your PATH or use `-DCMAKE_PREFIX_PATH=/path/to/qt6`
- **Eigen3 missing**: Install via package manager

### Runtime Issues

- **Dataset loading fails**: Check that MNIST data files are available in core/data/MNIST/
- **GUI doesn't start**: Verify Qt6 runtime libraries are available
- **Tests fail**: Ensure all dependencies are properly linked and Eigen3 is available

### Test Issues

- **Assertion failures**: Check that the implementation matches expected behavior
- **Compilation errors**: Verify all test files have correct include paths
- **Numerical precision**: Some tests use tolerance-based comparisons (1e-10) for floating-point operations
