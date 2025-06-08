# braiNNlet - a lightweight C++ neural network library

A lightweight C++ neural network library built from scratch using Eigen for linear algebra operations. This project provides fundamental building blocks for creating and training neural networks.

## Features

- **Tensor Operations**: Custom tensor class with support for basic mathematical operations
- **Neural Network Layers**: Configurable dense layers with forward pass implementation
- **Eigen Integration**: High-performance linear algebra operations using Eigen library
- **Modern C++**: Written in C++17 with proper memory management and move semantics
- **Batch Processing**: Support for batch forward passes through neural network layers

## Project Structure

```
braiNNlet/
├── src/
│   └── core/
│       └── nn/
│           ├── tensor.hpp/cpp    # Tensor data structure and operations
│           ├── layer.hpp/cpp     # Neural network layer implementation
│           └── neuron.hpp/cpp    # Individual neuron implementation
├── tests/
│   └── test.cpp                  # Unit tests
├── CMakeLists.txt               # Build configuration
└── main.cpp                     # Example usage and demonstrations
```

## Dependencies

- **CMake** 3.20 or higher
- **C++17** compatible compiler
- **Eigen3** (automatically downloaded via FetchContent)

## Building

1. Clone the repository:
```bash
git clone git@github.com:brgsk/braiNNlet.git
cd braiNNlet
```

2. Create build directory and compile:
```bash
mkdir build
cd build
cmake ..
make
```

3. Run the main example:
```bash
./main
```

4. Run tests:
```bash
./test
```

## Usage

### Basic Tensor Operations

```cpp
#include "src/core/nn/tensor.hpp"

// Create tensors
Tensor tensor(10, 10, 10);  // 3D tensor
Tensor tensor2(Matrix::Zero(10, 10));  // From Eigen matrix

// Mathematical operations
Tensor result = tensor1 + tensor2;
Tensor scaled = tensor1 * 2.0;
tensor1 *= 2.0;  // In-place operations
```

### Neural Network Layer

```cpp
#include "src/core/nn/layer.hpp"

// Create a layer with 3 neurons and 4 inputs
Layer layer(3, 4);

// Create input data
Matrix inputMatrix(1, 4);
inputMatrix << 1.0, 2.0, 3.0, 4.0;
Tensor input(inputMatrix);

// Forward pass
Tensor output = layer.forward(input);

// Batch processing
Matrix batchMatrix(2, 4);  // 2 samples, 4 features each
batchMatrix << 1.0, 2.0, 3.0, 4.0,
               5.0, 6.0, 7.0, 8.0;
Tensor batchInput(batchMatrix);
Tensor batchOutput = layer.forward(batchInput);
```

## Current Implementation Status

### ✅ Implemented
- Tensor class with basic operations (add, subtract, multiply, divide)
- Dense layer with configurable neurons and inputs
- Forward pass implementation
- Weight initialization
- Batch processing support
- Memory management with move semantics

### 🚧 In Development
- Backward pass (gradient computation)
- Activation functions
- Loss functions
- Optimizers
- Multi-layer network class

### 📋 Planned Features
- Convolutional layers
- Recurrent layers (LSTM, GRU)
- Model serialization/deserialization
- GPU acceleration
- Python bindings

## Author

Bartosz Roguski 