# braiNNlet - a lightweight C++ neural network library

A lightweight C++ neural network library built from scratch using Eigen for linear algebra operations. This project provides fundamental building blocks for creating and training neural networks.

## Features

- **Comprehensive Tensor Operations**: 
  - Element-wise operations (add, subtract, hadamard product, divide)
  - Matrix operations (matrix multiplication, transpose)
  - Broadcasting for bias addition
  - Reduction operations (sum, mean, norm)
  - Shape manipulation (reshape)
  - Function application (for activation functions)
  - Element access and shape queries
- **Neural Network Layers**: Configurable dense layers with forward pass implementation
- **Eigen Integration**: High-performance linear algebra operations using Eigen library
- **Modern C++**: Written in C++17 with proper memory management and move semantics
- **Batch Processing**: Support for batch forward passes through neural network layers
- **Comprehensive Testing**: Full test suite covering all tensor operations

## Project Structure

```
braiNNlet/
├── src/
│   └── core/
│       └── nn/
│           ├── tensor.hpp/cpp    # Tensor data structure and operations
│           └── layer.hpp/cpp     # Neural network layer implementation
├── tests/
│   └── test.cpp                  # Comprehensive unit tests
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

### Tensor Operations

```cpp
#include "src/core/nn/tensor.hpp"

// Create tensors
Tensor tensor(10, 10);  // 2D tensor
Tensor tensor2(Matrix::Zero(10, 10));  // From Eigen matrix

// Element-wise operations
Tensor sum = tensor1 + tensor2;
Tensor hadamard = tensor1.hadamard(tensor2);  // Element-wise multiplication
Tensor scaled = tensor1 * 2.0;
tensor1 *= 2.0;  // In-place operations

// Matrix operations
Tensor result = matrix1.matmul(matrix2);  // Matrix multiplication
Tensor transposed = matrix1.transpose();

// Broadcasting (for bias addition)
Tensor with_bias = data.broadcast_add(bias);

// Reductions
Tensor sum_all = tensor.sum(-1);  // Sum all elements
Tensor sum_cols = tensor.sum(0);  // Sum along columns
double norm = tensor.norm();

// Shape operations
auto shape = tensor.shape();  // Get dimensions
Tensor reshaped = tensor.reshape(new_rows, new_cols);

// Element access
double value = tensor(row, col);
tensor(row, col) = new_value;

// Function application (for activations)
auto relu = [](double x) { return std::max(0.0, x); };
Tensor activated = tensor.apply(relu);
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

### ✅ Fully Implemented
- **Tensor Operations**:
  - Element-wise arithmetic (add, subtract, divide, hadamard product)
  - Matrix multiplication and transpose
  - Broadcasting for bias addition
  - Reduction operations (sum, mean, norm)
  - Shape manipulation and element access
  - Function application for activations
  - Comprehensive dimension validation
- **Dense Layers**:
  - Configurable neurons and inputs
  - Xavier/Glorot weight initialization
  - Forward pass with matrix multiplication and bias addition
  - Batch processing support
- **Memory Management**: 
  - Proper move semantics and copy operations
  - Exception safety and error handling

### 🚧 In Development
- Backward pass (gradient computation)
- Activation functions (ReLU, Sigmoid, Softmax, etc.)
- Loss functions (MSE, Cross-entropy)
- Optimizers (SGD, Adam, RMSprop)
- Multi-layer network class

### 📋 Planned Features
- Convolutional layers
- Recurrent layers (LSTM, GRU)
- Model serialization/deserialization
- GPU acceleration
- Python bindings

## Testing

The project includes comprehensive tests covering:
- Tensor creation and initialization
- All mathematical operations
- Matrix operations and broadcasting
- Reduction operations and function application
- Shape manipulation and element access
- Neural network layer functionality

Run tests with:
```bash
./test
```

## Author

Bartosz Roguski 