//
// Created by Bartosz Roguski on 08/06/2025.
//

#include "tensor.hpp"
#include <iostream>
#include <stdexcept>

Tensor::Tensor(int rows, int cols, int depth) : _data(Matrix::Zero(rows, cols)) {
    if (depth != 1) {
        std::cerr << "Warning: depth parameter ignored. This implementation only supports 2D matrices." << std::endl;
    }
}

Tensor::Tensor(const Matrix& data) : _data(data) {}

Tensor::Tensor(Matrix&& data) : _data(std::move(data)) {}

void Tensor::printTensor() {
    std::cout << _data << std::endl;
}

void Tensor::updateTensor(const Matrix& value) {
    _data = value;
}

void Tensor::validateDimensions(const Tensor& other) const {
    if (_data.rows() != other._data.rows() || _data.cols() != other._data.cols()) {
        throw std::invalid_argument("Tensor dimensions don't match: (" + 
                                  std::to_string(_data.rows()) + "x" + std::to_string(_data.cols()) + 
                                  ") vs (" + std::to_string(other._data.rows()) + "x" + 
                                  std::to_string(other._data.cols()) + ")");
    }
}

// Element-wise operations
Tensor Tensor::operator+(const Tensor& other) const {
    validateDimensions(other);
    return Tensor(_data + other._data);
}

Tensor Tensor::operator-(const Tensor& other) const {
    validateDimensions(other);
    return Tensor(_data - other._data);
}

Tensor Tensor::hadamard(const Tensor& other) const {
    validateDimensions(other);
    return Tensor(_data.array() * other._data.array());
}

Tensor Tensor::operator/(const Tensor& other) const {
    validateDimensions(other);
    return Tensor(_data.array() / other._data.array());
}

// Compound assignment operators
Tensor& Tensor::operator+=(const Tensor& other) {
    validateDimensions(other);
    _data += other._data;
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    validateDimensions(other);
    _data -= other._data;
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
    validateDimensions(other);
    _data.array() /= other._data.array();
    return *this;
}

// Matrix operations
Tensor Tensor::matmul(const Tensor& other) const {
    if (_data.cols() != other._data.rows()) {
        throw std::invalid_argument("Matrix multiplication dimension mismatch: (" + 
                                  std::to_string(_data.rows()) + "x" + std::to_string(_data.cols()) + 
                                  ") * (" + std::to_string(other._data.rows()) + "x" + 
                                  std::to_string(other._data.cols()) + ")");
    }
    return Tensor(_data * other._data);
}

Tensor Tensor::transpose() const {
    return Tensor(_data.transpose());
}

// Scalar operations
Tensor Tensor::operator*=(double scalar) {
    _data *= scalar;
    return *this;
}

Tensor Tensor::operator*(double scalar) const {
    return Tensor(_data * scalar);
}

Tensor Tensor::operator/(double scalar) const {
    if (std::abs(scalar) < 1e-10) {
        throw std::invalid_argument("Division by zero or near-zero scalar");
    }
    return Tensor(_data / scalar);
}

Tensor Tensor::operator/=(double scalar) {
    if (std::abs(scalar) < 1e-10) {
        throw std::invalid_argument("Division by zero or near-zero scalar");
    }
    _data /= scalar;
    return *this;
}

// Broadcasting operations
Tensor Tensor::broadcast_add(const Tensor& other) const {
    const Matrix& otherData = other._data;
    
    // Handle bias addition: (batch_size, features) + (1, features)
    if (otherData.rows() == 1 && otherData.cols() == _data.cols()) {
        Matrix result = _data;
        for (int i = 0; i < result.rows(); ++i) {
            result.row(i) += otherData.row(0);
        }
        return Tensor(result);
    }
    // Handle case: (batch_size, features) + (batch_size, 1)
    else if (otherData.cols() == 1 && otherData.rows() == _data.rows()) {
        Matrix result = _data;
        for (int j = 0; j < result.cols(); ++j) {
            result.col(j) += otherData.col(0);
        }
        return Tensor(result);
    }
    // Same dimensions - regular addition
    else if (otherData.rows() == _data.rows() && otherData.cols() == _data.cols()) {
        return *this + other;
    }
    else {
        throw std::invalid_argument("Cannot broadcast tensors with shapes (" + 
                                  std::to_string(_data.rows()) + "x" + std::to_string(_data.cols()) + 
                                  ") and (" + std::to_string(otherData.rows()) + "x" + 
                                  std::to_string(otherData.cols()) + ")");
    }
}

// Reduction operations
Tensor Tensor::sum(int axis) const {
    if (axis == -1) {
        // Sum all elements
        double total = _data.sum();
        Matrix result(1, 1);
        result(0, 0) = total;
        return Tensor(result);
    } else if (axis == 0) {
        // Sum along rows (result is 1 x cols)
        return Tensor(_data.colwise().sum());
    } else if (axis == 1) {
        // Sum along columns (result is rows x 1)
        return Tensor(_data.rowwise().sum());
    } else {
        throw std::invalid_argument("Invalid axis: " + std::to_string(axis) + ". Use -1, 0, or 1");
    }
}

Tensor Tensor::mean(int axis) const {
    if (axis == -1) {
        // Mean of all elements
        double avg = _data.mean();
        Matrix result(1, 1);
        result(0, 0) = avg;
        return Tensor(result);
    } else if (axis == 0) {
        // Mean along rows (result is 1 x cols)
        return Tensor(_data.colwise().mean());
    } else if (axis == 1) {
        // Mean along columns (result is rows x 1)
        return Tensor(_data.rowwise().mean());
    } else {
        throw std::invalid_argument("Invalid axis: " + std::to_string(axis) + ". Use -1, 0, or 1");
    }
}

double Tensor::norm() const {
    return _data.norm();
}

// Function application
Tensor Tensor::apply(std::function<double(double)> func) const {
    Matrix result = _data;
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            result(i, j) = func(result(i, j));
        }
    }
    return Tensor(result);
}

// Shape manipulation
Tensor Tensor::reshape(int rows, int cols) const {
    if (rows * cols != _data.rows() * _data.cols()) {
        throw std::invalid_argument("Cannot reshape tensor from (" + 
                                  std::to_string(_data.rows()) + "x" + std::to_string(_data.cols()) + 
                                  ") to (" + std::to_string(rows) + "x" + std::to_string(cols) + 
                                  "): incompatible sizes");
    }
    
    // Use Eigen's Map functionality for proper reshape
    // First, get the data as a flattened vector (row-major order)
    Matrix reshaped(rows, cols);
    int idx = 0;
    for (int i = 0; i < _data.rows(); ++i) {
        for (int j = 0; j < _data.cols(); ++j) {
            int new_row = idx / cols;
            int new_col = idx % cols;
            reshaped(new_row, new_col) = _data(i, j);
            idx++;
        }
    }
    return Tensor(reshaped);
}
