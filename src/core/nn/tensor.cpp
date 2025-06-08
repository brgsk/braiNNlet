//
// Created by Bartosz Roguski on 08/06/2025.
//

#include "tensor.hpp"
#include <iostream>

Tensor::Tensor(int rows, int cols, int depth) : _data(Matrix::Zero(rows, cols)) {
    // Note: This creates a 2D matrix. For true 3D tensor, you'd need a different approach
    // For now, treating depth as ignored or flattening to 2D
}

Tensor::Tensor(const Matrix& data) : _data(data) {}

Tensor::Tensor(Matrix&& data) : _data(std::move(data)) {}

void Tensor::printTensor() {
    std::cout << _data << std::endl;
}

void Tensor::updateTensor(const Matrix& value) {
    _data = value;
}

Tensor Tensor::operator+(const Tensor& other) const {
    return Tensor(_data + other._data);
}

Tensor Tensor::operator-(const Tensor& other) const {
    return Tensor(_data - other._data);
}

Tensor Tensor::operator*(const Tensor& other) const {
    return Tensor(_data.array() * other._data.array());
}

Tensor Tensor::operator/(const Tensor& other) const {
    return Tensor(_data.array() / other._data.array());
}

Tensor Tensor::operator*=(double scalar) {
    _data *= scalar;
    return *this;
}

Tensor Tensor::operator*(double scalar) const {
    return Tensor(_data * scalar);
}

Tensor Tensor::operator/(double scalar) const {
    return Tensor(_data / scalar);
}

Tensor Tensor::operator/=(double scalar) {
    _data /= scalar;
    return *this;
}
