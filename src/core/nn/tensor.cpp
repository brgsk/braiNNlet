//
// Created by Bartosz Roguski on 08/06/2025.
//

#include "tensor.hpp"

#include <iostream>
#include <random>
#include <stdexcept>

Tensor::Tensor(const Matrix& data) : data_(data) {}

Tensor::Tensor(Matrix&& data) : data_(std::move(data)) {}

Tensor::Tensor(int rows, int cols) : data_(rows, cols) {
    data_.setZero();
}

void Tensor::validateDimensions(const Tensor& other) const {
    if (data_.rows() != other.data_.rows() || data_.cols() != other.data_.cols()) {
        throw std::invalid_argument(
            "Tensor dimensions don't match: (" + std::to_string(data_.rows()) + "x" +
            std::to_string(data_.cols()) + ") vs (" + std::to_string(other.data_.rows()) + "x" +
            std::to_string(other.data_.cols()) + ")");
    }
}

// Element-wise operations
Tensor Tensor::operator+(const Tensor& other) const {
    validateDimensions(other);
    return Tensor(data_ + other.data_);
}

Tensor Tensor::operator-(const Tensor& other) const {
    validateDimensions(other);
    return Tensor(data_ - other.data_);
}

Tensor Tensor::operator*(const Tensor& other) const {
    // Matrix multiplication: validate that A.cols == B.rows
    if (data_.cols() != other.data_.rows()) {
        throw std::invalid_argument(
            "Matrix multiplication dimension mismatch: (" + std::to_string(data_.rows()) + "x" +
            std::to_string(data_.cols()) + ") * (" + std::to_string(other.data_.rows()) + "x" +
            std::to_string(other.data_.cols()) + ")");
    }
    return Tensor(data_ * other.data_);
}

Tensor& Tensor::operator+=(const Tensor& other) {
    validateDimensions(other);
    data_ += other.data_;
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    validateDimensions(other);
    data_ -= other.data_;
    return *this;
}

// Scalar operations
Tensor Tensor::operator*(double scalar) const {
    return Tensor(data_ * scalar);
}

Tensor Tensor::operator/(double scalar) const {
    if (std::abs(scalar) < 1e-10) {
        throw std::invalid_argument("Division by zero or near-zero scalar");
    }
    return Tensor(data_ / scalar);
}

Tensor& Tensor::operator*=(double scalar) {
    data_ *= scalar;
    return *this;
}

Tensor& Tensor::operator/=(double scalar) {
    if (std::abs(scalar) < 1e-10) {
        throw std::invalid_argument("Division by zero");
    }
    data_ /= scalar;
    return *this;
}

// Utility methods

void Tensor::zero() {
    data_.setZero();
}

void Tensor::random(double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);

    for (int i = 0; i < rows(); ++i) {
        for (int j = 0; j < cols(); ++j) {
            data_(i, j) = dis(gen);
        }
    }
}

void Tensor::fill(double value) {
    data_.fill(value);
}

// Statistics

double Tensor::mean() const {
    return data_.mean();
}

double Tensor::sum() const {
    return data_.sum();
}

Tensor Tensor::sum(int axis) const {
    if (axis == 0) {
        // Sum along rows (result is 1 x cols)
        Matrix result = data_.colwise().sum();
        return Tensor(result);
    } else if (axis == 1) {
        // Sum along columns (result is rows x 1)
        Matrix result = data_.rowwise().sum();
        return Tensor(result);
    } else {
        throw std::invalid_argument("Invalid axis: " + std::to_string(axis) +
                                    ". Use 0 (rows) or 1 (cols)");
    }
}

double Tensor::norm() const {
    return data_.norm();
}

// Shape manipulation

Tensor Tensor::transpose() const {
    return Tensor(data_.transpose());
}

void Tensor::resize(int rows, int cols) {
    data_.resize(rows, cols);
}

// Serialization

std::vector<double> Tensor::toVector() const {
    std::vector<double> result(data_.size());
    for (int i = 0; i < data_.size(); ++i) {
        result[i] = data_(i);
    }
    return result;
}

void Tensor::fromVector(const std::vector<double>& vec, int rows, int cols) {
    if (static_cast<int>(vec.size()) != rows * cols) {
        throw std::invalid_argument("Vector size doesn't match tensor dimensions");
    }

    data_.resize(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data_(i, j) = vec[i * cols + j];
        }
    }
}

// Free functions

Tensor operator*(double scalar, const Tensor& tensor) {
    return tensor * scalar;
}

// Stream operator for printing
std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << tensor.data();
    return os;
}

// Function application
Tensor Tensor::apply(std::function<double(double)> func) const {
    Matrix result = data_;
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            result(i, j) = func(result(i, j));
        }
    }
    return Tensor(result);
}

Tensor Tensor::hadamard(const Tensor& other) const {
    validateDimensions(other);
    return Tensor(data_.array() * other.data_.array());
}

Tensor Tensor::operator/(const Tensor& other) const {
    validateDimensions(other);
    return Tensor(data_.array() / other.data_.array());
}

// Broadcasting operations
Tensor Tensor::broadcast_add(const Tensor& other) const {
    const Matrix& otherData = other.data_;

    // Handle bias addition: (batch_size, features) + (1, features)
    if (otherData.rows() == 1 && otherData.cols() == data_.cols()) {
        // Use Eigen's rowwise broadcasting
        return Tensor(data_.rowwise() + otherData.row(0));
    }
    // Handle case: (batch_size, features) + (batch_size, 1)
    else if (otherData.cols() == 1 && otherData.rows() == data_.rows()) {
        // Use Eigen's colwise broadcasting
        return Tensor(data_.colwise() + otherData.col(0));
    }
    // Same dimensions - regular addition
    else if (otherData.rows() == data_.rows() && otherData.cols() == data_.cols()) {
        return *this + other;
    } else {
        throw std::invalid_argument(
            "Cannot broadcast tensors with shapes (" + std::to_string(data_.rows()) + "x" +
            std::to_string(data_.cols()) + ") and (" + std::to_string(otherData.rows()) + "x" +
            std::to_string(otherData.cols()) + ")");
    }
}