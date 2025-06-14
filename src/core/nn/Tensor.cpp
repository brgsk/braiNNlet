#include "Tensor.hpp"
#include <random>
#include <stdexcept>

namespace nn {

Tensor::Tensor(const Matrix& data) : data_(data) {}

Tensor::Tensor(Matrix&& data) : data_(std::move(data)) {}

Tensor::Tensor(int rows, int cols) : data_(rows, cols) {
    data_.setZero();
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (rows() != other.rows() || cols() != other.cols()) {
        throw std::invalid_argument("Tensor dimensions must match for addition");
    }
    return Tensor(data_ + other.data_);
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (rows() != other.rows() || cols() != other.cols()) {
        throw std::invalid_argument("Tensor dimensions must match for subtraction");
    }
    return Tensor(data_ - other.data_);
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (cols() != other.rows()) {
        throw std::invalid_argument("Invalid dimensions for matrix multiplication");
    }
    return Tensor(data_ * other.data_);
}

Tensor& Tensor::operator+=(const Tensor& other) {
    if (rows() != other.rows() || cols() != other.cols()) {
        throw std::invalid_argument("Tensor dimensions must match for addition");
    }
    data_ += other.data_;
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    if (rows() != other.rows() || cols() != other.cols()) {
        throw std::invalid_argument("Tensor dimensions must match for subtraction");
    }
    data_ -= other.data_;
    return *this;
}

Tensor Tensor::operator*(double scalar) const {
    return Tensor(data_ * scalar);
}

Tensor Tensor::operator/(double scalar) const {
    if (std::abs(scalar) < 1e-10) {
        throw std::invalid_argument("Division by zero");
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

double Tensor::mean() const {
    return data_.mean();
}

double Tensor::sum() const {
    return data_.sum();
}

double Tensor::norm() const {
    return data_.norm();
}

Tensor Tensor::transpose() const {
    return Tensor(data_.transpose());
}

void Tensor::resize(int rows, int cols) {
    data_.resize(rows, cols);
}

std::vector<double> Tensor::to_vector() const {
    std::vector<double> result(size());
    for (int i = 0; i < size(); ++i) {
        result[i] = data_(i);
    }
    return result;
}

void Tensor::from_vector(const std::vector<double>& vec, int rows, int cols) {
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

Tensor operator*(double scalar, const Tensor& tensor) {
    return tensor * scalar;
}

} // namespace nn 