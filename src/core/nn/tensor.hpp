//
// Created by Bartosz Roguski on 08/06/2025.
//

#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <functional>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

class Tensor {
public:
    Tensor() = default;
    Tensor(int rows, int cols, int depth = 1);  // Made depth explicit with default
    explicit Tensor(const Matrix& data);
    explicit Tensor(Matrix&& data);
    
    // Rule of 5: explicitly default these since Eigen handles memory correctly
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) = default;
    ~Tensor() = default;
    
    // Utility methods
    void printTensor();
    void updateTensor(const Matrix& value);

    // Shape and dimension access
    int rows() const { return _data.rows(); }
    int cols() const { return _data.cols(); }
    std::pair<int, int> shape() const { return {_data.rows(), _data.cols()}; }
    
    // Element access
    double& operator()(int row, int col) { return _data(row, col); }
    const double& operator()(int row, int col) const { return _data(row, col); }

    // Mathematical operations (element-wise)
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor hadamard(const Tensor& other) const;  // Element-wise multiplication
    Tensor operator/(const Tensor& other) const;
    
    // Compound assignment operators
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);

    // Matrix operations
    Tensor matmul(const Tensor& other) const;  // Matrix multiplication
    Tensor transpose() const;

    // Scalar operations
    Tensor operator*(double scalar) const;
    Tensor operator/(double scalar) const;
    Tensor operator*=(double scalar);
    Tensor operator/=(double scalar);
    
    // Broadcasting operations
    Tensor broadcast_add(const Tensor& other) const;  // For bias addition
    
    // Reduction operations
    Tensor sum(int axis = -1) const;  // -1 means sum all elements
    Tensor mean(int axis = -1) const;
    double norm() const;
    
    // Function application
    Tensor apply(std::function<double(double)> func) const;
    
    // Shape manipulation
    Tensor reshape(int rows, int cols) const;

    // Accessors
    const Matrix& getData() const { return _data; }
    Matrix& getData() { return _data; }
    
private:
    Matrix _data;
    
    // Helper for dimension validation
    void validateDimensions(const Tensor& other) const;
};



