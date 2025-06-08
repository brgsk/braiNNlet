//
// Created by Bartosz Roguski on 08/06/2025.
//

#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

class Tensor {
public:
    Tensor() = default;
    Tensor(int rows, int cols, int depth);
    explicit Tensor(const Matrix& data);
    explicit Tensor(Matrix&& data);
    
    // Rule of 5: explicitly default these since Eigen handles memory correctly
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) = default;
    ~Tensor() = default;
    
    void printTensor();
    void updateTensor(const Matrix& value);

    // Mathematical operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    // Scalar operations
    Tensor operator*(double scalar) const;
    Tensor operator/(double scalar) const;
    Tensor operator*=(double scalar);
    Tensor operator/=(double scalar);

    // Accessors
    const Matrix& getData() const { return _data; }
    
private:
    Matrix _data;
};



