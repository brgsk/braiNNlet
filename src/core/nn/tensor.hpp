//
// Created by Bartosz Roguski on 08/06/2025.
//

#pragma once

#include <Eigen/Dense>
#include <functional>
#include <memory>
#include <ostream>
#include <vector>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

class Tensor {
  public:
    Tensor() = default;
    explicit Tensor(const Matrix& data);
    explicit Tensor(Matrix&& data);
    Tensor(int rows, int cols);

    // Basic operations
    const Matrix& data() const {
        return data_;
    }
    Matrix& data() {
        return data_;
    }

    // Rule of 5: explicitly default these since Eigen handles memory correctly
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) = default;
    ~Tensor() = default;

    // Shape and dimension access
    int rows() const {
        return data_.rows();
    }
    int cols() const {
        return data_.cols();
    }
    std::pair<int, int> shape() const {
        return {data_.rows(), data_.cols()};
    }

    // Element access
    double& operator()(int row, int col) {
        return data_(row, col);
    }
    const double& operator()(int row, int col) const {
        return data_(row, col);
    }

    // Mathematical operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;  // Matrix multiplication
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);

    // Element-wise operations
    Tensor hadamard(const Tensor& other) const;   // Element-wise multiplication
    Tensor operator/(const Tensor& other) const;  // Element-wise division

    // Broadcasting operations
    Tensor broadcast_add(const Tensor& other) const;  // For bias addition

    // Matrix operations
    Tensor transpose() const;

    // Scalar operations
    Tensor operator*(double scalar) const;
    Tensor operator/(double scalar) const;
    Tensor& operator*=(double scalar);
    Tensor& operator/=(double scalar);

    // Utility methods
    void zero();
    void random(double min, double max);
    void fill(double value);

    // Statistics
    double sum() const;          // Sum all elements
    Tensor sum(int axis) const;  // Sum along axis (0=rows, 1=cols)
    double mean() const;
    double norm() const;

    // Function application
    Tensor apply(std::function<double(double)> func) const;

    // Shape manipulation
    void resize(int rows, int cols);

    // Serialization
    std::vector<double> toVector() const;
    void fromVector(const std::vector<double>& vector, int rows, int cols);

  private:
    Matrix data_;

    // Helper for dimension validation
    void validateDimensions(const Tensor& other) const;
};

// Free functions
Tensor operator*(double scalar, const Tensor& tensor);

// Stream operator for printing
std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
