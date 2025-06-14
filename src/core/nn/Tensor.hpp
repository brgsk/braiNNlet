#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace nn
{

    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;

    class Tensor
    {
    public:
        Tensor() = default;
        explicit Tensor(const Matrix &data);
        explicit Tensor(Matrix &&data);
        Tensor(int rows, int cols);

        // Basic operations
        const Matrix &data() const { return data_; }
        Matrix &data() { return data_; }

        int rows() const { return static_cast<int>(data_.rows()); }
        int cols() const { return static_cast<int>(data_.cols()); }
        int size() const { return static_cast<int>(data_.size()); }

        // Element access
        double operator()(int row, int col) const { return data_(row, col); }
        double &operator()(int row, int col) { return data_(row, col); }

        // Mathematical operations
        Tensor operator+(const Tensor &other) const;
        Tensor operator-(const Tensor &other) const;
        Tensor operator*(const Tensor &other) const;
        Tensor &operator+=(const Tensor &other);
        Tensor &operator-=(const Tensor &other);

        // Scalar operations
        Tensor operator*(double scalar) const;
        Tensor operator/(double scalar) const;
        Tensor &operator*=(double scalar);
        Tensor &operator/=(double scalar);

        // Utility functions
        void zero();
        void random(double min = -1.0, double max = 1.0);
        void fill(double value);

        // Statistics
        double mean() const;
        double sum() const;
        double norm() const;

        // Shape operations
        Tensor transpose() const;
        void resize(int rows, int cols);

        // Serialization
        std::vector<double> to_vector() const;
        void from_vector(const std::vector<double> &vec, int rows, int cols);

    private:
        Matrix data_;
    };

    // Free functions
    Tensor operator*(double scalar, const Tensor &tensor);

} // namespace nn