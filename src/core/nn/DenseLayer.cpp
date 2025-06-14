#include "DenseLayer.hpp"
#include <cmath>
#include <random>
#include <stdexcept>

namespace nn {

DenseLayer::DenseLayer(int input_size, int output_size, ActivationType activation)
    : input_size_(input_size)
    , output_size_(output_size)
    , weights_(input_size, output_size)
    , biases_(1, output_size)
    , weight_gradients_(input_size, output_size)
    , bias_gradients_(1, output_size)
    , activation_(create_activation(activation))
{
    // Initialize weights and biases
    xavier_init();
    biases_.zero();
    zero_gradients();
}

Tensor DenseLayer::forward(const Tensor& input) {
    if (input.cols() != input_size_) {
        throw std::invalid_argument("Input size mismatch in DenseLayer");
    }
    
    // Cache input for backpropagation
    last_input_ = input;
    
    // Linear transformation: output = input * weights + biases
    last_linear_output_ = input * weights_;
    
    // Add bias to each row
    for (int i = 0; i < last_linear_output_.rows(); ++i) {
        for (int j = 0; j < last_linear_output_.cols(); ++j) {
            last_linear_output_(i, j) += biases_(0, j);
        }
    }
    
    // Apply activation function
    return activation_->forward(last_linear_output_);
}

Tensor DenseLayer::backward(const Tensor& gradient) {
    if (gradient.cols() != output_size_) {
        throw std::invalid_argument("Gradient size mismatch in DenseLayer");
    }
    
    // Compute gradient w.r.t. activation input
    Tensor activation_grad = activation_->backward(last_linear_output_);
    
    // Element-wise multiplication of incoming gradient with activation gradient
    Tensor linear_grad(gradient.rows(), gradient.cols());
    for (int i = 0; i < gradient.rows(); ++i) {
        for (int j = 0; j < gradient.cols(); ++j) {
            linear_grad(i, j) = gradient(i, j) * activation_grad(i, j);
        }
    }
    
    // Compute gradients w.r.t. weights: dW = input^T * linear_grad
    weight_gradients_ += last_input_.transpose() * linear_grad;
    
    // Compute gradients w.r.t. biases: db = sum(linear_grad, axis=0)
    for (int j = 0; j < output_size_; ++j) {
        double bias_grad = 0.0;
        for (int i = 0; i < linear_grad.rows(); ++i) {
            bias_grad += linear_grad(i, j);
        }
        bias_gradients_(0, j) += bias_grad;
    }
    
    // Compute gradient w.r.t. input: dX = linear_grad * weights^T
    return linear_grad * weights_.transpose();
}

void DenseLayer::update_weights(double learning_rate) {
    // Update weights: W = W - lr * dW
    weights_ -= weight_gradients_ * learning_rate;
    
    // Update biases: b = b - lr * db
    biases_ -= bias_gradients_ * learning_rate;
}

void DenseLayer::zero_gradients() {
    weight_gradients_.zero();
    bias_gradients_.zero();
}

std::string DenseLayer::name() const {
    return "Dense(" + std::to_string(input_size_) + "->" + 
           std::to_string(output_size_) + ", " + activation_->name() + ")";
}

int DenseLayer::parameter_count() const {
    return input_size_ * output_size_ + output_size_;
}

void DenseLayer::set_weights(const Tensor& weights) {
    if (weights.rows() != input_size_ || weights.cols() != output_size_) {
        throw std::invalid_argument("Weight dimensions don't match layer size");
    }
    weights_ = weights;
}

void DenseLayer::set_biases(const Tensor& biases) {
    if (biases.rows() != 1 || biases.cols() != output_size_) {
        throw std::invalid_argument("Bias dimensions don't match layer size");
    }
    biases_ = biases;
}

void DenseLayer::xavier_init() {
    // Xavier/Glorot initialization: weights ~ U(-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out)))
    double limit = std::sqrt(6.0 / (input_size_ + output_size_));
    weights_.random(-limit, limit);
}

void DenseLayer::he_init() {
    // He initialization for ReLU: weights ~ N(0, sqrt(2/fan_in))
    double std_dev = std::sqrt(2.0 / input_size_);
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<double> dis(0.0, std_dev);
    
    for (int i = 0; i < weights_.rows(); ++i) {
        for (int j = 0; j < weights_.cols(); ++j) {
            weights_(i, j) = dis(gen);
        }
    }
}

} // namespace nn 