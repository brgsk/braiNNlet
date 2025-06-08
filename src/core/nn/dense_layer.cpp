#include "dense_layer.hpp"

#include <cmath>
#include <stdexcept>

#include "activations.hpp"

DenseLayer::DenseLayer(int input_size, int output_size, ActivationType activation)
    : input_size_(input_size),
      output_size_(output_size),
      weights_(input_size, output_size),
      biases_(1, output_size),
      weight_grad_(input_size, output_size),
      bias_grad_(1, output_size),
      activation_(create_activation(activation)) {
    // Initialize with Xavier/Glorot initialization
    xavier_init();
}

Tensor DenseLayer::forward(const Tensor& input) {
    // Store input for backward pass (if training)
    if (training_) {
        last_input_ = input;
    }

    // Linear transformation: output = input * weights + bias
    // input shape: (batch_size, input_size)
    // weights shape: (input_size, output_size)
    // result shape: (batch_size, output_size)

    Tensor linear_output = input * weights_;

    // Add bias using broadcasting
    Tensor output = linear_output.broadcast_add(biases_);

    // Apply activation function
    Tensor activated = activation_->forward(output);

    // Store for backward pass
    if (training_) {
        last_linear_output_ = linear_output;
        last_output_ = activated;
    }

    return activated;
}

Tensor DenseLayer::backward(const Tensor& gradient) {
    if (!training_) {
        throw std::runtime_error("Cannot call backward during inference mode");
    }

    // Backward through activation function
    Tensor activation_grad = activation_->backward(gradient);

    // Compute gradients for weights and biases
    // weight_grad = last_input_.T * activation_grad
    weight_grad_ = last_input_.transpose() * activation_grad;

    // bias_grad = sum(activation_grad, axis=0)
    bias_grad_ = activation_grad.sum(0);

    // Compute gradient for input (to pass to previous layer)
    // input_grad = activation_grad * weights_.T
    Tensor input_grad = activation_grad * weights_.transpose();

    return input_grad;
}

void DenseLayer::update_weights(double learning_rate) {
    weights_ -= weight_grad_ * learning_rate;
    biases_ -= bias_grad_ * learning_rate;
}

void DenseLayer::zero_gradients() {
    weight_grad_.zero();
    bias_grad_.zero();
}

void DenseLayer::set_weights(const Tensor& weights) {
    if (weights.rows() != input_size_ || weights.cols() != output_size_) {
        throw std::invalid_argument("Weight dimensions don't match layer dimensions");
    }
    weights_ = weights;
}

void DenseLayer::set_biases(const Tensor& biases) {
    if (biases.rows() != 1 || biases.cols() != output_size_) {
        throw std::invalid_argument("Bias dimensions don't match layer output size");
    }
    biases_ = biases;
}

void DenseLayer::xavier_init() {
    double limit = std::sqrt(6.0 / (input_size_ + output_size_));
    weights_.random(-limit, limit);
    biases_.zero();
}

void DenseLayer::he_init() {
    double std_dev = std::sqrt(2.0 / input_size_);
    weights_.random(-std_dev, std_dev);
    biases_.zero();
}
