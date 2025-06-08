#include "activations.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

// ReLU Implementation
Tensor ReLU::forward(const Tensor& input) const {
    Tensor output(input.rows(), input.cols());
    for (int i = 0; i < input.rows(); i++) {
        for (int j = 0; j < input.cols(); j++) {
            output(i, j) = std::max(0.0, input(i, j));
        }
    }
    return output;
}

Tensor ReLU::backward(const Tensor& input) const {
    Tensor gradient(input.rows(), input.cols());
    for (int i = 0; i < input.rows(); i++) {
        for (int j = 0; j < input.cols(); j++) {
            gradient(i, j) = input(i, j) > 0 ? 1 : 0;
        }
    }
    return gradient;
}

// Sigmoid Implementation
Tensor Sigmoid::forward(const Tensor& input) const {
    Tensor output(input.rows(), input.cols());
    for (int i = 0; i < input.rows(); i++) {
        for (int j = 0; j < input.cols(); j++) {
            double x = std::max(-500.0, std::min(500.0, input(i, j)));
            output(i, j) = 1.0 / (1.0 + std::exp(-x));
        }
    }
    return output;
}

Tensor Sigmoid::backward(const Tensor& input) const {
    Tensor sigmoid_output = forward(input);
    Tensor gradient(input.rows(), input.cols());
    for (int i = 0; i < input.rows(); i++) {
        for (int j = 0; j < input.cols(); j++) {
            double s = sigmoid_output(i, j);
            gradient(i, j) = s * (1 - s);
        }
    }
    return gradient;
}

// Tanh Implementation
Tensor Tanh::forward(const Tensor& input) const {
    Tensor output(input.rows(), input.cols());
    for (int i = 0; i < input.rows(); i++) {
        for (int j = 0; j < input.cols(); j++) {
            output(i, j) = std::tanh(input(i, j));
        }
    }
    return output;
}

Tensor Tanh::backward(const Tensor& input) const {
    Tensor tanh_output = forward(input);
    Tensor gradient(input.rows(), input.cols());
    for (int i = 0; i < input.rows(); i++) {
        for (int j = 0; j < input.cols(); j++) {
            double s = tanh_output(i, j);
            gradient(i, j) = 1 - s * s;
        }
    }
    return gradient;
}

// Linear Implementation
Tensor Linear::forward(const Tensor& input) const {
    return input;
}

Tensor Linear::backward(const Tensor& input) const {
    Tensor gradient(input.rows(), input.cols());
    gradient.fill(1.0);  // Identity function, so gradient is 1.0
    return gradient;
}

// Factory function to create activation functions
std::unique_ptr<ActivationFunction> create_activation(ActivationType type) {
    switch (type) {
        case ActivationType::ReLU:
            return std::make_unique<ReLU>();
        case ActivationType::Sigmoid:
            return std::make_unique<Sigmoid>();
        case ActivationType::Tanh:
            return std::make_unique<Tanh>();
        case ActivationType::Linear:
            return std::make_unique<Linear>();
        default:
            throw std::invalid_argument("Unknown activation type");
    }
}

ActivationType activation_from_string(const std::string& name) {
    if (name == "ReLU")
        return ActivationType::ReLU;
    if (name == "Sigmoid")
        return ActivationType::Sigmoid;
    if (name == "Tanh")
        return ActivationType::Tanh;
    if (name == "Linear")
        return ActivationType::Linear;
    throw std::invalid_argument("Unknown activation function " + name);
}

std::string activation_to_string(ActivationType type) {
    switch (type) {
        case ActivationType::ReLU:
            return "ReLU";
        case ActivationType::Sigmoid:
            return "Sigmoid";
        case ActivationType::Tanh:
            return "Tanh";
        case ActivationType::Linear:
            return "Linear";
        default:
            return "Unknown";
    }
}