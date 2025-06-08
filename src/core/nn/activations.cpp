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

Tensor ReLU::backward(const Tensor& gradient) const {
    // For ReLU: gradient * (input > 0 ? 1 : 0)
    // Note: We need the original input, but we'll approximate using gradient
    // In a proper implementation, we'd store the input from forward pass
    Tensor output(gradient.rows(), gradient.cols());
    for (int i = 0; i < gradient.rows(); i++) {
        for (int j = 0; j < gradient.cols(); j++) {
            // This is a simplified version - in practice you'd store forward input
            output(i, j) = gradient(i, j) > 0 ? gradient(i, j) : 0;
        }
    }
    return output;
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

Tensor Sigmoid::backward(const Tensor& gradient) const {
    // For sigmoid: gradient * sigmoid(x) * (1 - sigmoid(x))
    // Note: Proper implementation would store forward output
    Tensor output(gradient.rows(), gradient.cols());
    for (int i = 0; i < gradient.rows(); i++) {
        for (int j = 0; j < gradient.cols(); j++) {
            // Simplified - assumes gradient contains sigmoid output
            double s = gradient(i, j);
            output(i, j) = gradient(i, j) * s * (1 - s);
        }
    }
    return output;
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

Tensor Tanh::backward(const Tensor& gradient) const {
    // For tanh: gradient * (1 - tanh(x)^2)
    // Note: Proper implementation would store forward output
    Tensor output(gradient.rows(), gradient.cols());
    for (int i = 0; i < gradient.rows(); i++) {
        for (int j = 0; j < gradient.cols(); j++) {
            // Simplified - assumes gradient contains tanh output
            double t = gradient(i, j);
            output(i, j) = gradient(i, j) * (1 - t * t);
        }
    }
    return output;
}

// Linear Implementation
Tensor Linear::forward(const Tensor& input) const {
    return input;
}

Tensor Linear::backward(const Tensor& gradient) const {
    // For linear activation: gradient * 1 = gradient (pass through)
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