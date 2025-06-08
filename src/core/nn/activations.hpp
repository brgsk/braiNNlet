#pragma once

#include <memory>
#include <string>

#include "tensor.hpp"

enum class ActivationType { ReLU, Sigmoid, Tanh, Linear };

class ActivationFunction {
  public:
    virtual ~ActivationFunction() = default;
    virtual Tensor forward(const Tensor& input) const = 0;
    virtual Tensor backward(const Tensor& gradient) const = 0;
    virtual std::string name() const = 0;
    virtual ActivationType type() const = 0;
};

class ReLU : public ActivationFunction {
  public:
    Tensor forward(const Tensor& input) const override;
    Tensor backward(const Tensor& gradient) const override;
    std::string name() const override {
        return "ReLU";
    }
    ActivationType type() const override {
        return ActivationType::ReLU;
    }
};

class Sigmoid : public ActivationFunction {
  public:
    Tensor forward(const Tensor& input) const override;
    Tensor backward(const Tensor& gradient) const override;
    std::string name() const override {
        return "Sigmoid";
    }
    ActivationType type() const override {
        return ActivationType::Sigmoid;
    }
};

class Tanh : public ActivationFunction {
  public:
    Tensor forward(const Tensor& input) const override;
    Tensor backward(const Tensor& gradient) const override;
    std::string name() const override {
        return "Tanh";
    }
    ActivationType type() const override {
        return ActivationType::Tanh;
    }
};

class Linear : public ActivationFunction {
  public:
    Tensor forward(const Tensor& input) const override;
    Tensor backward(const Tensor& gradient) const override;
    std::string name() const override {
        return "Linear";
    }
    ActivationType type() const override {
        return ActivationType::Linear;
    }
};

// Factory function to create activation functions
std::unique_ptr<ActivationFunction> create_activation(ActivationType type);
ActivationType activation_from_string(const std::string& name);
std::string activation_to_string(ActivationType type);