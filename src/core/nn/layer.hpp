#pragma once

#include <memory>
#include <string>

#include "tensor.hpp"

class Layer {
  public:
    virtual ~Layer() = default;

    // Pure virtual methods that must be implemented by subclasses
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& gradient) = 0;

    // Optional overrides
    virtual void update_weights(double learning_rate) {}
    virtual void zero_gradients() {}

    // Getters
    virtual std::string name() const = 0;
    virtual int input_size() const = 0;
    virtual int output_size() const = 0;
    virtual bool has_parameters() const {
        return false;
    }

    // Serialization and configuration
    virtual void set_training(bool training) {
        training_ = training;
    }
    virtual bool is_training() const {
        return training_;
    }

    // Network topology visualization
    virtual int parameter_count() const {
        return 0;
    }

  private:
    bool training_ = true;
};