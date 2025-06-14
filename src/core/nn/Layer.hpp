#pragma once

#include "Tensor.hpp"
#include <string>
#include <memory>

namespace nn {

class Layer {
public:
    virtual ~Layer() = default;
    
    // Pure virtual functions that must be implemented by derived classes
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& gradient) = 0;
    
    // Optional overrides
    virtual void update_weights(double learning_rate) {}
    virtual void zero_gradients() {}
    
    // Getters
    virtual std::string name() const = 0;
    virtual int input_size() const = 0;
    virtual int output_size() const = 0;
    virtual bool has_parameters() const { return false; }
    
    // For serialization and configuration
    virtual void set_training(bool training) { training_ = training; }
    virtual bool is_training() const { return training_; }
    
    // For network topology visualization
    virtual int parameter_count() const { return 0; }

protected:
    bool training_ = true;
};

} // namespace nn 