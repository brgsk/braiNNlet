#pragma once

#include <memory>

#include "activations.hpp"
#include "layer.hpp"

class DenseLayer : public Layer {
  public:
    DenseLayer(int input_size, int output_size, ActivationType activation = ActivationType::ReLU);

    // Layer interface
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradient) override;
    void update_weights(double learning_rate) override;
    void zero_gradients() override;

    // Getters
    std::string name() const override {
        return "Dense";
    }
    int input_size() const override {
        return input_size_;
    }
    int output_size() const override {
        return output_size_;
    }
    bool has_parameters() const override {
        return true;
    }
    int parameter_count() const override {
        return input_size_ * output_size_ + output_size_;
    }

    // Weight management
    const Tensor& weights() const {
        return weights_;
    }
    const Tensor& biases() const {
        return biases_;
    }
    void set_weights(const Tensor& weights);
    void set_biases(const Tensor& biases);

    // Activation function
    ActivationType activation_type() const {
        return activation_->type();
    }
    const ActivationFunction& activation() const {
        return *activation_;
    }

    // Initialize weights
    void xavier_init();  // Xavier initialization
    void he_init();      // He initialization for ReLU

  private:
    int input_size_;
    int output_size_;

    // Parameters
    Tensor weights_;  // Shape: input_size x output_size
    Tensor biases_;   // Shape: 1 x output_size

    // Gradients
    Tensor weight_grad_;
    Tensor bias_grad_;

    // Activation function
    std::unique_ptr<ActivationFunction> activation_;

    // Training state
    bool training_ = true;

    // Intermediate values for backward pass
    Tensor last_input_;
    Tensor last_linear_output_;
    Tensor last_output_;
};