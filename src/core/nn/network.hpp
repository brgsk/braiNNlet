#pragma once

#include <memory>
#include <string>
#include <vector>

#include "dense_layer.hpp"
#include "layer.hpp"
#include "loss.hpp"

struct LayerConfig {
    int neurons;
    ActivationType activation;

    LayerConfig(int n, ActivationType act) : neurons(n), activation(act) {}
};

class Network {
  public:
    Network();
    ~Network() = default;

    // Network building
    void add_layer(int neurons, ActivationType activation = ActivationType::ReLU);
    void add_layer(const LayerConfig& config);
    void remove_layer(int index = -1);  // -1 removes last layer
    void clear();

    // Forward and backward pass
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& loss_gradient);

    // Training
    void set_loss_function(LossType loss_type);
    double compute_loss(const Tensor& preds, const Tensor& targets);
    Tensor compute_loss_gradient(const Tensor& preds, const Tensor& targets);

    void update_weights(double learning_rate);
    void zero_gradients();

    // Network info
    int layer_count() const {
        return static_cast<int>(layers_.size());
    }
    int parameter_count() const;
    std::string summary() const;

    // Layer access
    const Layer& layer(int index) const;
    Layer& layer(int index);

    // Validation
    bool is_valid() const;
    std::string validation_error() const;

    // Training mode
    void set_training(bool training);
    bool is_training() const;

  private:
    std::vector<std::unique_ptr<Layer>> layers_;
    std::unique_ptr<LossFunction> loss_function_;
    bool training_ = true;

    void validate_network() const;
};