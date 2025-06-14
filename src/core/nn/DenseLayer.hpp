#pragma once

#include "Layer.hpp"
#include "Activations.hpp"
#include <memory>

namespace nn
{

    class DenseLayer : public Layer
    {
    public:
        DenseLayer(int input_size, int output_size, ActivationType activation = ActivationType::ReLU);

        // Layer interface implementation
        Tensor forward(const Tensor &input) override;
        Tensor backward(const Tensor &gradient) override;
        void update_weights(double learning_rate) override;
        void zero_gradients() override;

        // Getters
        std::string name() const override;
        int input_size() const override { return input_size_; }
        int output_size() const override { return output_size_; }
        bool has_parameters() const override { return true; }
        int parameter_count() const override;

        // Weight management
        const Tensor &weights() const { return weights_; }
        const Tensor &biases() const { return biases_; }
        void set_weights(const Tensor &weights);
        void set_biases(const Tensor &biases);

        // Gradient access
        const Tensor &weight_gradients() const { return weight_gradients_; }
        const Tensor &bias_gradients() const { return bias_gradients_; }

        // Activation function
        ActivationType activation_type() const { return activation_->type(); }
        const ActivationFunction &activation() const { return *activation_; }

        // Initialize weights using Xavier/Glorot initialization
        void xavier_init();
        void he_init(); // For ReLU networks

    private:
        int input_size_;
        int output_size_;

        // Parameters
        Tensor weights_; // input_size x output_size
        Tensor biases_;  // 1 x output_size

        // Gradients
        Tensor weight_gradients_;
        Tensor bias_gradients_;

        // Activation function
        std::unique_ptr<ActivationFunction> activation_;

        // Cached values for backpropagation
        Tensor last_input_;
        Tensor last_linear_output_;
    };

} // namespace nn