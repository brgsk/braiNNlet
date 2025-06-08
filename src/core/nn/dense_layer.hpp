#pragma once

#include "layer.hpp"

class DenseLayer : public Layer {
public:
    DenseLayer(int input_size, int output_size);
    ~DenseLayer() = default;

    Tensor forward(const Tensor& input) override;
    void backward(Tensor& gradient) override;

private:
    int _input_size;
    int _output_size;
    
    // Parameters
    Tensor weights_;
    Tensor biases_;

    // Gradients
    Tensor weights__grad;
    Tensor biases__grad;
};