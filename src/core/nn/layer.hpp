#pragma once

#include <vector>
#include <iostream>
#include <random>

#include "tensor.hpp"

class Layer {
public:
    Layer(int numNeurons, int numInputs);
    ~Layer() = default;

    // Getters
    int getNumNeurons() const { return numNeurons_; }
    int getNumInputs() const { return numInputs_; }
    const Tensor& getWeights() const { return weights_; }
    const Tensor& getBiases() const { return biases_; }

    // Other methods
    void printLayer();
    Tensor forward(const Tensor& input);
    void backward(Tensor& gradient);

private:
    void initializeWeights();
    
    int numNeurons_;
    int numInputs_;
    Tensor weights_;  // Shape: (numInputs, numNeurons)
    Tensor biases_;   // Shape: (1, numNeurons)
};