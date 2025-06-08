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
    int getNumNeurons() const { return _numNeurons; }
    int getNumInputs() const { return _numInputs; }
    const Tensor& getWeights() const { return _weights; }
    const Tensor& getBiases() const { return _biases; }

    // Other methods
    void printLayer();
    Tensor forward(const Tensor& input);
    void backward(Tensor& gradient);

private:
    void initializeWeights();
    
    int _numNeurons;
    int _numInputs;
    Tensor _weights;  // Shape: (numInputs, numNeurons)
    Tensor _biases;   // Shape: (1, numNeurons)
};