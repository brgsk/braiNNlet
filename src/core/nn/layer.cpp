#include "layer.hpp"
#include <stdexcept>

Layer::Layer(int numNeurons, int numInputs) : _numNeurons(numNeurons), _numInputs(numInputs) {
    initializeWeights();
}

void Layer::initializeWeights() {
    // Xavier/Glorot initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    double variance = 2.0 / (_numInputs + _numNeurons);
    std::normal_distribution<double> distribution(0.0, std::sqrt(variance));
    
    // Initialize weights matrix (numInputs x numNeurons)
    Matrix weights = Matrix(_numInputs, _numNeurons);
    for (int i = 0; i < _numInputs; ++i) {
        for (int j = 0; j < _numNeurons; ++j) {
            weights(i, j) = distribution(gen);
        }
    }
    _weights = Tensor(weights);
    
    // Initialize biases to zero (1 x numNeurons)
    Matrix biases = Matrix::Zero(1, _numNeurons);
    _biases = Tensor(biases);
}

void Layer::printLayer() {
    std::cout << "Layer shape: [" << _numInputs << " x " << _numNeurons << "]" << std::endl;
    std::cout << "Weights:\n" << _weights.getData() << std::endl;
    std::cout << "Biases:\n" << _biases.getData() << std::endl;
}

Tensor Layer::forward(const Tensor& input) {
    // Forward pass: output = input * weights + biases
    // input shape: (batch_size, numInputs) or (1, numInputs)
    // weights shape: (numInputs, numNeurons) 
    // output shape: (batch_size, numNeurons) or (1, numNeurons)
    
    // Validate dimensions
    if (input.cols() != _numInputs) {
        throw std::invalid_argument("Input dimension mismatch: expected " + 
                                  std::to_string(_numInputs) + 
                                  " but got " + std::to_string(input.cols()));
    }
    
    // Matrix multiplication: input * weights
    Tensor output = input.matmul(_weights);
    
    // Add biases using broadcasting
    output = output.broadcast_add(_biases);
    
    return output;
}

void Layer::backward(Tensor& gradient) {
    std::cout << "Backward" << std::endl;
}