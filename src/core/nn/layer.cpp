#include "layer.hpp"
#include <stdexcept>

Layer::Layer(int numNeurons, int numInputs) : numNeurons_(numNeurons), numInputs_(numInputs) {
    initializeWeights();
}

void Layer::initializeWeights() {
    // Xavier/Glorot initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    double variance = 2.0 / (numInputs_ + numNeurons_);
    std::normal_distribution<double> distribution(0.0, std::sqrt(variance));
    
    // Initialize weights matrix (numInputs x numNeurons)
    Matrix weights = Matrix(numInputs_, numNeurons_);
    for (int i = 0; i < numInputs_; ++i) {
        for (int j = 0; j < numNeurons_; ++j) {
            weights(i, j) = distribution(gen);
        }
    }
    weights_ = Tensor(weights);
    
    // Initialize biases to zero (1 x numNeurons)
    Matrix biases = Matrix::Zero(1, numNeurons_);
    biases_ = Tensor(biases);
}

void Layer::printLayer() {
    std::cout << "Layer shape: [" << numInputs_ << " x " << numNeurons_ << "]" << std::endl;
    std::cout << "Weights:\n" << weights_.getData() << std::endl;
    std::cout << "Biases:\n" << biases_.getData() << std::endl;
}

Tensor Layer::forward(const Tensor& input) {
    // Forward pass: output = input * weights + biases
    // input shape: (batch_size, numInputs) or (1, numInputs)
    // weights shape: (numInputs, numNeurons) 
    // output shape: (batch_size, numNeurons) or (1, numNeurons)
    
    // Validate dimensions
    if (input.cols() != numInputs_) {
        throw std::invalid_argument("Input dimension mismatch: expected " + 
                                  std::to_string(numInputs_) + 
                                  " but got " + std::to_string(input.cols()));
    }
    
    // Matrix multiplication: input * weights
    Tensor output = input.matmul(weights_);
    
    // Add biases using broadcasting
    output = output.broadcast_add(biases_);
    
    return output;
}

void Layer::backward(Tensor& gradient) {
    std::cout << "Backward" << std::endl;
}