#include "Loss.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace nn {

// Mean Squared Error Implementation
double MeanSquaredError::forward(const Tensor& predictions, const Tensor& targets) const {
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        throw std::invalid_argument("Predictions and targets must have same dimensions");
    }
    
    double total_loss = 0.0;
    int count = predictions.rows() * predictions.cols();
    
    for (int i = 0; i < predictions.rows(); ++i) {
        for (int j = 0; j < predictions.cols(); ++j) {
            double diff = predictions(i, j) - targets(i, j);
            total_loss += diff * diff;
        }
    }
    
    return total_loss / count;
}

Tensor MeanSquaredError::backward(const Tensor& predictions, const Tensor& targets) const {
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        throw std::invalid_argument("Predictions and targets must have same dimensions");
    }
    
    Tensor gradient(predictions.rows(), predictions.cols());
    double scale = 2.0 / (predictions.rows() * predictions.cols());
    
    for (int i = 0; i < predictions.rows(); ++i) {
        for (int j = 0; j < predictions.cols(); ++j) {
            gradient(i, j) = scale * (predictions(i, j) - targets(i, j));
        }
    }
    
    return gradient;
}

// Cross Entropy Implementation
Tensor CrossEntropy::softmax(const Tensor& input) const {
    Tensor output(input.rows(), input.cols());
    
    for (int i = 0; i < input.rows(); ++i) {
        // Find max for numerical stability
        double max_val = input(i, 0);
        for (int j = 1; j < input.cols(); ++j) {
            max_val = std::max(max_val, input(i, j));
        }
        
        // Compute exponentials and sum
        double sum = 0.0;
        for (int j = 0; j < input.cols(); ++j) {
            output(i, j) = std::exp(input(i, j) - max_val);
            sum += output(i, j);
        }
        
        // Normalize
        for (int j = 0; j < input.cols(); ++j) {
            output(i, j) /= sum;
        }
    }
    
    return output;
}

double CrossEntropy::forward(const Tensor& predictions, const Tensor& targets) const {
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        throw std::invalid_argument("Predictions and targets must have same dimensions");
    }
    
    Tensor probs = softmax(predictions);
    double total_loss = 0.0;
    
    for (int i = 0; i < predictions.rows(); ++i) {
        for (int j = 0; j < predictions.cols(); ++j) {
            if (targets(i, j) > 0.0) {
                // Add small epsilon to prevent log(0)
                double prob = std::max(1e-15, probs(i, j));
                total_loss -= targets(i, j) * std::log(prob);
            }
        }
    }
    
    return total_loss / predictions.rows();
}

Tensor CrossEntropy::backward(const Tensor& predictions, const Tensor& targets) const {
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        throw std::invalid_argument("Predictions and targets must have same dimensions");
    }
    
    Tensor probs = softmax(predictions);
    Tensor gradient = probs - targets;
    
    // Scale by batch size
    gradient /= static_cast<double>(predictions.rows());
    
    return gradient;
}

// Binary Cross Entropy Implementation
double BinaryCrossEntropy::forward(const Tensor& predictions, const Tensor& targets) const {
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        throw std::invalid_argument("Predictions and targets must have same dimensions");
    }
    
    double total_loss = 0.0;
    int count = predictions.rows() * predictions.cols();
    
    for (int i = 0; i < predictions.rows(); ++i) {
        for (int j = 0; j < predictions.cols(); ++j) {
            double pred = std::max(1e-15, std::min(1.0 - 1e-15, predictions(i, j)));
            double target = targets(i, j);
            
            total_loss -= target * std::log(pred) + (1.0 - target) * std::log(1.0 - pred);
        }
    }
    
    return total_loss / count;
}

Tensor BinaryCrossEntropy::backward(const Tensor& predictions, const Tensor& targets) const {
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        throw std::invalid_argument("Predictions and targets must have same dimensions");
    }
    
    Tensor gradient(predictions.rows(), predictions.cols());
    double scale = 1.0 / (predictions.rows() * predictions.cols());
    
    for (int i = 0; i < predictions.rows(); ++i) {
        for (int j = 0; j < predictions.cols(); ++j) {
            double pred = std::max(1e-15, std::min(1.0 - 1e-15, predictions(i, j)));
            double target = targets(i, j);
            
            gradient(i, j) = scale * ((pred - target) / (pred * (1.0 - pred)));
        }
    }
    
    return gradient;
}

// Factory functions
std::unique_ptr<LossFunction> create_loss(LossType type) {
    switch (type) {
        case LossType::MeanSquaredError:
            return std::make_unique<MeanSquaredError>();
        case LossType::CrossEntropy:
            return std::make_unique<CrossEntropy>();
        case LossType::BinaryCrossEntropy:
            return std::make_unique<BinaryCrossEntropy>();
        default:
            throw std::invalid_argument("Unknown loss type");
    }
}

LossType loss_from_string(const std::string& name) {
    if (name == "MSE" || name == "mse") return LossType::MeanSquaredError;
    if (name == "CrossEntropy" || name == "crossentropy") return LossType::CrossEntropy;
    if (name == "BinaryCrossEntropy" || name == "bce") return LossType::BinaryCrossEntropy;
    throw std::invalid_argument("Unknown loss function: " + name);
}

std::string loss_to_string(LossType type) {
    switch (type) {
        case LossType::MeanSquaredError: return "MSE";
        case LossType::CrossEntropy: return "CrossEntropy";
        case LossType::BinaryCrossEntropy: return "BinaryCrossEntropy";
        default: return "Unknown";
    }
}

} // namespace nn 