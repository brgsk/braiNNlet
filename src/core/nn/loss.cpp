#include "loss.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

// Mean Squared Error
double MeanSquaredError::forward(const Tensor& preds, const Tensor& targets) const {
    if (preds.rows() != targets.rows() || preds.cols() != targets.cols()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }

    double total_loss = 0.0;
    int count = preds.rows() * preds.cols();
    for (int i = 0; i < preds.rows(); ++i) {
        for (int j = 0; j < preds.cols(); ++j) {
            double loss = preds(i, j) - targets(i, j);
            total_loss += loss * loss;
        }
    }
    return total_loss / count;
}

Tensor MeanSquaredError::backward(const Tensor& preds, const Tensor& targets) const {
    if (preds.rows() != targets.rows() || preds.cols() != targets.cols()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }

    Tensor gradient(preds.rows(), preds.cols());
    double scale = 2.0 / (preds.rows() * preds.cols());
    for (int i = 0; i < preds.rows(); ++i) {
        for (int j = 0; j < preds.cols(); ++j) {
            gradient(i, j) = scale * (preds(i, j) - targets(i, j));
        }
    }
    return gradient;
}

// Cross Entropy
Tensor CrossEntropy::softmax(const Tensor& input) const {
    Tensor output(input.rows(), input.cols());

    for (int i = 0; i < input.rows(); ++i) {
        // Find max value for numerical stability
        double max_val = input(i, 0);
        for (int j = 1; j < input.cols(); ++j) {
            if (input(i, j) > max_val) {
                max_val = input(i, j);
            }
        }
        // Compute exp and sum
        double sum_exp = 0.0;
        for (int j = 0; j < input.cols(); ++j) {
            output(i, j) = std::exp(input(i, j) - max_val);
            sum_exp += output(i, j);
        }
        // Normalize
        for (int j = 0; j < input.cols(); ++j) {
            output(i, j) /= sum_exp;
        }
    }
    return output;
}

double CrossEntropy::forward(const Tensor& preds, const Tensor& targets) const {
    if (preds.rows() != targets.rows() || preds.cols() != targets.cols()) {
        throw std::invalid_argument("Predictions and targets must have same dimensions");
    }

    Tensor probs = softmax(preds);
    double total_loss = 0.0;

    for (int i = 0; i < preds.rows(); ++i) {
        for (int j = 0; j < preds.cols(); ++j) {
            if (targets(i, j) > 0.0) {
                // Add small epsilon to prevent log(0)
                double prob = std::max(1e-15, probs(i, j));
                total_loss -= targets(i, j) * std::log(prob);
            }
        }
    }

    return total_loss / preds.rows();
}

Tensor CrossEntropy::backward(const Tensor& preds, const Tensor& targets) const {
    if (preds.rows() != targets.rows() || preds.cols() != targets.cols()) {
        throw std::invalid_argument("Predictions and targets must have same dimensions");
    }

    Tensor probs = softmax(preds);
    Tensor gradient = probs - targets;

    // Scale by batch size
    gradient /= static_cast<double>(preds.rows());

    return gradient;
}

// Binary Cross Entropy
double BinaryCrossEntropy::forward(const Tensor& preds, const Tensor& targets) const {
    if (preds.rows() != targets.rows() || preds.cols() != targets.cols()) {
        throw std::invalid_argument("Predictions and targets must have same dimensions");
    }

    double total_loss = 0.0;
    int count = preds.rows() * preds.cols();

    for (int i = 0; i < preds.rows(); ++i) {
        for (int j = 0; j < preds.cols(); ++j) {
            double pred = std::max(1e-15, std::min(1.0 - 1e-15, preds(i, j)));
            double target = targets(i, j);

            total_loss -= target * std::log(pred) + (1.0 - target) * std::log(1.0 - pred);
        }
    }

    return total_loss / count;
}

Tensor BinaryCrossEntropy::backward(const Tensor& preds, const Tensor& targets) const {
    if (preds.rows() != targets.rows() || preds.cols() != targets.cols()) {
        throw std::invalid_argument("Predictions and targets must have same dimensions");
    }

    Tensor gradient(preds.rows(), preds.cols());
    double scale = 1.0 / (preds.rows() * preds.cols());

    for (int i = 0; i < preds.rows(); ++i) {
        for (int j = 0; j < preds.cols(); ++j) {
            double pred = std::max(1e-15, std::min(1.0 - 1e-15, preds(i, j)));
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
    if (name == "MSE" || name == "mse")
        return LossType::MeanSquaredError;
    if (name == "CrossEntropy" || name == "crossentropy")
        return LossType::CrossEntropy;
    if (name == "BinaryCrossEntropy" || name == "bce")
        return LossType::BinaryCrossEntropy;
    throw std::invalid_argument("Unknown loss function: " + name);
}

std::string loss_to_string(LossType type) {
    switch (type) {
        case LossType::MeanSquaredError:
            return "MSE";
        case LossType::CrossEntropy:
            return "CrossEntropy";
        case LossType::BinaryCrossEntropy:
            return "BinaryCrossEntropy";
        default:
            return "Unknown";
    }
}