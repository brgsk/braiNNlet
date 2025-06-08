#include "Dataset.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>

namespace data {

std::pair<Tensor, Tensor> Dataset::get_batch(const std::vector<int>& indices) const {
    if (indices.empty()) {
        throw std::invalid_argument("Cannot create batch from empty indices");
    }

    // Get first sample to determine dimensions
    DataSample first_sample = get(indices[0]);
    int feature_size = first_sample.features.cols();
    int label_size = first_sample.label.cols();
    int batch_size = static_cast<int>(indices.size());

    // Create batch tensors
    Tensor batch_features(batch_size, feature_size);
    Tensor batch_labels(batch_size, label_size);

    // Fill batch tensors
    for (int i = 0; i < batch_size; ++i) {
        DataSample sample = get(indices[i]);

        // Copy features
        for (int j = 0; j < feature_size; ++j) {
            batch_features(i, j) = sample.features(0, j);
        }

        // Copy labels
        for (int j = 0; j < label_size; ++j) {
            batch_labels(i, j) = sample.label(0, j);
        }
    }

    return std::make_pair(batch_features, batch_labels);
}

std::pair<Tensor, Tensor> Dataset::get_batch(int start, int batch_size) const {
    if (start < 0 || start >= size()) {
        throw std::out_of_range("Start index out of range");
    }

    int end = std::min(start + batch_size, size());
    std::vector<int> indices;
    for (int i = start; i < end; ++i) {
        indices.push_back(i);
    }

    return get_batch(indices);
}

Tensor Dataset::compute_mean() const {
    if (!loaded_ || samples_.empty()) {
        throw std::runtime_error("Dataset not loaded or empty");
    }

    int feature_size = samples_[0].features.cols();
    Tensor mean(1, feature_size);
    mean.zero();

    // Sum all features
    for (const auto& sample : samples_) {
        mean += sample.features;
    }

    // Divide by number of samples
    mean /= static_cast<double>(samples_.size());

    return mean;
}

Tensor Dataset::compute_std() const {
    if (!loaded_ || samples_.empty()) {
        throw std::runtime_error("Dataset not loaded or empty");
    }

    Tensor mean = compute_mean();
    int feature_size = samples_[0].features.cols();
    Tensor variance(1, feature_size);
    variance.zero();

    // Compute variance
    for (const auto& sample : samples_) {
        Tensor diff = sample.features - mean;
        for (int j = 0; j < feature_size; ++j) {
            variance(0, j) += diff(0, j) * diff(0, j);
        }
    }

    // Divide by number of samples and take square root
    variance /= static_cast<double>(samples_.size());

    Tensor std_dev(1, feature_size);
    for (int j = 0; j < feature_size; ++j) {
        std_dev(0, j) = std::sqrt(variance(0, j));
    }

    return std_dev;
}

void Dataset::normalize() {
    if (!loaded_ || samples_.empty()) {
        throw std::runtime_error("Dataset not loaded or empty");
    }

    if (normalized_) {
        return;  // Already normalized
    }

    mean_ = compute_mean();
    std_ = compute_std();

    // Normalize all samples
    for (auto& sample : samples_) {
        sample.features = (sample.features - mean_);

        // Avoid division by zero
        for (int j = 0; j < sample.features.cols(); ++j) {
            if (std_(0, j) > 1e-8) {
                sample.features(0, j) /= std_(0, j);
            }
        }
    }

    normalized_ = true;
}

// Utility functions
std::vector<int> generate_indices(int size) {
    std::vector<int> indices(size);
    for (int i = 0; i < size; ++i) {
        indices[i] = i;
    }
    return indices;
}

std::vector<int> shuffle_indices(const std::vector<int>& indices) {
    std::vector<int> shuffled = indices;

    static std::random_device rd;
    static std::mt19937 gen(rd());

    std::shuffle(shuffled.begin(), shuffled.end(), gen);
    return shuffled;
}

void split_dataset(const std::vector<int>& indices,
                   double train_ratio,
                   std::vector<int>& train_indices,
                   std::vector<int>& val_indices) {
    if (train_ratio < 0.0 || train_ratio > 1.0) {
        throw std::invalid_argument("Train ratio must be between 0 and 1");
    }

    int train_size = static_cast<int>(indices.size() * train_ratio);

    train_indices.clear();
    val_indices.clear();

    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
        if (i < train_size) {
            train_indices.push_back(indices[i]);
        } else {
            val_indices.push_back(indices[i]);
        }
    }
}

}  // namespace data