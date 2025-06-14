#pragma once

#include <functional>
#include <vector>

#include "../data/Dataset.hpp"
#include "../nn/Network.hpp"

namespace training {

struct TrainingMetrics {
    double loss = 0.0;
    double accuracy = 0.0;
    int epoch = 0;
    int batch = 0;

    TrainingMetrics() = default;
    TrainingMetrics(double l, double acc, int e, int b)
        : loss(l), accuracy(acc), epoch(e), batch(b) {}
};

struct TrainingConfig {
    int epochs = 10;
    int batch_size = 32;
    double learning_rate = 0.001;
    double validation_split = 0.2;
    bool shuffle = true;
    int print_every = 10;  // Print every N batches

    TrainingConfig() = default;
    TrainingConfig(int e, int bs, double lr) : epochs(e), batch_size(bs), learning_rate(lr) {}
};

// Callback function types
using OnEpochEndCallback = std::function<void(
    int epoch, const TrainingMetrics& train_metrics, const TrainingMetrics& val_metrics)>;
using OnBatchEndCallback = std::function<void(int batch, const TrainingMetrics& metrics)>;
using OnTrainingStartCallback = std::function<void(int total_epochs, int batches_per_epoch)>;
using OnTrainingEndCallback = std::function<void(const std::vector<TrainingMetrics>& history)>;

class Trainer {
  public:
    Trainer(nn::Network& network, data::Dataset& dataset);

    // Training
    void train(const TrainingConfig& config);
    void stop() {
        should_stop_ = true;
    }

    // Callbacks
    void set_on_epoch_end(OnEpochEndCallback callback) {
        on_epoch_end_ = callback;
    }
    void set_on_batch_end(OnBatchEndCallback callback) {
        on_batch_end_ = callback;
    }
    void set_on_training_start(OnTrainingStartCallback callback) {
        on_training_start_ = callback;
    }
    void set_on_training_end(OnTrainingEndCallback callback) {
        on_training_end_ = callback;
    }

    // Evaluation
    TrainingMetrics evaluate(const std::vector<int>& indices, int batch_size = 32);
    double compute_accuracy(const nn::Tensor& predictions, const nn::Tensor& targets);

    // History
    const std::vector<TrainingMetrics>& training_history() const {
        return training_history_;
    }
    const std::vector<TrainingMetrics>& validation_history() const {
        return validation_history_;
    }

    bool is_training() const {
        return is_training_;
    }

  private:
    nn::Network& network_;
    data::Dataset& dataset_;

    // Training state
    bool is_training_ = false;
    bool should_stop_ = false;

    // History
    std::vector<TrainingMetrics> training_history_;
    std::vector<TrainingMetrics> validation_history_;

    // Callbacks
    OnEpochEndCallback on_epoch_end_;
    OnBatchEndCallback on_batch_end_;
    OnTrainingStartCallback on_training_start_;
    OnTrainingEndCallback on_training_end_;

    // Helper methods
    void train_epoch(const std::vector<int>& train_indices,
                     const TrainingConfig& config,
                     int epoch);
    std::vector<int> create_batches(const std::vector<int>& indices, int batch_size);
};

}  // namespace training