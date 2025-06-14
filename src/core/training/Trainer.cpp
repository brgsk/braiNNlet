#include "Trainer.hpp"

#include <algorithm>
#include <iostream>

#include "../data/Dataset.hpp"

namespace training {

Trainer::Trainer(nn::Network& network, data::Dataset& dataset)
    : network_(network), dataset_(dataset) {}

void Trainer::train(const TrainingConfig& config) {
    if (!dataset_.is_loaded()) {
        throw std::runtime_error("Dataset not loaded");
    }

    is_training_ = true;
    should_stop_ = false;

    // Clear previous history
    training_history_.clear();
    validation_history_.clear();

    // Create indices and split data
    std::vector<int> all_indices = data::generate_indices(dataset_.size());
    if (config.shuffle) {
        all_indices = data::shuffle_indices(all_indices);
    }

    std::vector<int> train_indices, val_indices;
    data::split_dataset(all_indices, 1.0 - config.validation_split, train_indices, val_indices);

    int batches_per_epoch =
        static_cast<int>((train_indices.size() + config.batch_size - 1) / config.batch_size);

    // Notify training start
    if (on_training_start_) {
        on_training_start_(config.epochs, batches_per_epoch);
    }

    // Training loop
    for (int epoch = 0; epoch < config.epochs && !should_stop_; ++epoch) {
        // Shuffle training data each epoch
        if (config.shuffle) {
            train_indices = data::shuffle_indices(train_indices);
        }

        // Train one epoch
        train_epoch(train_indices, config, epoch);

        // Evaluate on training and validation sets
        TrainingMetrics train_metrics = evaluate(train_indices, config.batch_size);
        train_metrics.epoch = epoch;
        training_history_.push_back(train_metrics);

        TrainingMetrics val_metrics;
        if (!val_indices.empty()) {
            val_metrics = evaluate(val_indices, config.batch_size);
            val_metrics.epoch = epoch;
            validation_history_.push_back(val_metrics);
        }

        // Notify epoch end
        if (on_epoch_end_) {
            on_epoch_end_(epoch, train_metrics, val_metrics);
        }

        // Print progress
        std::cout << "Epoch " << epoch + 1 << "/" << config.epochs
                  << " - Loss: " << train_metrics.loss << " - Accuracy: " << train_metrics.accuracy;

        if (!val_indices.empty()) {
            std::cout << " - Val Loss: " << val_metrics.loss
                      << " - Val Accuracy: " << val_metrics.accuracy;
        }
        std::cout << std::endl;
    }

    is_training_ = false;

    // Notify training end
    if (on_training_end_) {
        on_training_end_(training_history_);
    }
}

void Trainer::train_epoch(const std::vector<int>& train_indices,
                          const TrainingConfig& config,
                          int epoch) {
    network_.set_training(true);

    int num_batches =
        static_cast<int>((train_indices.size() + config.batch_size - 1) / config.batch_size);

    for (int batch_idx = 0; batch_idx < num_batches && !should_stop_; ++batch_idx) {
        // Get batch indices
        int start_idx = batch_idx * config.batch_size;
        int end_idx =
            std::min(start_idx + config.batch_size, static_cast<int>(train_indices.size()));

        std::vector<int> batch_indices;
        for (int i = start_idx; i < end_idx; ++i) {
            batch_indices.push_back(train_indices[i]);
        }

        // Get batch data
        auto [batch_features, batch_targets] = dataset_.get_batch(batch_indices);

        // Forward pass
        nn::Tensor predictions = network_.forward(batch_features);

        // Compute loss
        double loss = network_.compute_loss(predictions, batch_targets);

        // Compute accuracy
        double accuracy = compute_accuracy(predictions, batch_targets);

        // Backward pass
        network_.zero_gradients();
        nn::Tensor loss_grad = network_.compute_loss_gradient(predictions, batch_targets);
        network_.backward(loss_grad);

        // Update weights
        network_.update_weights(config.learning_rate);

        // Create metrics
        TrainingMetrics metrics(loss, accuracy, epoch, batch_idx);

        // Notify batch end
        if (on_batch_end_) {
            on_batch_end_(batch_idx, metrics);
        }

        // Print progress
        if (batch_idx % config.print_every == 0) {
            std::cout << "  Batch " << batch_idx + 1 << "/" << num_batches << " - Loss: " << loss
                      << " - Accuracy: " << accuracy << std::endl;
        }
    }
}

TrainingMetrics Trainer::evaluate(const std::vector<int>& indices, int batch_size) {
    if (indices.empty()) {
        return TrainingMetrics();
    }

    network_.set_training(false);

    double total_loss = 0.0;
    double total_accuracy = 0.0;
    int num_batches = static_cast<int>((indices.size() + batch_size - 1) / batch_size);
    int total_samples = 0;

    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        // Get batch indices
        int start_idx = batch_idx * batch_size;
        int end_idx = std::min(start_idx + batch_size, static_cast<int>(indices.size()));

        std::vector<int> batch_indices;
        for (int i = start_idx; i < end_idx; ++i) {
            batch_indices.push_back(indices[i]);
        }

        // Get batch data
        auto [batch_features, batch_targets] = dataset_.get_batch(batch_indices);

        // Forward pass
        nn::Tensor predictions = network_.forward(batch_features);

        // Compute metrics
        double batch_loss = network_.compute_loss(predictions, batch_targets);
        double batch_accuracy = compute_accuracy(predictions, batch_targets);

        // Accumulate
        int batch_size_actual = static_cast<int>(batch_indices.size());
        total_loss += batch_loss * batch_size_actual;
        total_accuracy += batch_accuracy * batch_size_actual;
        total_samples += batch_size_actual;
    }

    return TrainingMetrics(total_loss / total_samples, total_accuracy / total_samples, 0, 0);
}

double Trainer::compute_accuracy(const nn::Tensor& predictions, const nn::Tensor& targets) {
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        return 0.0;
    }

    int correct = 0;
    int total = predictions.rows();

    for (int i = 0; i < predictions.rows(); ++i) {
        // Find predicted class (argmax)
        int pred_class = 0;
        double max_pred = predictions(i, 0);
        for (int j = 1; j < predictions.cols(); ++j) {
            if (predictions(i, j) > max_pred) {
                max_pred = predictions(i, j);
                pred_class = j;
            }
        }

        // Find true class (argmax)
        int true_class = 0;
        double max_true = targets(i, 0);
        for (int j = 1; j < targets.cols(); ++j) {
            if (targets(i, j) > max_true) {
                max_true = targets(i, j);
                true_class = j;
            }
        }

        if (pred_class == true_class) {
            correct++;
        }
    }

    return static_cast<double>(correct) / total;
}

}  // namespace training