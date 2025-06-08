#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../../core/nn/network.hpp"

struct NetworkConfig {
    struct LayerConfig {
        int neurons;
        std::string activation;
    };

    int input_size;
    std::vector<LayerConfig> layers;
    std::string loss_function;

    // Training configuration
    double learning_rate = 0.01;
    int epochs = 100;
    int batch_size = 32;

    // Metadata
    std::string name;
    std::string description;
    std::string created_date;
};

struct TrainingConfig {
    double learning_rate;
    int epochs;
    int batch_size;
    double validation_split;
    bool shuffle_data;
    int random_seed;

    // Early stopping
    bool use_early_stopping;
    int patience;
    double min_delta;

    // Learning rate scheduling
    bool use_lr_decay;
    double lr_decay_factor;
    int lr_decay_epochs;
};

class ConfigManager {
  public:
    // Network configuration
    static void saveNetworkConfig(const Network& network, const std::string& filename);
    static std::unique_ptr<Network> loadNetworkConfig(const std::string& filename);

    // Trained model serialization
    static void saveTrainedModel(const Network& network, const std::string& filename);
    static std::unique_ptr<Network> loadTrainedModel(const std::string& filename);

    // Training configuration
    static void saveTrainingConfig(const TrainingConfig& config, const std::string& filename);
    static TrainingConfig loadTrainingConfig(const std::string& filename);

    // JSON utilities
    static std::string networkToJson(const Network& network);
    static std::unique_ptr<Network> networkFromJson(const std::string& json);

    // File utilities
    static bool fileExists(const std::string& filename);
    static std::vector<std::string> listConfigFiles(const std::string& directory = ".");

  private:
    static std::string getCurrentDateTime();
    static ActivationType stringToActivationType(const std::string& str);
    static std::string activationTypeToString(ActivationType type);
    static LossType stringToLossType(const std::string& str);
    static std::string lossTypeToString(LossType type);
};