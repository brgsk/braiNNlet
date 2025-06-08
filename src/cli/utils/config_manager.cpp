#include "config_manager.hpp"

#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>

#include "../../core/nn/activations.hpp"
#include "../../core/nn/loss.hpp"

void ConfigManager::saveNetworkConfig(const Network& network, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    file << "# braiNNlet Network Configuration\n";
    file << "# Created: " << getCurrentDateTime() << "\n\n";

    file << "layers:\n";
    for (int i = 0; i < network.layer_count(); ++i) {
        const Layer& layer = network.layer(i);
        if (layer.name() == "Dense") {
            const DenseLayer* dense = static_cast<const DenseLayer*>(&layer);
            file << "  - neurons: " << dense->output_size() << "\n";
            file << "    activation: " << activationTypeToString(dense->activation_type()) << "\n";
        }
    }

    file << "\nloss_function: MSE\n";  // Default for demo
    file << "parameters: " << network.parameter_count() << "\n";

    std::cout << "✅ Network configuration saved to " << filename << "\n";
}

std::unique_ptr<Network> ConfigManager::loadNetworkConfig(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        // Add more debug information
        std::string error_msg = "Cannot open file for reading: " + filename;
        if (filename.empty()) {
            error_msg += " (filename is empty)";
        }
        throw std::runtime_error(error_msg);
    }

    auto network = std::make_unique<Network>();
    std::string line;
    bool in_layers = false;

    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#')
            continue;

        if (line.find("layers:") == 0) {
            in_layers = true;
            continue;
        }

        if (in_layers && line.find("  - neurons:") == 0) {
            // Parse neurons
            size_t pos = line.find(": ");
            if (pos != std::string::npos) {
                int neurons = std::stoi(line.substr(pos + 2));

                // Get next line for activation
                std::getline(file, line);
                pos = line.find(": ");
                if (pos != std::string::npos) {
                    std::string activation_str = line.substr(pos + 2);
                    ActivationType activation = stringToActivationType(activation_str);
                    network->add_layer(neurons, activation);
                }
            }
        }

        if (line.find("loss_function:") == 0) {
            size_t pos = line.find(": ");
            if (pos != std::string::npos) {
                std::string loss_str = line.substr(pos + 2);
                LossType loss_type = stringToLossType(loss_str);
                network->set_loss_function(loss_type);
            }
            in_layers = false;
        }
    }

    return network;
}

void ConfigManager::saveTrainedModel(const Network& network, const std::string& filename) {
    // For now, save as configuration + metadata indicating it's trained
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    file << "# braiNNlet Trained Model\n";
    file << "# Saved: " << getCurrentDateTime() << "\n\n";
    file << "model_type: trained\n";

    // Save network structure
    file << "layers:\n";
    for (int i = 0; i < network.layer_count(); ++i) {
        const Layer& layer = network.layer(i);
        if (layer.name() == "Dense") {
            const DenseLayer* dense = static_cast<const DenseLayer*>(&layer);
            file << "  - neurons: " << dense->output_size() << "\n";
            file << "    activation: " << activationTypeToString(dense->activation_type()) << "\n";
        }
    }

    // Note: In a full implementation, we would serialize the actual weights here
    file << "\n# Note: Weight serialization not implemented in this demo version\n";
    file << "# Weights would be saved as binary data or base64 encoded matrices\n";

    std::cout << "✅ Trained model saved to " << filename << "\n";
}

std::unique_ptr<Network> ConfigManager::loadTrainedModel(const std::string& filename) {
    // For demo, just load as regular config
    // In full implementation, this would also restore weights
    auto network = loadNetworkConfig(filename);
    std::cout
        << "📝 Note: This demo loads network structure only. Weight restoration not implemented.\n";
    return network;
}

void ConfigManager::saveTrainingConfig(const TrainingConfig& config, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    file << "# braiNNlet Training Configuration\n";
    file << "# Created: " << getCurrentDateTime() << "\n\n";

    file << "learning_rate: " << config.learning_rate << "\n";
    file << "epochs: " << config.epochs << "\n";
    file << "batch_size: " << config.batch_size << "\n";
    file << "validation_split: " << config.validation_split << "\n";
    file << "shuffle_data: " << (config.shuffle_data ? "true" : "false") << "\n";
    file << "random_seed: " << config.random_seed << "\n";

    if (config.use_early_stopping) {
        file << "early_stopping:\n";
        file << "  patience: " << config.patience << "\n";
        file << "  min_delta: " << config.min_delta << "\n";
    }

    if (config.use_lr_decay) {
        file << "lr_decay:\n";
        file << "  factor: " << config.lr_decay_factor << "\n";
        file << "  epochs: " << config.lr_decay_epochs << "\n";
    }
}

TrainingConfig ConfigManager::loadTrainingConfig(const std::string& filename) {
    TrainingConfig config = {};

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#')
            continue;

        size_t pos = line.find(": ");
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 2);

            if (key == "learning_rate")
                config.learning_rate = std::stod(value);
            else if (key == "epochs")
                config.epochs = std::stoi(value);
            else if (key == "batch_size")
                config.batch_size = std::stoi(value);
            else if (key == "validation_split")
                config.validation_split = std::stod(value);
            else if (key == "shuffle_data")
                config.shuffle_data = (value == "true");
            else if (key == "random_seed")
                config.random_seed = std::stoi(value);
        }
    }

    return config;
}

bool ConfigManager::fileExists(const std::string& filename) {
    if (filename.empty()) {
        return false;
    }
    std::ifstream file(filename);
    bool exists = file.good();
    file.close();
    return exists;
}

std::vector<std::string> ConfigManager::listConfigFiles(const std::string& directory) {
    std::vector<std::string> files;

    // Simple implementation without filesystem library
    std::cout << "Note: Directory listing not implemented in this demo version\n";

    return files;
}

std::string ConfigManager::getCurrentDateTime() {
    std::time_t now = std::time(nullptr);
    char buffer[100];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return std::string(buffer);
}

ActivationType ConfigManager::stringToActivationType(const std::string& str) {
    std::string lower_str = str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), ::tolower);

    if (lower_str == "relu")
        return ActivationType::ReLU;
    if (lower_str == "sigmoid")
        return ActivationType::Sigmoid;
    if (lower_str == "tanh")
        return ActivationType::Tanh;
    if (lower_str == "linear")
        return ActivationType::Linear;

    return ActivationType::ReLU;  // Default
}

std::string ConfigManager::activationTypeToString(ActivationType type) {
    switch (type) {
        case ActivationType::ReLU:
            return "ReLU";
        case ActivationType::Sigmoid:
            return "Sigmoid";
        case ActivationType::Tanh:
            return "Tanh";
        case ActivationType::Linear:
            return "Linear";
        default:
            return "ReLU";
    }
}

LossType ConfigManager::stringToLossType(const std::string& str) {
    std::string lower_str = str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), ::tolower);

    if (lower_str == "mse" || lower_str == "meansquarederror")
        return LossType::MeanSquaredError;
    if (lower_str == "bce" || lower_str == "binarycrossentropy")
        return LossType::BinaryCrossEntropy;
    if (lower_str == "crossentropy")
        return LossType::CrossEntropy;

    return LossType::MeanSquaredError;  // Default
}

std::string ConfigManager::lossTypeToString(LossType type) {
    switch (type) {
        case LossType::MeanSquaredError:
            return "MSE";
        case LossType::BinaryCrossEntropy:
            return "BinaryCrossEntropy";
        case LossType::CrossEntropy:
            return "CrossEntropy";
        default:
            return "MSE";
    }
}