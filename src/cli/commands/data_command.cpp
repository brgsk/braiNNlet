#include "data_command.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

#include "../utils/cli_utils.hpp"

int DataCommand::execute(const std::vector<std::string>& args) {
    if (args.size() < 2) {
        showHelp();
        return 1;
    }

    // Parse arguments
    std::string type;
    int samples = 1000;
    std::string output = "generated_data.csv";

    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--type" && i + 1 < args.size()) {
            type = args[++i];
        } else if (args[i] == "--samples" && i + 1 < args.size()) {
            samples = std::stoi(args[++i]);
        } else if (args[i] == "--output" && i + 1 < args.size()) {
            output = args[++i];
        } else if (args[i] == "--help") {
            showHelp();
            return 0;
        }
    }

    if (type.empty()) {
        std::cout << "❌ Data type not specified. Use --type "
                     "[regression|classification|multiclass|mnist]\n";
        return 1;
    }

    try {
        if (type == "mnist") {
            std::cout << "🔄 Loading MNIST dataset...\n";
            auto dataset = loadMnistDataset();
            if (dataset && dataset->is_loaded()) {
                std::cout << "✅ MNIST dataset loaded: " << dataset->size() << " samples\n";

                // Convert to CSV format if requested
                if (output != "generated_data.csv") {
                    auto [inputs, targets] = datasetToVectors(*dataset, samples);
                    saveToCSV(inputs, targets, output);
                    std::cout << "✅ Saved " << inputs.size() << " MNIST samples to " << output
                              << "\n";
                }
            } else {
                std::cout << "❌ Failed to load MNIST dataset\n";
                return 1;
            }
        } else {
            std::cout << "🔄 Generating " << samples << " samples of " << type << " data...\n";

            if (type == "regression") {
                auto [inputs, targets] = generateRegressionData(samples);
                saveToCSV(inputs, targets, output);
            } else if (type == "classification") {
                auto [inputs, targets] = generateClassificationData(samples);
                saveToCSV(inputs, targets, output);
            } else if (type == "multiclass") {
                auto [inputs, targets] = generateMultiClassData(samples, 3);  // Default 3 classes
                saveToCSV(inputs, targets, output);
            } else {
                std::cout << "❌ Unknown data type: " << type << "\n";
                return 1;
            }
        }

        std::cout << "✅ Data saved to " << output << "\n";
        return 0;

    } catch (const std::exception& e) {
        std::cout << "❌ Error generating data: " << e.what() << "\n";
        return 1;
    }
}

std::pair<std::vector<Matrix>, std::vector<Matrix>> DataCommand::generateRegressionData(
    int samples) {
    std::vector<Matrix> inputs;
    std::vector<Matrix> targets;

    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dis(-2.0, 2.0);

    for (int i = 0; i < samples; ++i) {
        Matrix input(1, 3);
        input << dis(gen), dis(gen), dis(gen);

        // Target: y = 0.5*x1 + 0.3*x2 + 0.1*x3 + 0.2 + noise
        double target = 0.5 * input(0, 0) + 0.3 * input(0, 1) + 0.1 * input(0, 2) + 0.2;
        target += 0.05 * dis(gen);  // Add small noise

        Matrix target_matrix(1, 1);
        target_matrix << target;

        inputs.push_back(input);
        targets.push_back(target_matrix);
    }

    return {inputs, targets};
}

std::pair<std::vector<Matrix>, std::vector<Matrix>> DataCommand::generateClassificationData(
    int samples) {
    std::vector<Matrix> inputs;
    std::vector<Matrix> targets;

    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    for (int i = 0; i < samples; ++i) {
        Matrix input(1, 2);
        input << dis(gen), dis(gen);

        // XOR-like target: positive if x1*x2 > 0, negative otherwise
        double target = (input(0, 0) * input(0, 1) > 0) ? 1.0 : 0.0;

        Matrix target_matrix(1, 1);
        target_matrix << target;

        inputs.push_back(input);
        targets.push_back(target_matrix);
    }

    return {inputs, targets};
}

std::pair<std::vector<Matrix>, std::vector<Matrix>> DataCommand::generateMultiClassData(
    int samples, int num_classes) {
    std::vector<Matrix> inputs;
    std::vector<Matrix> targets;

    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dis(-2.0, 2.0);

    for (int i = 0; i < samples; ++i) {
        Matrix input(1, 2);
        input << dis(gen), dis(gen);

        // Create concentric circles pattern
        double radius = std::sqrt(input(0, 0) * input(0, 0) + input(0, 1) * input(0, 1));
        int target_class = static_cast<int>(radius * num_classes / 3.0) % num_classes;

        Matrix target_matrix(1, 1);
        target_matrix << static_cast<double>(target_class);

        inputs.push_back(input);
        targets.push_back(target_matrix);
    }

    return {inputs, targets};
}

void DataCommand::saveToCSV(const std::vector<Matrix>& inputs,
                            const std::vector<Matrix>& targets,
                            const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    // Write header
    int input_dims = inputs.empty() ? 0 : inputs[0].cols();
    int target_dims = targets.empty() ? 0 : targets[0].cols();

    for (int i = 0; i < input_dims; ++i) {
        file << "input_" << i;
        if (i < input_dims - 1 || target_dims > 0)
            file << ",";
    }

    for (int i = 0; i < target_dims; ++i) {
        file << "target_" << i;
        if (i < target_dims - 1)
            file << ",";
    }
    file << "\n";

    // Write data
    for (size_t i = 0; i < inputs.size(); ++i) {
        // Input features
        for (int j = 0; j < inputs[i].cols(); ++j) {
            file << inputs[i](0, j);
            if (j < inputs[i].cols() - 1 || targets[i].cols() > 0)
                file << ",";
        }

        // Target values
        for (int j = 0; j < targets[i].cols(); ++j) {
            file << targets[i](0, j);
            if (j < targets[i].cols() - 1)
                file << ",";
        }
        file << "\n";
    }
}

std::pair<std::vector<Matrix>, std::vector<Matrix>> DataCommand::loadFromCSV(
    const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    std::vector<Matrix> inputs;
    std::vector<Matrix> targets;
    std::string line;

    // Skip header
    if (!std::getline(file, line)) {
        throw std::runtime_error("Empty CSV file");
    }

    // Count columns from header
    std::stringstream header_ss(line);
    std::string token;
    int input_cols = 0, target_cols = 0;

    while (std::getline(header_ss, token, ',')) {
        if (token.find("input_") == 0)
            input_cols++;
        else if (token.find("target_") == 0)
            target_cols++;
    }

    // Read data
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        Matrix input(1, input_cols);
        Matrix target(1, target_cols);

        // Read input values
        for (int i = 0; i < input_cols; ++i) {
            if (!std::getline(ss, token, ',')) {
                throw std::runtime_error("Invalid CSV format");
            }
            input(0, i) = std::stod(token);
        }

        // Read target values
        for (int i = 0; i < target_cols; ++i) {
            if (!std::getline(ss, token, ',')) {
                throw std::runtime_error("Invalid CSV format");
            }
            target(0, i) = std::stod(token);
        }

        inputs.push_back(input);
        targets.push_back(target);
    }

    return {inputs, targets};
}

void DataCommand::shuffleData(std::vector<Matrix>& inputs, std::vector<Matrix>& targets) {
    if (inputs.size() != targets.size()) {
        throw std::runtime_error("Input and target vectors must have same size");
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    // Create indices and shuffle them
    std::vector<size_t> indices(inputs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);

    // Reorder based on shuffled indices
    std::vector<Matrix> shuffled_inputs, shuffled_targets;
    for (size_t idx : indices) {
        shuffled_inputs.push_back(inputs[idx]);
        shuffled_targets.push_back(targets[idx]);
    }

    inputs = std::move(shuffled_inputs);
    targets = std::move(shuffled_targets);
}

std::pair<std::vector<Matrix>, std::vector<Matrix>> DataCommand::splitData(
    const std::vector<Matrix>& inputs, const std::vector<Matrix>& targets, double train_ratio) {
    size_t split_point = static_cast<size_t>(inputs.size() * train_ratio);

    std::vector<Matrix> train_inputs(inputs.begin(), inputs.begin() + split_point);
    std::vector<Matrix> train_targets(targets.begin(), targets.begin() + split_point);

    return {train_inputs, train_targets};
}

std::unique_ptr<data::Dataset> DataCommand::loadMnistDataset(const std::string& path) {
    auto dataset = std::make_unique<data::MnistDataset>();

    if (dataset->load(path)) {
        return dataset;
    }

    return nullptr;
}

bool DataCommand::checkMnistAvailability() {
    // Check if MNIST files exist in the default location
    std::string mnist_dir = "src/core/data/MNIST";
    std::string train_images = mnist_dir + "/train-images.idx3-ubyte";
    std::string train_labels = mnist_dir + "/train-labels.idx1-ubyte";

    std::ifstream images_file(train_images);
    std::ifstream labels_file(train_labels);

    return images_file.good() && labels_file.good();
}

std::pair<std::vector<Matrix>, std::vector<Matrix>> DataCommand::datasetToVectors(
    const data::Dataset& dataset, int max_samples) {
    std::vector<Matrix> inputs;
    std::vector<Matrix> targets;

    int num_samples = dataset.size();
    if (max_samples > 0 && max_samples < num_samples) {
        num_samples = max_samples;
    }

    inputs.reserve(num_samples);
    targets.reserve(num_samples);

    for (int i = 0; i < num_samples; ++i) {
        auto sample = dataset.get(i);

        // Convert Tensor to Matrix (assuming single row tensors)
        Matrix input_matrix(1, sample.features.cols());
        Matrix target_matrix(1, sample.label.cols());

        for (int j = 0; j < sample.features.cols(); ++j) {
            input_matrix(0, j) = sample.features(0, j);
        }

        for (int j = 0; j < sample.label.cols(); ++j) {
            target_matrix(0, j) = sample.label(0, j);
        }

        inputs.push_back(input_matrix);
        targets.push_back(target_matrix);
    }

    return {inputs, targets};
}

void DataCommand::showHelp() {
    std::cout << "\nbraiNNlet CLI - Data Generation Command\n";
    std::cout << "=====================================\n\n";
    std::cout << "Usage: braiNNlet-cli generate-data [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --type TYPE         Type of data to generate/load "
                 "[regression|classification|multiclass|mnist]\n";
    std::cout << "  --samples N         Number of samples to generate/export (default: 1000)\n";
    std::cout << "  --output FILE       Output CSV filename (default: generated_data.csv)\n";
    std::cout << "  --help             Show this help message\n\n";
    std::cout << "Data Types:\n";
    std::cout << "  regression          Synthetic regression data (3 features -> 1 target)\n";
    std::cout << "  classification      Synthetic binary classification data (XOR-like)\n";
    std::cout << "  multiclass          Synthetic multi-class data (concentric circles)\n";
    std::cout << "  mnist               Load MNIST handwritten digits dataset\n\n";
    std::cout << "Examples:\n";
    std::cout
        << "  braiNNlet-cli generate-data --type regression --samples 500 --output train.csv\n";
    std::cout << "  braiNNlet-cli generate-data --type classification --samples 1000\n";
    std::cout << "  braiNNlet-cli generate-data --type mnist --samples 1000 --output "
                 "mnist_subset.csv\n\n";

    // Show MNIST availability
    if (checkMnistAvailability()) {
        std::cout << "✅ MNIST dataset files found and available\n";
    } else {
        std::cout << "⚠️  MNIST dataset files not found - will use dummy data if requested\n";
        std::cout << "   Place MNIST files in: src/core/data/MNIST/\n";
    }
    std::cout << "\n";
}