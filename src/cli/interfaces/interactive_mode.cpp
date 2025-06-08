#include "interactive_mode.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#include "../commands/data_command.hpp"
#include "../utils/cli_utils.hpp"

int InteractiveMode::run() {
    running_ = true;

    CliUtils::clearScreen();
    std::cout << "\n";
    std::cout << "██████╗ ██████╗  █████╗ ██╗███╗   ██╗███╗   ██╗██╗     ███████╗████████╗\n";
    std::cout << "██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║████╗  ██║██║     ██╔════╝╚══██╔══╝\n";
    std::cout << "██████╔╝██████╔╝███████║██║██╔██╗ ██║██╔██╗ ██║██║     █████╗     ██║   \n";
    std::cout << "██╔══██╗██╔══██╗██╔══██║██║██║╚██╗██║██║╚██╗██║██║     ██╔══╝     ██║   \n";
    std::cout << "██████╔╝██║  ██║██║  ██║██║██║ ╚████║██║ ╚████║███████╗███████╗   ██║   \n";
    std::cout << "╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝╚══════╝╚══════╝   ╚═╝   \n";
    std::cout << "\n🧠 Welcome to braiNNlet CLI - Interactive Mode\n";
    std::cout << "==============================================\n\n";

    std::cout << "This interactive mode will guide you through creating,\n";
    std::cout << "training, and evaluating neural networks.\n\n";

    CliUtils::waitForEnter();

    while (running_) {
        try {
            showMainMenu();
        } catch (const std::exception& e) {
            std::cout << "❌ Error: " << e.what() << "\n";
            CliUtils::waitForEnter("Press Enter to continue...");
        }
    }

    std::cout << "👋 Thank you for using braiNNlet CLI!\n";
    return 0;
}

void InteractiveMode::showMainMenu() {
    CliUtils::clearScreen();
    CliUtils::printSectionHeader("braiNNlet CLI - Main Menu");

    if (network_) {
        std::cout << "🟢 Network loaded: " << network_->layer_count() << " layers, "
                  << network_->parameter_count() << " parameters\n\n";
    } else {
        std::cout << "🔴 No network loaded\n\n";
    }

    std::cout << "Available actions:\n";
    std::cout << "1. Create new network\n";
    std::cout << "2. Load existing network\n";
    std::cout << "3. Show network information\n";
    std::cout << "4. Train network\n";
    std::cout << "5. Evaluate network\n";
    std::cout << "6. Generate training data\n";
    std::cout << "7. Save network\n";
    std::cout << "8. MNIST operations\n";
    std::cout << "9. Manage configurations\n";
    std::cout << "10. Exit\n\n";

    int choice = getUserChoice("Select an option", 1, 10);

    switch (choice) {
        case 1:
            createNetwork();
            break;
        case 2:
            loadNetwork();
            break;
        case 3:
            showNetworkInfo();
            break;
        case 4:
            trainNetwork();
            break;
        case 5:
            evaluateNetwork();
            break;
        case 6:
            generateData();
            break;
        case 7:
            saveNetwork();
            break;
        case 8:
            mnistOperations();
            break;
        case 9:
            manageConfigs();
            break;
        case 10:
            running_ = false;
            break;
    }
}

void InteractiveMode::createNetwork() {
    CliUtils::printSectionHeader("Create New Network");

    std::cout << "Let's build your neural network step by step.\n\n";

    // Get input size
    int input_size = getUserChoice("Enter input size (number of features)", 1, 1000);

    // Create network
    network_ = std::make_unique<Network>();

    std::cout << "\nNow let's add hidden layers. You can add multiple layers.\n";
    std::cout << "For the output layer, we'll configure it based on your problem type.\n\n";

    // Add hidden layers
    while (true) {
        int neurons = getUserChoice("Number of neurons in this layer", 1, 1000);

        std::cout << "\nActivation functions:\n";
        std::cout << "1. ReLU (recommended for hidden layers)\n";
        std::cout << "2. Sigmoid\n";
        std::cout << "3. Tanh\n";
        std::cout << "4. Linear\n";

        int activation_choice = getUserChoice("Select activation function", 1, 4);
        ActivationType activation;
        switch (activation_choice) {
            case 1:
                activation = ActivationType::ReLU;
                break;
            case 2:
                activation = ActivationType::Sigmoid;
                break;
            case 3:
                activation = ActivationType::Tanh;
                break;
            case 4:
                activation = ActivationType::Linear;
                break;
        }

        network_->add_layer(neurons, activation);

        std::cout << "✅ Added layer: " << neurons << " neurons, "
                  << (activation_choice == 1   ? "ReLU"
                      : activation_choice == 2 ? "Sigmoid"
                      : activation_choice == 3 ? "Tanh"
                                               : "Linear")
                  << " activation\n\n";

        if (!getUserBool("Add another layer? (y/n)")) {
            break;
        }
    }

    // Configure output layer
    std::cout << "\nNow let's configure the output layer:\n";
    std::cout << "1. Regression (continuous values) - 1 output, Linear activation\n";
    std::cout << "2. Binary Classification - 1 output, Sigmoid activation\n";
    std::cout << "3. Multi-class Classification - Multiple outputs, Linear activation\n";

    int problem_type = getUserChoice("Select problem type", 1, 3);

    int output_neurons;
    ActivationType output_activation;
    LossType loss_type;

    switch (problem_type) {
        case 1:  // Regression
            output_neurons = 1;
            output_activation = ActivationType::Linear;
            loss_type = LossType::MeanSquaredError;
            break;
        case 2:  // Binary Classification
            output_neurons = 1;
            output_activation = ActivationType::Sigmoid;
            loss_type = LossType::BinaryCrossEntropy;
            break;
        case 3:  // Multi-class
            output_neurons = getUserChoice("Number of classes", 2, 100);
            output_activation = ActivationType::Linear;
            loss_type = LossType::CrossEntropy;
            break;
    }

    network_->add_layer(output_neurons, output_activation);
    network_->set_loss_function(loss_type);

    std::cout << "✅ Network created successfully!\n";
    std::cout << "📊 Total layers: " << network_->layer_count() << "\n";
    std::cout << "🔢 Total parameters: " << network_->parameter_count() << "\n\n";

    CliUtils::waitForEnter();
}

void InteractiveMode::loadNetwork() {
    CliUtils::printSectionHeader("Load Network");

    // Show available config files first
    std::vector<std::string> potential_files = {"temp_network.cfg", "network.cfg", "my_network.cfg",
                                                "trained_model.trained"};

    std::vector<std::string> found_files;
    for (const auto& file : potential_files) {
        if (ConfigManager::fileExists(file)) {
            found_files.push_back(file);
        }
    }

    if (!found_files.empty()) {
        std::cout << "Available configuration files:\n";
        for (size_t i = 0; i < found_files.size(); ++i) {
            std::cout << "  • " << found_files[i] << "\n";
        }
        std::cout << "\n";
    }

    std::string filename;
    while (true) {
        filename = getUserInput("Enter network configuration filename (.cfg)");
        if (!filename.empty()) {
            break;
        }
        std::cout << "❌ Please enter a valid filename.\n";
    }

    try {
        // Check if file exists before attempting to load
        if (!ConfigManager::fileExists(filename)) {
            std::cout << "❌ File '" << filename << "' does not exist in the current directory.\n";
            if (!found_files.empty()) {
                std::cout << "💡 Try one of the available files listed above.\n";
            } else {
                std::cout << "💡 Make sure you're running the CLI from the correct directory.\n";
                std::cout << "💡 You can also use an absolute path to the config file.\n";
            }
            std::cout << "\n";
            CliUtils::waitForEnter();
            return;
        }

        network_ = ConfigManager::loadNetworkConfig(filename);
        std::cout << "✅ Network loaded successfully from '" << filename << "'!\n";
        std::cout << "📊 Layers: " << network_->layer_count() << "\n";
        std::cout << "🔢 Parameters: " << network_->parameter_count() << "\n\n";
    } catch (const std::exception& e) {
        std::cout << "❌ Failed to load network: " << e.what() << "\n\n";
    }

    CliUtils::waitForEnter();
}

void InteractiveMode::showNetworkInfo() {
    if (!network_) {
        std::cout << "❌ No network loaded. Please create or load a network first.\n";
        CliUtils::waitForEnter();
        return;
    }

    CliUtils::printSectionHeader("Network Information");

    std::cout << "📊 Network Architecture:\n";
    std::cout << "  • Layers: " << network_->layer_count() << "\n";
    std::cout << "  • Parameters: " << network_->parameter_count() << "\n";
    std::cout << "  • Training mode: " << (network_->is_training() ? "ON" : "OFF") << "\n\n";

    std::cout << "🏗️  Layer Details:\n";
    for (int i = 0; i < network_->layer_count(); ++i) {
        const Layer& layer = network_->layer(i);
        if (layer.name() == "Dense") {
            const DenseLayer* dense = static_cast<const DenseLayer*>(&layer);
            std::cout << "  Layer " << (i + 1) << ": " << dense->output_size() << " neurons";
            // Note: In a full implementation, we'd show activation function here
            std::cout << "\n";
        }
    }

    std::cout << "\n";
    CliUtils::waitForEnter();
}

void InteractiveMode::trainNetwork() {
    if (!network_) {
        std::cout << "❌ No network loaded. Please create or load a network first.\n";
        CliUtils::waitForEnter();
        return;
    }

    CliUtils::printSectionHeader("Train Network");

    // Load or generate data
    std::cout << "Training data options:\n";
    std::cout << "1. Load from CSV file\n";
    std::cout << "2. Generate synthetic data\n";
    std::cout << "3. Load MNIST dataset\n";

    int data_choice = getUserChoice("Select data source", 1, 3);

    std::vector<Matrix> inputs, targets;

    if (data_choice == 1) {
        std::string filename = getUserInput("Enter CSV filename");
        try {
            auto [loaded_inputs, loaded_targets] = DataCommand::loadFromCSV(filename);
            inputs = loaded_inputs;
            targets = loaded_targets;
            std::cout << "✅ Loaded " << inputs.size() << " samples from " << filename << "\n\n";
        } catch (const std::exception& e) {
            std::cout << "❌ Failed to load data: " << e.what() << "\n";
            CliUtils::waitForEnter();
            return;
        }
    } else if (data_choice == 2) {
        std::cout << "Data generation options:\n";
        std::cout << "1. Regression data\n";
        std::cout << "2. Binary classification data\n";
        std::cout << "3. Multi-class classification data\n";

        int gen_choice = getUserChoice("Select data type", 1, 3);
        int samples = getUserChoice("Number of samples", 100, 10000);

        switch (gen_choice) {
            case 1:
                std::tie(inputs, targets) = DataCommand::generateRegressionData(samples);
                break;
            case 2:
                std::tie(inputs, targets) = DataCommand::generateClassificationData(samples);
                break;
            case 3:
                std::tie(inputs, targets) = DataCommand::generateMultiClassData(samples, 3);
                break;
        }

        std::cout << "✅ Generated " << inputs.size() << " samples\n\n";
    } else if (data_choice == 3) {
        // MNIST dataset
        std::cout << "🔄 Loading MNIST dataset...\n";
        auto dataset = DataCommand::loadMnistDataset();

        if (dataset && dataset->is_loaded()) {
            std::cout << "✅ MNIST dataset loaded: " << dataset->size() << " samples\n";
            std::cout << "📊 Input size: " << dataset->input_size()
                      << ", Output size: " << dataset->output_size() << "\n\n";

            // Validate network architecture for MNIST
            if (network_->layer_count() == 0) {
                std::cout << "❌ Network has no layers. Please create a network first.\n";
                CliUtils::waitForEnter();
                return;
            }

            // Check if network is compatible with MNIST (784 inputs, 10 outputs)
            // The network will auto-adjust input size, but let's check output size
            int expected_outputs = 10;
            int actual_outputs = network_->layer(network_->layer_count() - 1).output_size();

            if (actual_outputs != expected_outputs) {
                std::cout << "⚠️  Network output size (" << actual_outputs
                          << ") doesn't match MNIST classes (" << expected_outputs << ")\n";
                std::cout << "Training may not work correctly.\n\n";
            }

            // Set appropriate loss function for multi-class classification
            std::cout << "🔧 Setting CrossEntropy loss for MNIST classification...\n";
            network_->set_loss_function(LossType::CrossEntropy);

            // Use the new Trainer class for MNIST
            training::Trainer trainer(*network_, *dataset);

            // Configure training for MNIST
            training::TrainingConfig trainer_config;
            trainer_config.epochs = getUserChoice("Number of epochs", 1, 100);
            trainer_config.batch_size = getUserChoice("Batch size", 1, 256);

            // Suggest better learning rate for MNIST with CrossEntropy
            std::cout << "💡 Recommended learning rate for MNIST: 0.005-0.01 (default: 0.01)\n";
            trainer_config.learning_rate = getUserDouble("Learning rate", 0.001, 0.1);
            trainer_config.validation_split = getUserDouble("Validation split (0.0-0.5)", 0.0, 0.5);
            trainer_config.shuffle = getUserBool("Shuffle data each epoch? (y/n)");
            trainer_config.print_every =
                std::max(1, (dataset->size() / trainer_config.batch_size) / 10);

            std::cout << "\n🚀 Starting MNIST training with Trainer class...\n";
            std::cout << "📋 Config: " << trainer_config.epochs << " epochs, batch size "
                      << trainer_config.batch_size << ", learning rate "
                      << trainer_config.learning_rate << "\n\n";

            // Train using the Trainer class
            trainer.train(trainer_config);

            std::cout << "\n✅ MNIST training completed!\n";
            CliUtils::waitForEnter();
            return;
        } else {
            std::cout << "❌ Failed to load MNIST dataset. Falling back to synthetic data.\n";
            std::tie(inputs, targets) = DataCommand::generateMultiClassData(1000, 10);
            std::cout << "✅ Generated " << inputs.size() << " samples\n\n";
        }
    }

    // Configure training (for CSV and synthetic data)
    TrainingConfig config = configureTraining();

    // Split data
    DataCommand::shuffleData(inputs, targets);
    auto [train_inputs, train_targets] =
        DataCommand::splitData(inputs, targets, 1.0 - config.validation_split);

    size_t val_start = train_inputs.size();
    std::vector<Matrix> val_inputs(inputs.begin() + val_start, inputs.end());
    std::vector<Matrix> val_targets(targets.begin() + val_start, targets.end());

    std::cout << "📊 Training set: " << train_inputs.size() << " samples\n";
    std::cout << "📊 Validation set: " << val_inputs.size() << " samples\n\n";

    // Train the network
    runTrainingLoop(train_inputs, train_targets, val_inputs, val_targets, config);

    CliUtils::waitForEnter();
}

TrainingConfig InteractiveMode::configureTraining() {
    std::cout << "🔧 Training Configuration:\n\n";

    TrainingConfig config;
    config.learning_rate = getUserDouble("Learning rate", 0.0001, 1.0);
    config.epochs = getUserChoice("Number of epochs", 1, 10000);
    config.batch_size = getUserChoice("Batch size", 1, 1000);
    config.validation_split = getUserDouble("Validation split (0.0-0.5)", 0.0, 0.5);
    config.shuffle_data = getUserBool("Shuffle data each epoch? (y/n)");

    std::cout << "\n✅ Training configuration set!\n\n";
    return config;
}

void InteractiveMode::runTrainingLoop(const std::vector<Matrix>& train_inputs,
                                      const std::vector<Matrix>& train_targets,
                                      const std::vector<Matrix>& val_inputs,
                                      const std::vector<Matrix>& val_targets,
                                      const TrainingConfig& config) {
    std::cout << "🚀 Starting training...\n\n";
    network_->set_training(true);

    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        double total_loss = 0.0;
        int num_batches = 0;

        // Training loop
        for (size_t i = 0; i < train_inputs.size(); i += config.batch_size) {
            size_t batch_end = std::min(i + config.batch_size, train_inputs.size());

            // Create batch
            std::vector<Matrix> batch_inputs(train_inputs.begin() + i,
                                             train_inputs.begin() + batch_end);
            std::vector<Matrix> batch_targets(train_targets.begin() + i,
                                              train_targets.begin() + batch_end);

            // Forward pass
            std::vector<Matrix> predictions;
            for (const auto& input : batch_inputs) {
                Tensor input_tensor(input);
                Tensor pred_tensor = network_->forward(input_tensor);
                predictions.push_back(pred_tensor.data());
            }

            // Calculate loss
            double batch_loss = 0.0;
            for (size_t j = 0; j < predictions.size(); ++j) {
                Tensor pred_tensor(predictions[j]);
                Tensor target_tensor(batch_targets[j]);
                batch_loss += network_->compute_loss(pred_tensor, target_tensor);
            }
            batch_loss /= predictions.size();
            total_loss += batch_loss;
            num_batches++;

            // Backward pass
            for (size_t j = 0; j < predictions.size(); ++j) {
                Tensor pred_tensor(predictions[j]);
                Tensor target_tensor(batch_targets[j]);
                Tensor loss_grad = network_->compute_loss_gradient(pred_tensor, target_tensor);
                network_->backward(loss_grad);
            }

            // Update weights
            network_->update_weights(config.learning_rate);
            network_->zero_gradients();
        }

        double avg_loss = total_loss / num_batches;

        // Validation
        double val_loss = 0.0;
        if (!val_inputs.empty()) {
            network_->set_training(false);
            for (size_t i = 0; i < val_inputs.size(); ++i) {
                Tensor input_tensor(val_inputs[i]);
                Tensor target_tensor(val_targets[i]);
                Tensor pred = network_->forward(input_tensor);
                val_loss += network_->compute_loss(pred, target_tensor);
            }
            val_loss /= val_inputs.size();
            network_->set_training(true);
        }

        // Display progress
        displayTrainingProgress(epoch + 1, config.epochs, avg_loss, val_loss);

        // Simple early stopping
        if (epoch > 10 && avg_loss < 1e-6) {
            std::cout << "\n🎯 Converged! Training stopped early.\n";
            break;
        }
    }

    std::cout << "\n✅ Training completed!\n";
}

void InteractiveMode::evaluateNetwork() {
    if (!network_) {
        std::cout << "❌ No network loaded. Please create or load a network first.\n";
        CliUtils::waitForEnter();
        return;
    }

    CliUtils::printSectionHeader("Evaluate Network");

    std::string filename = getUserInput("Enter test data CSV filename");

    try {
        auto [test_inputs, test_targets] = DataCommand::loadFromCSV(filename);

        std::cout << "🔄 Evaluating on " << test_inputs.size() << " samples...\n\n";

        network_->set_training(false);

        std::vector<Matrix> predictions;
        double total_loss = 0.0;

        for (size_t i = 0; i < test_inputs.size(); ++i) {
            Tensor input_tensor(test_inputs[i]);
            Tensor target_tensor(test_targets[i]);
            Tensor pred = network_->forward(input_tensor);
            predictions.push_back(pred.data());
            total_loss += network_->compute_loss(pred, target_tensor);
        }

        double avg_loss = total_loss / test_inputs.size();

        std::cout << "📊 Evaluation Results:\n";
        std::cout << "  • Average Loss: " << CliUtils::formatNumber(avg_loss, 6) << "\n";
        std::cout << "  • Samples: " << test_inputs.size() << "\n\n";

        displayEvaluationResults(predictions, test_targets);

    } catch (const std::exception& e) {
        std::cout << "❌ Failed to evaluate: " << e.what() << "\n";
    }

    CliUtils::waitForEnter();
}

void InteractiveMode::generateData() {
    CliUtils::printSectionHeader("Generate Training Data");

    std::cout << "Data types:\n";
    std::cout << "1. Regression data (continuous targets)\n";
    std::cout << "2. Binary classification data\n";
    std::cout << "3. Multi-class classification data\n";

    int type_choice = getUserChoice("Select data type", 1, 3);
    int samples = getUserChoice("Number of samples", 100, 50000);
    std::string filename = getUserInput("Output filename (.csv)");

    try {
        std::vector<Matrix> inputs, targets;

        switch (type_choice) {
            case 1:
                std::tie(inputs, targets) = DataCommand::generateRegressionData(samples);
                break;
            case 2:
                std::tie(inputs, targets) = DataCommand::generateClassificationData(samples);
                break;
            case 3:
                int num_classes = getUserChoice("Number of classes", 2, 10);
                std::tie(inputs, targets) =
                    DataCommand::generateMultiClassData(samples, num_classes);
                break;
        }

        DataCommand::saveToCSV(inputs, targets, filename);
        std::cout << "✅ Generated and saved " << samples << " samples to " << filename << "\n";

    } catch (const std::exception& e) {
        std::cout << "❌ Failed to generate data: " << e.what() << "\n";
    }

    CliUtils::waitForEnter();
}

void InteractiveMode::saveNetwork() {
    if (!network_) {
        std::cout << "❌ No network loaded. Please create or load a network first.\n";
        CliUtils::waitForEnter();
        return;
    }

    CliUtils::printSectionHeader("Save Network");

    std::string filename = getUserInput("Enter filename to save configuration (.cfg)");

    try {
        ConfigManager::saveNetworkConfig(*network_, filename);
        std::cout << "✅ Network configuration saved to " << filename << "\n";
    } catch (const std::exception& e) {
        std::cout << "❌ Failed to save network: " << e.what() << "\n";
    }

    CliUtils::waitForEnter();
}

void InteractiveMode::mnistOperations() {
    CliUtils::printSectionHeader("MNIST Operations");

    std::cout << "MNIST dataset operations:\n";
    std::cout << "1. Check MNIST availability\n";
    std::cout << "2. Load and inspect MNIST dataset\n";
    std::cout << "3. Train network on MNIST\n";
    std::cout << "4. Export MNIST samples to CSV\n";
    std::cout << "5. Back to main menu\n\n";

    int choice = getUserChoice("Select operation", 1, 5);

    switch (choice) {
        case 1: {
            // Check MNIST availability
            if (DataCommand::checkMnistAvailability()) {
                std::cout << "✅ MNIST dataset files found and available!\n";
                std::cout << "📁 Location: src/core/data/MNIST/\n";
            } else {
                std::cout << "❌ MNIST dataset files not found.\n";
                std::cout << "📁 Expected location: src/core/data/MNIST/\n";
                std::cout << "📋 Required files:\n";
                std::cout << "   • train-images.idx3-ubyte\n";
                std::cout << "   • train-labels.idx1-ubyte\n";
                std::cout << "💡 Download from: http://yann.lecun.com/exdb/mnist/\n";
            }
            break;
        }
        case 2: {
            // Load and inspect MNIST
            std::cout << "🔄 Loading MNIST dataset...\n";
            auto dataset = DataCommand::loadMnistDataset();

            if (dataset && dataset->is_loaded()) {
                std::cout << "✅ MNIST dataset loaded successfully!\n";
                std::cout << "📊 Dataset info:\n";
                std::cout << "   • Total samples: " << dataset->size() << "\n";
                std::cout << "   • Input size: " << dataset->input_size() << " (28x28 pixels)\n";
                std::cout << "   • Output size: " << dataset->output_size()
                          << " (10 classes: 0-9)\n";

                // Show a few sample labels
                std::cout << "\n🔍 First 10 sample labels: ";
                for (int i = 0; i < std::min(10, dataset->size()); ++i) {
                    auto sample = dataset->get(i);
                    // Find the class (argmax of one-hot)
                    int label = 0;
                    double max_val = sample.label(0, 0);
                    for (int j = 1; j < sample.label.cols(); ++j) {
                        if (sample.label(0, j) > max_val) {
                            max_val = sample.label(0, j);
                            label = j;
                        }
                    }
                    std::cout << label << " ";
                }
                std::cout << "\n";
            } else {
                std::cout << "❌ Failed to load MNIST dataset.\n";
            }
            break;
        }
        case 3: {
            // Train network on MNIST
            if (!network_) {
                std::cout << "❌ No network loaded. Please create or load a network first.\n";
                std::cout << "💡 For MNIST, create a network with:\n";
                std::cout << "   • Input size: 784 (28x28 pixels)\n";
                std::cout << "   • Output size: 10 (digits 0-9)\n";
                break;
            }

            std::cout << "🔄 Loading MNIST dataset for training...\n";
            auto dataset = DataCommand::loadMnistDataset();

            if (dataset && dataset->is_loaded()) {
                std::cout << "✅ MNIST dataset loaded: " << dataset->size() << " samples\n";

                // Verify network compatibility
                if (dataset->input_size() != 784) {
                    std::cout << "⚠️  Warning: Expected input size 784, but dataset has "
                              << dataset->input_size() << "\n";
                }
                if (dataset->output_size() != 10) {
                    std::cout << "⚠️  Warning: Expected output size 10, but dataset has "
                              << dataset->output_size() << "\n";
                }

                // Use the new Trainer class
                training::Trainer trainer(*network_, *dataset);

                // Configure training
                training::TrainingConfig config;
                config.epochs = getUserChoice("Number of epochs", 1, 50);
                config.batch_size = getUserChoice("Batch size", 16, 256);
                config.learning_rate = getUserDouble("Learning rate", 0.0001, 0.1);
                config.validation_split = getUserDouble("Validation split (0.0-0.5)", 0.0, 0.5);
                config.shuffle = true;
                config.print_every = std::max(1, (dataset->size() / config.batch_size) / 10);

                std::cout << "\n🚀 Starting MNIST training...\n";
                std::cout << "📋 Config: " << config.epochs << " epochs, batch size "
                          << config.batch_size << ", learning rate " << config.learning_rate
                          << "\n\n";

                // Train the network
                trainer.train(config);

                std::cout << "\n✅ MNIST training completed!\n";
            } else {
                std::cout << "❌ Failed to load MNIST dataset.\n";
            }
            break;
        }
        case 4: {
            // Export MNIST to CSV
            std::cout << "🔄 Loading MNIST dataset...\n";
            auto dataset = DataCommand::loadMnistDataset();

            if (dataset && dataset->is_loaded()) {
                int num_samples =
                    getUserChoice("Number of samples to export", 100, dataset->size());
                std::string filename = getUserInput("Output CSV filename");

                if (filename.empty()) {
                    filename = "mnist_export.csv";
                }

                std::cout << "🔄 Exporting " << num_samples << " MNIST samples to " << filename
                          << "...\n";

                auto [inputs, targets] = DataCommand::datasetToVectors(*dataset, num_samples);
                DataCommand::saveToCSV(inputs, targets, filename);

                std::cout << "✅ Exported " << inputs.size() << " samples to " << filename << "\n";
            } else {
                std::cout << "❌ Failed to load MNIST dataset.\n";
            }
            break;
        }
        case 5:
            return;
    }

    CliUtils::waitForEnter();
}

void InteractiveMode::manageConfigs() {
    CliUtils::printSectionHeader("Manage Configurations");

    // Show current working directory
    std::cout << "📁 Current working directory: " << std::filesystem::current_path() << "\n\n";

    // Simple implementation to show common config files
    std::vector<std::string> potential_files = {"temp_network.cfg", "network.cfg", "my_network.cfg",
                                                "trained_model.trained"};

    std::vector<std::string> found_files;
    for (const auto& file : potential_files) {
        if (ConfigManager::fileExists(file)) {
            found_files.push_back(file);
        }
    }

    if (found_files.empty()) {
        std::cout << "No configuration files found in current directory.\n";
        std::cout << "Available files to check: temp_network.cfg, network.cfg, my_network.cfg, "
                     "trained_model.trained\n";
    } else {
        std::cout << "Configuration files found:\n";
        for (size_t i = 0; i < found_files.size(); ++i) {
            std::cout << "  " << (i + 1) << ". " << found_files[i] << "\n";
        }
        std::cout << "\nYou can load any of these files using option 2 (Load existing network).\n";
    }

    CliUtils::waitForEnter();
}

std::string InteractiveMode::getUserInput(const std::string& prompt) {
    std::string input;
    std::cout << prompt << ": ";
    std::getline(std::cin, input);
    return CliUtils::trim(input);
}

int InteractiveMode::getUserChoice(const std::string& prompt, int min_choice, int max_choice) {
    while (true) {
        std::cout << prompt << " (" << min_choice << "-" << max_choice << "): ";
        std::string input;
        std::getline(std::cin, input);

        try {
            int choice = std::stoi(CliUtils::trim(input));
            if (choice >= min_choice && choice <= max_choice) {
                return choice;
            }
        } catch (const std::exception&) {
            // Invalid input, continue loop
        }

        std::cout << "❌ Please enter a number between " << min_choice << " and " << max_choice
                  << "\n";
    }
}

double InteractiveMode::getUserDouble(const std::string& prompt, double min_val, double max_val) {
    while (true) {
        std::cout << prompt << " (" << min_val << "-" << max_val << "): ";
        std::string input;
        std::getline(std::cin, input);

        try {
            double value = std::stod(CliUtils::trim(input));
            if (value >= min_val && value <= max_val) {
                return value;
            }
        } catch (const std::exception&) {
            // Invalid input, continue loop
        }

        std::cout << "❌ Please enter a number between " << min_val << " and " << max_val << "\n";
    }
}

bool InteractiveMode::getUserBool(const std::string& prompt) {
    while (true) {
        std::cout << prompt << ": ";
        std::string input;
        std::getline(std::cin, input);
        input = CliUtils::toLowerCase(CliUtils::trim(input));

        if (input == "y" || input == "yes" || input == "true" || input == "1") {
            return true;
        } else if (input == "n" || input == "no" || input == "false" || input == "0") {
            return false;
        }

        std::cout << "❌ Please enter y/n, yes/no, true/false, or 1/0\n";
    }
}

void InteractiveMode::displayTrainingProgress(int epoch,
                                              int total_epochs,
                                              double loss,
                                              double val_loss) {
    std::cout << "Epoch " << std::setw(4) << epoch << "/" << total_epochs;
    std::cout << " | Loss: " << CliUtils::formatNumber(loss, 6);
    if (val_loss > 0) {
        std::cout << " | Val Loss: " << CliUtils::formatNumber(val_loss, 6);
    }
    std::cout << "\n";
}

void InteractiveMode::displayEvaluationResults(const std::vector<Matrix>& predictions,
                                               const std::vector<Matrix>& targets) {
    if (predictions.empty())
        return;

    std::cout << "📈 Sample Predictions vs Targets:\n";

    size_t samples_to_show = std::min(static_cast<size_t>(10), predictions.size());

    std::vector<std::vector<std::string>> table_data;
    std::vector<std::string> headers = {"Sample", "Prediction", "Target", "Error"};

    for (size_t i = 0; i < samples_to_show; ++i) {
        std::vector<std::string> row;
        row.push_back(std::to_string(i + 1));
        row.push_back(CliUtils::formatNumber(predictions[i](0, 0), 4));
        row.push_back(CliUtils::formatNumber(targets[i](0, 0), 4));

        double error = std::abs(predictions[i](0, 0) - targets[i](0, 0));
        row.push_back(CliUtils::formatNumber(error, 4));

        table_data.push_back(row);
    }

    CliUtils::printTable(table_data, headers);

    if (predictions.size() > samples_to_show) {
        std::cout << "... and " << (predictions.size() - samples_to_show) << " more samples\n";
    }
}
