#include "train_command.hpp"

#include <iomanip>
#include <iostream>

#include "../utils/config_manager.hpp"
#include "data_command.hpp"

int TrainCommand::execute(const std::vector<std::string>& args) {
    if (args.size() < 2) {
        showHelp();
        return 1;
    }

    std::string config_file;
    std::string data_file;
    std::string data_type = "csv";
    std::string output_model;
    int epochs = 10;
    int batch_size = 32;
    double learning_rate = 0.01;  // Better default for MNIST
    double validation_split = 0.2;

    // Parse arguments
    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--config" && i + 1 < args.size()) {
            config_file = args[++i];
        } else if (args[i] == "--data" && i + 1 < args.size()) {
            data_file = args[++i];
        } else if (args[i] == "--data-type" && i + 1 < args.size()) {
            data_type = args[++i];
        } else if (args[i] == "--output" && i + 1 < args.size()) {
            output_model = args[++i];
        } else if (args[i] == "--epochs" && i + 1 < args.size()) {
            epochs = std::stoi(args[++i]);
        } else if (args[i] == "--batch-size" && i + 1 < args.size()) {
            batch_size = std::stoi(args[++i]);
        } else if (args[i] == "--learning-rate" && i + 1 < args.size()) {
            learning_rate = std::stod(args[++i]);
        } else if (args[i] == "--validation-split" && i + 1 < args.size()) {
            validation_split = std::stod(args[++i]);
        } else if (args[i] == "--help") {
            showHelp();
            return 0;
        }
    }

    if (config_file.empty()) {
        std::cout << "❌ --config is required\n";
        showHelp();
        return 1;
    }

    try {
        // Load network
        auto network = ConfigManager::loadNetworkConfig(config_file);
        std::cout << "✅ Loaded network from " << config_file << "\n";

        // Load dataset
        std::unique_ptr<data::Dataset> dataset;

        if (data_type == "mnist") {
            std::cout << "🔄 Loading MNIST dataset...\n";
            dataset = DataCommand::loadMnistDataset(data_file);
            if (!dataset || !dataset->is_loaded()) {
                std::cout << "❌ Failed to load MNIST dataset\n";
                return 1;
            }
        } else if (data_type == "csv") {
            if (data_file.empty()) {
                std::cout << "❌ --data is required for CSV data type\n";
                showHelp();
                return 1;
            }

            // For CSV, we'll need to create a simple dataset wrapper
            // For now, fall back to the old method
            auto [inputs, targets] = DataCommand::loadFromCSV(data_file);
            std::cout << "✅ Loaded " << inputs.size() << " training samples from " << data_file
                      << "\n";

            // Convert to simple training loop for CSV data
            network->set_training(true);
            std::cout << "🚀 Training for " << epochs << " epochs...\n";

            for (int epoch = 0; epoch < epochs; ++epoch) {
                double total_loss = 0.0;

                for (size_t i = 0; i < inputs.size(); ++i) {
                    Tensor input_tensor(inputs[i]);
                    Tensor target_tensor(targets[i]);
                    Tensor prediction = network->forward(input_tensor);
                    double loss = network->compute_loss(prediction, target_tensor);
                    total_loss += loss;

                    Tensor loss_grad = network->compute_loss_gradient(prediction, target_tensor);
                    network->backward(loss_grad);
                    network->update_weights(learning_rate);
                    network->zero_gradients();
                }

                if (epoch % std::max(1, epochs / 10) == 0 || epoch == epochs - 1) {
                    std::cout << "Epoch " << std::setw(3) << epoch + 1 << "/" << epochs
                              << " - Loss: " << std::fixed << std::setprecision(6)
                              << (total_loss / inputs.size()) << "\n";
                }
            }

            std::cout << "✅ Training completed!\n";

            if (!output_model.empty()) {
                ConfigManager::saveTrainedModel(*network, output_model);
            }

            return 0;
        } else {
            std::cout << "❌ Unknown data type: " << data_type << "\n";
            return 1;
        }

        std::cout << "✅ Loaded dataset: " << dataset->size() << " samples\n";
        std::cout << "📊 Input size: " << dataset->input_size()
                  << ", Output size: " << dataset->output_size() << "\n";

        // Set appropriate loss function for MNIST classification
        if (data_type == "mnist") {
            std::cout << "🔧 Setting CrossEntropy loss for MNIST classification...\n";
            network->set_loss_function(LossType::CrossEntropy);
        }

        // Create trainer
        training::Trainer trainer(*network, *dataset);

        // Setup training callbacks for progress display
        setupTrainingCallbacks(trainer);

        // Configure training
        training::TrainingConfig config;
        config.epochs = epochs;
        config.batch_size = batch_size;
        config.learning_rate = learning_rate;
        config.validation_split = validation_split;
        config.shuffle = true;
        config.print_every =
            std::max(1, (dataset->size() / batch_size) / 10);  // Print 10 times per epoch

        std::cout << "🚀 Starting training with Trainer class...\n";
        std::cout << "📋 Config: " << epochs << " epochs, batch size " << batch_size
                  << ", learning rate " << learning_rate << "\n\n";

        // Train the network
        trainer.train(config);

        std::cout << "\n✅ Training completed!\n";

        if (!output_model.empty()) {
            ConfigManager::saveTrainedModel(*network, output_model);
        }

        return 0;

    } catch (const std::exception& e) {
        std::cout << "❌ Training failed: " << e.what() << "\n";
        return 1;
    }
}

void TrainCommand::setupTrainingCallbacks(training::Trainer& trainer) {
    // Setup epoch end callback for progress display
    trainer.set_on_epoch_end([](int epoch, const training::TrainingMetrics& train_metrics,
                                const training::TrainingMetrics& val_metrics) {
        // This is handled by the Trainer's built-in progress display
    });

    // Setup training start callback
    trainer.set_on_training_start([](int total_epochs, int batches_per_epoch) {
        std::cout << "📊 Training will run " << total_epochs << " epochs with " << batches_per_epoch
                  << " batches per epoch\n";
    });
}

void TrainCommand::showHelp() {
    std::cout << "\nbraiNNlet CLI - Train Network Command\n";
    std::cout << "===================================\n\n";
    std::cout << "Usage: braiNNlet-cli train [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --config FILE           Network configuration file (.cfg)\n";
    std::cout << "  --data FILE             Training data file (for CSV data type)\n";
    std::cout << "  --data-type TYPE        Data type: csv, mnist (default: csv)\n";
    std::cout << "  --epochs N              Number of training epochs (default: 10)\n";
    std::cout << "  --batch-size N          Batch size (default: 32)\n";
    std::cout << "  --learning-rate RATE    Learning rate (default: 0.001)\n";
    std::cout << "  --validation-split RATIO Validation split ratio (default: 0.2)\n";
    std::cout << "  --output FILE           Output trained model file (optional)\n";
    std::cout << "  --help                  Show this help message\n\n";
    std::cout << "Data Types:\n";
    std::cout << "  csv                     Load data from CSV file (requires --data)\n";
    std::cout << "  mnist                   Load MNIST dataset (--data optional for path)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  braiNNlet-cli train --config network.cfg --data train.csv\n";
    std::cout << "  braiNNlet-cli train --config mnist_net.cfg --data-type mnist --epochs 20\n";
    std::cout << "  braiNNlet-cli train --config net.cfg --data data.csv --batch-size 64 "
                 "--learning-rate 0.01\n\n";
    std::cout << "Note: For interactive training with more options, use:\n";
    std::cout << "  braiNNlet-cli interactive\n\n";
}