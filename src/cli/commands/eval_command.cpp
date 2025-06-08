#include "eval_command.hpp"

#include <iomanip>
#include <iostream>

#include "../utils/cli_utils.hpp"
#include "../utils/config_manager.hpp"
#include "data_command.hpp"

int EvalCommand::execute(const std::vector<std::string>& args) {
    if (args.size() < 2) {
        showHelp();
        return 1;
    }

    std::string model_file;
    std::string test_file;

    // Parse arguments
    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--model" && i + 1 < args.size()) {
            model_file = args[++i];
        } else if (args[i] == "--test" && i + 1 < args.size()) {
            test_file = args[++i];
        } else if (args[i] == "--help") {
            showHelp();
            return 0;
        }
    }

    if (model_file.empty() || test_file.empty()) {
        std::cout << "❌ Both --model and --test are required\n";
        showHelp();
        return 1;
    }

    try {
        // Load model
        auto network = ConfigManager::loadTrainedModel(model_file);
        std::cout << "✅ Loaded model from " << model_file << "\n";

        // Load test data
        auto [test_inputs, test_targets] = DataCommand::loadFromCSV(test_file);
        std::cout << "✅ Loaded " << test_inputs.size() << " test samples from " << test_file
                  << "\n\n";

        // Evaluate
        network->set_training(false);

        std::vector<Matrix> predictions;
        double total_loss = 0.0;

        std::cout << "🔄 Evaluating...\n";

        for (size_t i = 0; i < test_inputs.size(); ++i) {
            Tensor input_tensor(test_inputs[i]);
            Tensor target_tensor(test_targets[i]);
            Tensor pred = network->forward(input_tensor);
            predictions.push_back(pred.data());
            total_loss += network->compute_loss(pred, target_tensor);
        }

        double avg_loss = total_loss / test_inputs.size();

        std::cout << "\n📊 Evaluation Results:\n";
        std::cout << "  • Average Loss: " << CliUtils::formatNumber(avg_loss, 6) << "\n";
        std::cout << "  • Test Samples: " << test_inputs.size() << "\n\n";

        // Show sample predictions
        std::cout << "📈 Sample Predictions:\n";
        std::vector<std::vector<std::string>> table_data;
        std::vector<std::string> headers = {"Sample", "Prediction", "Target", "Error"};

        size_t samples_to_show = std::min(static_cast<size_t>(10), predictions.size());

        for (size_t i = 0; i < samples_to_show; ++i) {
            std::vector<std::string> row;
            row.push_back(std::to_string(i + 1));
            row.push_back(CliUtils::formatNumber(predictions[i](0, 0), 4));
            row.push_back(CliUtils::formatNumber(test_targets[i](0, 0), 4));

            double error = std::abs(predictions[i](0, 0) - test_targets[i](0, 0));
            row.push_back(CliUtils::formatNumber(error, 4));

            table_data.push_back(row);
        }

        CliUtils::printTable(table_data, headers);

        if (predictions.size() > samples_to_show) {
            std::cout << "... and " << (predictions.size() - samples_to_show) << " more samples\n";
        }

        return 0;

    } catch (const std::exception& e) {
        std::cout << "❌ Evaluation failed: " << e.what() << "\n";
        return 1;
    }
}

void EvalCommand::showHelp() {
    std::cout << "\nbraiNNlet CLI - Evaluate Model Command\n";
    std::cout << "====================================\n\n";
    std::cout << "Usage: braiNNlet-cli evaluate [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --model FILE        Trained model file (.trained or .cfg)\n";
    std::cout << "  --test FILE         Test data file (.csv)\n";
    std::cout << "  --help             Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  braiNNlet-cli evaluate --model trained_model.trained --test test.csv\n";
    std::cout << "  braiNNlet-cli evaluate --model network.cfg --test validation.csv\n\n";
}