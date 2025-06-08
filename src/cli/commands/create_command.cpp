#include "create_command.hpp"

#include <iostream>

#include "../../core/nn/network.hpp"
#include "../utils/config_manager.hpp"

int CreateCommand::execute(const std::vector<std::string>& args) {
    if (args.size() < 2) {
        showHelp();
        return 1;
    }

    return createFromCommandLine(args);
}

int CreateCommand::createFromCommandLine(const std::vector<std::string>& args) {
    // This is a simplified version - in practice you'd want more options
    std::string output_file = "network.cfg";

    // Parse command line arguments
    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--output" && i + 1 < args.size()) {
            output_file = args[++i];
        } else if (args[i] == "--help") {
            showHelp();
            return 0;
        }
    }

    try {
        // Create a simple default network
        Network network;
        network.add_layer(10, ActivationType::ReLU);   // Hidden layer
        network.add_layer(1, ActivationType::Linear);  // Output layer
        network.set_loss_function(LossType::MeanSquaredError);

        ConfigManager::saveNetworkConfig(network, output_file);
        std::cout << "✅ Default network configuration saved to " << output_file << "\n";
        std::cout << "You can modify this configuration file or use interactive mode for custom "
                     "networks.\n";

        return 0;
    } catch (const std::exception& e) {
        std::cout << "❌ Error creating network: " << e.what() << "\n";
        return 1;
    }
}

void CreateCommand::showHelp() {
    std::cout << "\nbraiNNlet CLI - Create Network Command\n";
    std::cout << "====================================\n\n";
    std::cout << "Usage: braiNNlet-cli create [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --output FILE       Output configuration filename (default: network.cfg)\n";
    std::cout << "  --help             Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  braiNNlet-cli create --output my_network.cfg\n";
    std::cout << "  braiNNlet-cli create  # Creates network.cfg\n\n";
    std::cout << "Note: This creates a default network configuration.\n";
    std::cout << "Use interactive mode for custom architectures:\n";
    std::cout << "  braiNNlet-cli interactive\n\n";
}