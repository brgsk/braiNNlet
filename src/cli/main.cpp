#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "../core/nn/network.hpp"
#include "commands/create_command.hpp"
#include "commands/data_command.hpp"
#include "commands/eval_command.hpp"
#include "commands/train_command.hpp"
#include "interfaces/interactive_mode.hpp"
#include "utils/cli_utils.hpp"

void printUsage() {
    std::cout << "\n🧠 braiNNlet CLI - Neural Network Training Tool\n";
    std::cout << "===============================================\n\n";
    std::cout << "Usage: braiNNlet-cli [COMMAND] [OPTIONS]\n\n";
    std::cout << "Commands:\n";
    std::cout << "  interactive    Start interactive mode (default)\n";
    std::cout << "  create         Create a new network configuration\n";
    std::cout << "  train          Train a neural network\n";
    std::cout << "  evaluate       Evaluate a trained model\n";
    std::cout << "  generate-data  Generate synthetic training data\n";
    std::cout << "  help           Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  braiNNlet-cli                                    # Interactive mode\n";
    std::cout << "  braiNNlet-cli create --output my_network.json   # Create network config\n";
    std::cout << "  braiNNlet-cli train --config net.json --data training_data.csv\n";
    std::cout << "  braiNNlet-cli evaluate --model trained.json --test test_data.csv\n\n";
    std::cout << "For detailed help on specific commands, use:\n";
    std::cout << "  braiNNlet-cli [COMMAND] --help\n\n";
}

int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        std::vector<std::string> args(argv + 1, argv + argc);

        if (args.empty() || args[0] == "interactive") {
            // Start interactive mode
            InteractiveMode interactive;
            return interactive.run();
        }

        std::string command = args[0];

        if (command == "help" || command == "--help" || command == "-h") {
            printUsage();
            return 0;
        } else if (command == "create") {
            CreateCommand cmd;
            return cmd.execute(args);
        } else if (command == "train") {
            TrainCommand cmd;
            return cmd.execute(args);
        } else if (command == "evaluate") {
            EvalCommand cmd;
            return cmd.execute(args);
        } else if (command == "generate-data") {
            DataCommand cmd;
            return cmd.execute(args);
        } else {
            std::cerr << "❌ Unknown command: " << command << "\n";
            printUsage();
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}