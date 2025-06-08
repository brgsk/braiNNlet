#pragma once

#include <memory>

#include "../../core/data/Dataset.hpp"
#include "../../core/data/MnistLoader.hpp"
#include "../../core/nn/network.hpp"
#include "../../core/training/Trainer.hpp"
#include "../utils/config_manager.hpp"

class InteractiveMode {
  public:
    int run();

  private:
    void showMainMenu();
    void createNetwork();
    void loadNetwork();
    void showNetworkInfo();
    void trainNetwork();
    void evaluateNetwork();
    void generateData();
    void saveNetwork();
    void mnistOperations();
    void manageConfigs();

    // Network building helpers
    void buildNetworkInteractive();
    TrainingConfig configureTraining();
    void runTrainingLoop(const std::vector<Matrix>& train_inputs,
                         const std::vector<Matrix>& train_targets,
                         const std::vector<Matrix>& val_inputs,
                         const std::vector<Matrix>& val_targets,
                         const TrainingConfig& config);

    // Utility methods
    std::string getUserInput(const std::string& prompt);
    int getUserChoice(const std::string& prompt, int min_choice, int max_choice);
    double getUserDouble(const std::string& prompt, double min_val = -1e6, double max_val = 1e6);
    bool getUserBool(const std::string& prompt);

    void displayTrainingProgress(int epoch, int total_epochs, double loss, double val_loss);
    void displayEvaluationResults(const std::vector<Matrix>& predictions,
                                  const std::vector<Matrix>& targets);

    std::unique_ptr<Network> network_;
    bool running_;
};