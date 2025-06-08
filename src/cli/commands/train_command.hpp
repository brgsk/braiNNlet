#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../../core/data/Dataset.hpp"
#include "../../core/training/Trainer.hpp"

class TrainCommand {
  public:
    int execute(const std::vector<std::string>& args);

  private:
    void showHelp();
    void setupTrainingCallbacks(training::Trainer& trainer);
};