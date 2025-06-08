#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../../core/data/Dataset.hpp"
#include "../../core/data/MnistLoader.hpp"
#include "../../core/nn/tensor.hpp"

class DataCommand {
  public:
    int execute(const std::vector<std::string>& args);

    // Data generation methods
    static std::pair<std::vector<Matrix>, std::vector<Matrix>> generateRegressionData(int samples);
    static std::pair<std::vector<Matrix>, std::vector<Matrix>> generateClassificationData(
        int samples);
    static std::pair<std::vector<Matrix>, std::vector<Matrix>> generateMultiClassData(
        int samples, int num_classes);

    // MNIST data loading
    static std::unique_ptr<data::Dataset> loadMnistDataset(const std::string& path = "");
    static bool checkMnistAvailability();

    // File I/O
    static void saveToCSV(const std::vector<Matrix>& inputs,
                          const std::vector<Matrix>& targets,
                          const std::string& filename);
    static std::pair<std::vector<Matrix>, std::vector<Matrix>> loadFromCSV(
        const std::string& filename);

    // Dataset conversion utilities
    static std::pair<std::vector<Matrix>, std::vector<Matrix>> datasetToVectors(
        const data::Dataset& dataset, int max_samples = -1);

    // Data utilities
    static void shuffleData(std::vector<Matrix>& inputs, std::vector<Matrix>& targets);
    static std::pair<std::vector<Matrix>, std::vector<Matrix>> splitData(
        const std::vector<Matrix>& inputs, const std::vector<Matrix>& targets, double train_ratio);

  private:
    void showHelp();
    void generateDataInteractive();
};