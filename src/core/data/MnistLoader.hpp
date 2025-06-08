#pragma once

#include <string>
#include <vector>

#include "Dataset.hpp"

namespace data {

class MnistDataset : public Dataset {
  public:
    MnistDataset();

    // Dataset interface
    bool load(const std::string& path) override;
    int size() const override {
        return static_cast<int>(samples_.size());
    }
    DataSample get(int index) const override;
    std::string name() const override {
        return "MNIST";
    }

    int input_size() const override {
        return 784;
    }  // 28x28 pixels
    int output_size() const override {
        return 10;
    }  // 10 classes

    // Additional functionality for train/test splitting
    bool load_train_test_split(const std::string& mnist_dir, double train_ratio = 0.8);
    void create_validation_split(double validation_ratio = 0.2);

  private:
    // MNIST binary file loading
    std::vector<std::vector<uint8_t>> load_images(const std::string& filename);
    std::vector<uint8_t> load_labels(const std::string& filename);

    // Helper functions
    uint32_t read_big_endian_uint32(std::ifstream& file);
    Tensor create_one_hot(int label, int num_classes) const;
    Tensor normalize_image(const std::vector<uint8_t>& image_data) const;

    // For backwards compatibility (fallback to dummy data if files not found)
    void create_dummy_data();
};

}  // namespace data