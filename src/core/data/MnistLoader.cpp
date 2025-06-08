#include "MnistLoader.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

namespace data {

MnistDataset::MnistDataset() {
    loaded_ = false;
}

bool MnistDataset::load(const std::string& path) {
    // Try to load real MNIST data first
    if (!path.empty() && std::filesystem::exists(path)) {
        return load_train_test_split(path, 1.0);  // Load all training data
    }

    // Try default MNIST directory
    std::string mnist_dir = "src/core/data/MNIST";
    if (std::filesystem::exists(mnist_dir)) {
        std::cout << "Loading MNIST dataset from: " << mnist_dir << std::endl;
        return load_train_test_split(mnist_dir, 1.0);
    }

    // Fallback to dummy data if MNIST files not found
    std::cout << "MNIST files not found, using dummy data for testing" << std::endl;
    create_dummy_data();
    loaded_ = true;
    return true;
}

bool MnistDataset::load_train_test_split(const std::string& mnist_dir, double train_ratio) {
    try {
        std::string train_images_path = mnist_dir + "/train-images.idx3-ubyte";
        std::string train_labels_path = mnist_dir + "/train-labels.idx1-ubyte";

        // Load training data
        auto images = load_images(train_images_path);
        auto labels = load_labels(train_labels_path);

        if (images.size() != labels.size()) {
            throw std::runtime_error("Mismatch between number of images and labels");
        }

        std::cout << "Loaded " << images.size() << " MNIST training samples" << std::endl;

        // Convert to our internal format
        samples_.clear();
        samples_.reserve(images.size());

        for (size_t i = 0; i < images.size(); ++i) {
            Tensor features = normalize_image(images[i]);
            Tensor label_tensor = create_one_hot(labels[i], 10);
            samples_.emplace_back(features, label_tensor);
        }

        // Shuffle the data for better training
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(samples_.begin(), samples_.end(), gen);

        // If train_ratio < 1.0, keep only a portion of the data
        if (train_ratio < 1.0) {
            size_t keep_size = static_cast<size_t>(samples_.size() * train_ratio);
            samples_.erase(samples_.begin() + keep_size, samples_.end());
            std::cout << "Using " << keep_size << " samples (ratio: " << train_ratio << ")"
                      << std::endl;
        }

        loaded_ = true;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error loading MNIST data: " << e.what() << std::endl;
        // Fallback to dummy data
        create_dummy_data();
        loaded_ = true;
        return false;
    }
}

void MnistDataset::create_validation_split(double validation_ratio) {
    if (!loaded_ || samples_.empty()) {
        throw std::runtime_error("Dataset must be loaded before creating validation split");
    }

    // This could be implemented to split the current samples into train/validation
    // For now, we'll just shuffle the existing data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(samples_.begin(), samples_.end(), gen);

    std::cout << "Dataset shuffled for validation split (ratio: " << validation_ratio << ")"
              << std::endl;
}

std::vector<std::vector<uint8_t>> MnistDataset::load_images(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open image file: " + filename);
    }

    // Read header
    uint32_t magic = read_big_endian_uint32(file);
    if (magic != 0x00000803) {
        throw std::runtime_error("Invalid MNIST image file magic number");
    }

    uint32_t num_images = read_big_endian_uint32(file);
    uint32_t num_rows = read_big_endian_uint32(file);
    uint32_t num_cols = read_big_endian_uint32(file);

    std::cout << "MNIST Images: " << num_images << " samples, " << num_rows << "x" << num_cols
              << " pixels" << std::endl;

    if (num_rows != 28 || num_cols != 28) {
        throw std::runtime_error("Expected 28x28 images");
    }

    // Read image data
    std::vector<std::vector<uint8_t>> images;
    images.reserve(num_images);

    for (uint32_t i = 0; i < num_images; ++i) {
        std::vector<uint8_t> image(784);  // 28*28 = 784
        file.read(reinterpret_cast<char*>(image.data()), 784);
        if (!file) {
            throw std::runtime_error("Error reading image data");
        }
        images.push_back(std::move(image));
    }

    return images;
}

std::vector<uint8_t> MnistDataset::load_labels(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open label file: " + filename);
    }

    // Read header
    uint32_t magic = read_big_endian_uint32(file);
    if (magic != 0x00000801) {
        throw std::runtime_error("Invalid MNIST label file magic number");
    }

    uint32_t num_labels = read_big_endian_uint32(file);
    std::cout << "MNIST Labels: " << num_labels << " labels" << std::endl;

    // Read label data
    std::vector<uint8_t> labels(num_labels);
    file.read(reinterpret_cast<char*>(labels.data()), num_labels);
    if (!file) {
        throw std::runtime_error("Error reading label data");
    }

    return labels;
}

uint32_t MnistDataset::read_big_endian_uint32(std::ifstream& file) {
    uint32_t value = 0;
    file.read(reinterpret_cast<char*>(&value), 4);
    if (!file) {
        throw std::runtime_error("Error reading big-endian uint32");
    }

    // Convert from big-endian to host byte order
    return ((value & 0xFF000000) >> 24) | ((value & 0x00FF0000) >> 8) |
           ((value & 0x0000FF00) << 8) | ((value & 0x000000FF) << 24);
}

Tensor MnistDataset::normalize_image(const std::vector<uint8_t>& image_data) const {
    Tensor features(1, 784);

    // Normalize pixel values from [0, 255] to [0, 1]
    for (int i = 0; i < 784; ++i) {
        features(0, i) = static_cast<double>(image_data[i]) / 255.0;
    }

    return features;
}

DataSample MnistDataset::get(int index) const {
    if (!loaded_) {
        throw std::runtime_error("Dataset not loaded");
    }

    if (index < 0 || index >= size()) {
        throw std::out_of_range("Index out of range");
    }

    return samples_[index];
}

void MnistDataset::create_dummy_data() {
    samples_.clear();

    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> pixel_dist(0.0, 1.0);
    std::uniform_int_distribution<int> label_dist(0, 9);

    // Create 1000 dummy samples
    int num_samples = 1000;

    for (int i = 0; i < num_samples; ++i) {
        // Create random 28x28 image (flattened to 784 pixels)
        Tensor features(1, 784);
        for (int j = 0; j < 784; ++j) {
            features(0, j) = pixel_dist(gen);
        }

        // Create random label (0-9)
        int label = label_dist(gen);
        Tensor one_hot_label = create_one_hot(label, 10);

        samples_.emplace_back(features, one_hot_label);
    }
}

Tensor MnistDataset::create_one_hot(int label, int num_classes) const {
    Tensor one_hot(1, num_classes);
    one_hot.zero();

    if (label >= 0 && label < num_classes) {
        one_hot(0, label) = 1.0;
    }

    return one_hot;
}

}  // namespace data