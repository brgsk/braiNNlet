#pragma once

#include "../nn/Tensor.hpp"
#include <vector>
#include <string>

namespace data {

struct DataSample {
    nn::Tensor features;
    nn::Tensor label;
    
    // Default constructor
    DataSample() = default;
    
    // Constructor with parameters
    DataSample(const nn::Tensor& f, const nn::Tensor& l) : features(f), label(l) {}
};

class Dataset {
public:
    virtual ~Dataset() = default;
    
    // Pure virtual methods
    virtual bool load(const std::string& path) = 0;
    virtual int size() const = 0;
    virtual DataSample get(int index) const = 0;
    virtual std::string name() const = 0;
    
    // Optional methods with default implementations
    virtual int input_size() const = 0;
    virtual int output_size() const = 0;
    virtual bool is_loaded() const { return loaded_; }
    
    // Batch operations
    virtual std::pair<nn::Tensor, nn::Tensor> get_batch(const std::vector<int>& indices) const;
    virtual std::pair<nn::Tensor, nn::Tensor> get_batch(int start, int size) const;
    
    // Statistics
    virtual nn::Tensor compute_mean() const;
    virtual nn::Tensor compute_std() const;
    virtual void normalize();

protected:
    bool loaded_ = false;
    std::vector<DataSample> samples_;
    
    // Normalization parameters
    nn::Tensor mean_;
    nn::Tensor std_;
    bool normalized_ = false;
};

// Utility functions
std::vector<int> generate_indices(int size);
std::vector<int> shuffle_indices(const std::vector<int>& indices);
void split_dataset(const std::vector<int>& indices, double train_ratio, 
                  std::vector<int>& train_indices, std::vector<int>& val_indices);

} // namespace data 