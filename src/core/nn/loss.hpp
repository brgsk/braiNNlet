#pragma once

#include <memory>
#include <string>

#include "tensor.hpp"

enum class LossType {
    MeanSquaredError,
    CrossEntropy,
    BinaryCrossEntropy,
};

class LossFunction {
  public:
    virtual ~LossFunction() = default;
    virtual double forward(const Tensor& preds, const Tensor& targets) const = 0;
    virtual Tensor backward(const Tensor& preds, const Tensor& targets) const = 0;
    virtual std::string name() const = 0;
    virtual LossType type() const = 0;
};

class MeanSquaredError : public LossFunction {
  public:
    double forward(const Tensor& preds, const Tensor& targets) const override;
    Tensor backward(const Tensor& preds, const Tensor& targets) const override;
    std::string name() const override {
        return "MSE";
    }
    LossType type() const override {
        return LossType::MeanSquaredError;
    }
};

class CrossEntropy : public LossFunction {
  public:
    double forward(const Tensor& preds, const Tensor& targets) const override;
    Tensor backward(const Tensor& preds, const Tensor& targets) const override;
    std::string name() const override {
        return "CrossEntropy";
    }
    LossType type() const override {
        return LossType::CrossEntropy;
    }

  private:
    Tensor softmax(const Tensor& input) const;
};

class BinaryCrossEntropy : public LossFunction {
  public:
    double forward(const Tensor& preds, const Tensor& targets) const override;
    Tensor backward(const Tensor& preds, const Tensor& targets) const override;
    std::string name() const override {
        return "BinaryCrossEntropy";
    }
    LossType type() const override {
        return LossType::BinaryCrossEntropy;
    }
};

// Factory function to create loss functions
std::unique_ptr<LossFunction> create_loss(LossType type);
LossType loss_from_string(const std::string& name);
std::string loss_to_string(LossType type);
