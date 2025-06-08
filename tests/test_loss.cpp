//
// Created by Bartosz Roguski on 07/06/2025.
//
#include <cassert>
#include <cmath>
#include <iomanip>

#include "src/core/nn/loss.hpp"

void testMeanSquaredError() {
    printf("\n=== Testing Mean Squared Error ===\n");

    auto mse = create_loss(LossType::MeanSquaredError);

    // Simple test case: predictions [1, 2], targets [0.5, 2.5]
    Matrix preds_data(1, 2);
    preds_data << 1.0, 2.0;
    Matrix targets_data(1, 2);
    targets_data << 0.5, 2.5;

    Tensor preds(preds_data);
    Tensor targets(targets_data);

    // Test forward pass
    double loss = mse->forward(preds, targets);
    // Expected: ((1-0.5)^2 + (2-2.5)^2) / 2 = (0.25 + 0.25) / 2 = 0.25
    double expected_loss = 0.25;
    assert(std::abs(loss - expected_loss) < 1e-10);
    printf("✓ MSE forward pass test passed (loss = %.6f)\n", loss);

    // Test backward pass
    Tensor grad = mse->backward(preds, targets);
    // Expected gradient: 2 * (preds - targets) / count
    // = 2 * ([1, 2] - [0.5, 2.5]) / 2 = 2 * [0.5, -0.5] / 2 = [0.5, -0.5]
    Matrix expected_grad(1, 2);
    expected_grad << 0.5, -0.5;

    assert((grad.data() - expected_grad).norm() < 1e-10);
    printf("✓ MSE backward pass test passed\n");

    // Test properties
    assert(mse->name() == "MSE");
    assert(mse->type() == LossType::MeanSquaredError);
    printf("✓ MSE properties test passed\n");
}

void testBinaryCrossEntropy() {
    printf("\n=== Testing Binary Cross Entropy ===\n");

    auto bce = create_loss(LossType::BinaryCrossEntropy);

    // Test case: predictions [0.8, 0.3], targets [1, 0]
    Matrix preds_data(1, 2);
    preds_data << 0.8, 0.3;
    Matrix targets_data(1, 2);
    targets_data << 1.0, 0.0;

    Tensor preds(preds_data);
    Tensor targets(targets_data);

    // Test forward pass
    double loss = bce->forward(preds, targets);
    // Expected: -(1*log(0.8) + 0*log(0.2) + 0*log(0.3) + 1*log(0.7)) / 2
    double expected = -(std::log(0.8) + std::log(0.7)) / 2.0;
    assert(std::abs(loss - expected) < 1e-6);
    printf("✓ BCE forward pass test passed (loss = %.6f)\n", loss);

    // Test backward pass
    Tensor grad = bce->backward(preds, targets);
    // Expected gradient for BCE: (pred - target) / (pred * (1 - pred)) / count
    double grad1 = (0.8 - 1.0) / (0.8 * 0.2) / 2.0;  // -1.25
    double grad2 = (0.3 - 0.0) / (0.3 * 0.7) / 2.0;  // 0.714...

    assert(std::abs(grad(0, 0) - grad1) < 1e-6);
    assert(std::abs(grad(0, 1) - grad2) < 1e-6);
    printf("✓ BCE backward pass test passed\n");

    // Test properties
    assert(bce->name() == "BinaryCrossEntropy");
    assert(bce->type() == LossType::BinaryCrossEntropy);
    printf("✓ BCE properties test passed\n");
}

void testCrossEntropy() {
    printf("\n=== Testing Cross Entropy ===\n");

    auto ce = create_loss(LossType::CrossEntropy);

    // Test case: logits [2, 1, 0.1], target one-hot [0, 1, 0]
    Matrix preds_data(1, 3);
    preds_data << 2.0, 1.0, 0.1;
    Matrix targets_data(1, 3);
    targets_data << 0.0, 1.0, 0.0;

    Tensor preds(preds_data);
    Tensor targets(targets_data);

    // Test forward pass
    double loss = ce->forward(preds, targets);

    // Manual calculation:
    // Softmax: exp([2, 1, 0.1]) / sum = [7.389, 2.718, 1.105] / 11.212 = [0.659, 0.242, 0.099]
    // Loss: -log(0.242) ≈ 1.417
    double exp_sum = std::exp(2.0) + std::exp(1.0) + std::exp(0.1);
    double softmax_target = std::exp(1.0) / exp_sum;
    double expected_loss = -std::log(softmax_target);

    assert(std::abs(loss - expected_loss) < 1e-6);
    printf("✓ CE forward pass test passed (loss = %.4f)\n", loss);

    // Test backward pass
    Tensor grad = ce->backward(preds, targets);

    // Expected: (softmax - targets) / batch_size
    double exp2 = std::exp(2.0);
    double exp1 = std::exp(1.0);
    double exp01 = std::exp(0.1);
    double sum_exp = exp2 + exp1 + exp01;

    Matrix expected_grad(1, 3);
    expected_grad << (exp2 / sum_exp - 0.0), (exp1 / sum_exp - 1.0), (exp01 / sum_exp - 0.0);

    assert((grad.data() - expected_grad).norm() < 1e-6);
    printf("✓ CE backward pass test passed\n");

    // Test properties
    assert(ce->name() == "CrossEntropy");
    assert(ce->type() == LossType::CrossEntropy);
    printf("✓ CE properties test passed\n");
}

void testLossFactoryFunctions() {
    printf("\n=== Testing Loss Factory Functions ===\n");

    // Test string to loss type conversion
    assert(loss_from_string("MSE") == LossType::MeanSquaredError);
    assert(loss_from_string("mse") == LossType::MeanSquaredError);
    assert(loss_from_string("CrossEntropy") == LossType::CrossEntropy);
    assert(loss_from_string("crossentropy") == LossType::CrossEntropy);
    assert(loss_from_string("BinaryCrossEntropy") == LossType::BinaryCrossEntropy);
    assert(loss_from_string("bce") == LossType::BinaryCrossEntropy);
    printf("✓ String to loss type conversion test passed\n");

    // Test loss type to string conversion
    assert(loss_to_string(LossType::MeanSquaredError) == "MSE");
    assert(loss_to_string(LossType::CrossEntropy) == "CrossEntropy");
    assert(loss_to_string(LossType::BinaryCrossEntropy) == "BinaryCrossEntropy");
    printf("✓ Loss type to string conversion test passed\n");

    // Test unknown string handling
    try {
        loss_from_string("Unknown");
        assert(false);  // Should throw
    } catch (const std::invalid_argument&) {
        printf("✓ Unknown loss string handling test passed\n");
    }

    // Test factory function creates correct types
    auto mse = create_loss(LossType::MeanSquaredError);
    auto ce = create_loss(LossType::CrossEntropy);
    auto bce = create_loss(LossType::BinaryCrossEntropy);

    assert(mse->type() == LossType::MeanSquaredError);
    assert(ce->type() == LossType::CrossEntropy);
    assert(bce->type() == LossType::BinaryCrossEntropy);
    printf("✓ Factory function test passed\n");
}

void testLossNumericalStability() {
    printf("\n=== Testing Loss Numerical Stability ===\n");

    auto bce = create_loss(LossType::BinaryCrossEntropy);
    auto ce = create_loss(LossType::CrossEntropy);

    // Test BCE with extreme values
    Matrix extreme_preds(1, 2);
    extreme_preds << 1e-16, 1.0 - 1e-16;
    Matrix extreme_targets(1, 2);
    extreme_targets << 0.0, 1.0;

    Tensor preds(extreme_preds);
    Tensor targets(extreme_targets);

    double bce_loss = bce->forward(preds, targets);
    assert(std::isfinite(bce_loss));  // Should not be infinite
    printf("✓ BCE numerical stability test passed\n");

    // Test CE with large logits
    Matrix large_logits(1, 3);
    large_logits << 100, 50, 200;
    Matrix ce_targets(1, 3);
    ce_targets << 0, 0, 1;

    Tensor large_preds(large_logits);
    Tensor ce_target_tensor(ce_targets);

    double ce_loss = ce->forward(large_preds, ce_target_tensor);
    assert(std::isfinite(ce_loss));  // Should not be infinite
    printf("✓ CE numerical stability test passed\n");
}

void testLossBatchHandling() {
    printf("\n=== Testing Loss Batch Handling ===\n");

    auto mse = create_loss(LossType::MeanSquaredError);

    // Test with batch size > 1
    Matrix batch_preds(2, 2);
    batch_preds << 1.0, 2.0, 3.0, 4.0;
    Matrix batch_targets(2, 2);
    batch_targets << 0.5, 1.5, 2.5, 3.5;

    Tensor preds(batch_preds);
    Tensor targets(batch_targets);

    double loss = mse->forward(preds, targets);
    // Expected: ((1-0.5)^2 + (2-1.5)^2 + (3-2.5)^2 + (4-3.5)^2) / 4
    //         = (0.25 + 0.25 + 0.25 + 0.25) / 4 = 0.25
    assert(std::abs(loss - 0.25) < 1e-10);
    printf("✓ Batch loss calculation test passed\n");

    Tensor grad = mse->backward(preds, targets);
    // Expected: 2 * (preds - targets) / count
    Matrix expected_grad(2, 2);
    expected_grad << 0.25, 0.25, 0.25, 0.25;

    assert((grad.data() - expected_grad).norm() < 1e-10);
    printf("✓ Batch gradient calculation test passed\n");
}

void testLossWithNeuralNetwork() {
    printf("\n=== Testing Loss Integration with Neural Network ===\n");

    // Create a simple 1-layer network manually
    Matrix weights(2, 1);
    weights << 0.5, -0.3;
    Matrix bias(1, 1);
    bias << 0.1;

    Matrix input(1, 2);
    input << 2.0, 3.0;
    Matrix target(1, 1);
    target << 1.0;

    // Forward pass: output = input * weights + bias
    Tensor input_tensor(input);
    Tensor weights_tensor(weights);
    Tensor bias_tensor(bias);
    Tensor target_tensor(target);

    Tensor output = input_tensor * weights_tensor;
    output = output.broadcast_add(bias_tensor);
    // Expected: [2, 3] * [0.5; -0.3] + 0.1 = 1.0 - 0.9 + 0.1 = 0.2

    auto mse = create_loss(LossType::MeanSquaredError);
    double loss = mse->forward(output, target_tensor);

    // Expected loss: (0.2 - 1.0)^2 = 0.64
    assert(std::abs(loss - 0.64) < 1e-10);
    printf("✓ Network integration forward pass test passed (loss = %.6f)\n", loss);

    // Test backward pass through loss
    Tensor loss_grad = mse->backward(output, target_tensor);
    // Expected: 2 * (0.2 - 1.0) / 1 = -1.6
    assert(std::abs(loss_grad(0, 0) - (-1.6)) < 1e-10);
    printf("✓ Network integration backward pass test passed\n");
}

int runLossTests() {
    try {
        printf("=== LOSS FUNCTION TESTS ===\n");
        testMeanSquaredError();
        testBinaryCrossEntropy();
        testCrossEntropy();
        testLossFactoryFunctions();
        testLossNumericalStability();
        testLossBatchHandling();
        testLossWithNeuralNetwork();
        printf("✅ All loss function tests passed!\n\n");
        return 0;
    } catch (const std::exception& e) {
        printf("❌ Loss test failed: %s\n", e.what());
        return 1;
    }
}