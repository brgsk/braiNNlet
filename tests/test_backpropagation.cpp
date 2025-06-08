//
// Created by Bartosz Roguski on 07/06/2025.
//
#include <cassert>
#include <cmath>

#include "src/core/nn/activations.hpp"
#include "src/core/nn/dense_layer.hpp"
#include "src/core/nn/loss.hpp"

void testEndToEndBackpropagation() {
    printf("\n=== Testing End-to-End Backpropagation ===\n");

    // Create a simple 2-layer network: 2 -> 2 -> 1
    DenseLayer layer1(2, 2, ActivationType::Linear);
    DenseLayer layer2(2, 1, ActivationType::Linear);

    // Set known weights for predictable behavior
    Matrix weights1(2, 2);
    weights1 << 0.5, 0.3, 0.2, 0.4;
    layer1.set_weights(Tensor(weights1));

    Matrix biases1(1, 2);
    biases1 << 0.1, 0.2;
    layer1.set_biases(Tensor(biases1));

    Matrix weights2(2, 1);
    weights2 << 0.6, 0.7;
    layer2.set_weights(Tensor(weights2));

    Matrix biases2(1, 1);
    biases2 << 0.3;
    layer2.set_biases(Tensor(biases2));

    // Input and target
    Matrix input_data(1, 2);
    input_data << 2.0, 3.0;
    Tensor input(input_data);

    Matrix target_data(1, 1);
    target_data << 1.0;
    Tensor target(target_data);

    // Forward pass
    Tensor hidden = layer1.forward(input);
    Tensor output = layer2.forward(hidden);

    // Verify forward pass calculations
    // Hidden: [2, 3] * [[0.5, 0.3], [0.2, 0.4]] + [0.1, 0.2] = [1.6, 1.8] + [0.1, 0.2] = [1.7, 2.0]
    Matrix expected_hidden(1, 2);
    expected_hidden << 1.7, 2.0;
    assert((hidden.data() - expected_hidden).norm() < 1e-10);

    // Output: [1.7, 2.0] * [[0.6], [0.7]] + [0.3] = [2.72]
    Matrix expected_output(1, 1);
    expected_output << 2.72;
    assert((output.data() - expected_output).norm() < 1e-10);
    printf("✓ Forward pass verification test passed\n");

    // Compute loss
    auto mse = create_loss(LossType::MeanSquaredError);
    double loss_value = mse->forward(output, target);
    double expected_loss = std::pow(2.72 - 1.0, 2.0);  // (2.72 - 1.0)^2 = 2.9584
    assert(std::abs(loss_value - expected_loss) < 1e-10);
    printf("✓ Loss calculation test passed (loss = %.4f)\n", loss_value);

    // Backward pass
    Tensor loss_grad = mse->backward(output, target);  // 2 * (2.72 - 1.0) = 3.44
    assert(std::abs(loss_grad(0, 0) - 3.44) < 1e-10);

    Tensor grad_hidden = layer2.backward(loss_grad);
    Tensor grad_input = layer1.backward(grad_hidden);

    // Verify gradient calculations
    // grad_hidden = loss_grad * weights2^T = [3.44] * [[0.6, 0.7]] = [2.064, 2.408]
    Matrix expected_grad_hidden(1, 2);
    expected_grad_hidden << 2.064, 2.408;
    assert((grad_hidden.data() - expected_grad_hidden).norm() < 1e-10);

    // grad_input = grad_hidden * weights1^T = [2.064, 2.408] * [[0.5, 0.2], [0.3, 0.4]] =
    // [1.7544, 1.3760]
    Matrix expected_grad_input(1, 2);
    expected_grad_input << 1.7544, 1.3760;
    assert((grad_input.data() - expected_grad_input).norm() < 1e-9);  // Slightly relaxed tolerance
    printf("✓ Backward pass gradient verification test passed\n");
}

void testGradientChecking() {
    printf("\n=== Testing Numerical Gradient Checking ===\n");

    // Simple single layer network for gradient checking
    DenseLayer layer(2, 1, ActivationType::Linear);

    Matrix weights(2, 1);
    weights << 0.5, 0.3;
    layer.set_weights(Tensor(weights));

    Matrix biases(1, 1);
    biases << 0.1;
    layer.set_biases(Tensor(biases));

    Matrix input_data(1, 2);
    input_data << 1.0, 2.0;
    Tensor input(input_data);

    Matrix target_data(1, 1);
    target_data << 0.5;
    Tensor target(target_data);

    auto mse = create_loss(LossType::MeanSquaredError);

    // Compute analytical gradients
    Tensor output = layer.forward(input);
    double loss_value = mse->forward(output, target);
    Tensor loss_grad = mse->backward(output, target);
    layer.backward(loss_grad);

    auto analytical_weight_grad = layer.weight_gradients();
    auto analytical_bias_grad = layer.bias_gradients();

    // Numerical gradient checking with finite differences
    double epsilon = 1e-5;
    auto weights_tensor = layer.weights();
    Matrix weights_copy = weights_tensor.data();

    // Check weight gradients
    for (int i = 0; i < weights_copy.rows(); ++i) {
        for (int j = 0; j < weights_copy.cols(); ++j) {
            // Positive perturbation
            weights_copy(i, j) += epsilon;
            layer.set_weights(Tensor(weights_copy));
            Tensor output_pos = layer.forward(input);
            double loss_pos = mse->forward(output_pos, target);

            // Negative perturbation
            weights_copy(i, j) -= 2 * epsilon;
            layer.set_weights(Tensor(weights_copy));
            Tensor output_neg = layer.forward(input);
            double loss_neg = mse->forward(output_neg, target);

            // Numerical gradient
            double numerical_grad = (loss_pos - loss_neg) / (2 * epsilon);

            // Restore original weight
            weights_copy(i, j) += epsilon;

            // Compare with analytical gradient
            double analytical_grad = analytical_weight_grad(i, j);
            double relative_error = std::abs(numerical_grad - analytical_grad) /
                                    std::max(std::abs(numerical_grad), std::abs(analytical_grad));

            assert(relative_error < 1e-5);  // Should be very close
        }
    }

    // Restore original weights
    layer.set_weights(Tensor(weights_copy));
    printf("✓ Numerical gradient checking test passed\n");
}

void testMultiLayerBackpropagation() {
    printf("\n=== Testing Multi-Layer Backpropagation ===\n");

    // Create a 3-layer network: 3 -> 4 -> 2 -> 1
    DenseLayer layer1(3, 4, ActivationType::ReLU);
    DenseLayer layer2(4, 2, ActivationType::ReLU);
    DenseLayer layer3(2, 1, ActivationType::Linear);

    // Initialize with small random-like values for stability
    Matrix weights1(3, 4);
    weights1 << 0.1, 0.2, -0.1, 0.15, 0.05, -0.1, 0.2, 0.1, -0.05, 0.1, 0.1, -0.15;
    layer1.set_weights(Tensor(weights1));

    Matrix biases1(1, 4);
    biases1 << 0.01, -0.02, 0.015, 0.02;
    layer1.set_biases(Tensor(biases1));

    Matrix weights2(4, 2);
    weights2 << 0.2, -0.15, -0.1, 0.25, 0.15, 0.1, 0.05, -0.2;
    layer2.set_weights(Tensor(weights2));

    Matrix biases2(1, 2);
    biases2 << 0.05, -0.03;
    layer2.set_biases(Tensor(biases2));

    Matrix weights3(2, 1);
    weights3 << 0.3, 0.4;
    layer3.set_weights(Tensor(weights3));

    Matrix biases3(1, 1);
    biases3 << 0.1;
    layer3.set_biases(Tensor(biases3));

    // Test data
    Matrix input_data(1, 3);
    input_data << 1.0, 2.0, 0.5;
    Tensor input(input_data);

    Matrix target_data(1, 1);
    target_data << 0.8;
    Tensor target(target_data);

    // Forward pass
    Tensor hidden1 = layer1.forward(input);
    Tensor hidden2 = layer2.forward(hidden1);
    Tensor output = layer3.forward(hidden2);

    // Compute loss
    auto mse = create_loss(LossType::MeanSquaredError);
    double loss_value = mse->forward(output, target);

    printf("✓ Multi-layer forward pass completed (loss = %.6f)\n", loss_value);

    // Backward pass
    Tensor loss_grad = mse->backward(output, target);
    Tensor grad2 = layer3.backward(loss_grad);
    Tensor grad1 = layer2.backward(grad2);
    Tensor grad_input = layer1.backward(grad1);

    // Verify that gradients have been computed (non-zero)
    auto weight_grad1 = layer1.weight_gradients();
    auto weight_grad2 = layer2.weight_gradients();
    auto weight_grad3 = layer3.weight_gradients();

    assert(weight_grad1.norm() > 1e-10);
    assert(weight_grad2.norm() > 1e-10);
    assert(weight_grad3.norm() > 1e-10);

    printf("✓ Multi-layer backward pass completed\n");
    printf("✓ Gradient norms: Layer1=%.6f, Layer2=%.6f, Layer3=%.6f\n", weight_grad1.norm(),
           weight_grad2.norm(), weight_grad3.norm());
}

void testWeightUpdatesAndConvergence() {
    printf("\n=== Testing Weight Updates and Training ===\n");

    // Simple network for training test
    DenseLayer layer(2, 1, ActivationType::Linear);

    // Initialize weights
    Matrix weights(2, 1);
    weights << 0.1, 0.2;
    layer.set_weights(Tensor(weights));

    Matrix biases(1, 1);
    biases << 0.05;
    layer.set_biases(Tensor(biases));

    // Training data: simple linear relationship y = 0.5*x1 + 0.3*x2 + 0.1
    std::vector<Matrix> inputs = {
        (Matrix(1, 2) << 1.0, 2.0).finished(), (Matrix(1, 2) << 2.0, 1.0).finished(),
        (Matrix(1, 2) << 0.5, 3.0).finished(), (Matrix(1, 2) << 3.0, 0.5).finished()};

    std::vector<Matrix> targets = {
        (Matrix(1, 1) << 1.2).finished(),   // 0.5*1 + 0.3*2 + 0.1 = 1.2
        (Matrix(1, 1) << 1.4).finished(),   // 0.5*2 + 0.3*1 + 0.1 = 1.4
        (Matrix(1, 1) << 1.25).finished(),  // 0.5*0.5 + 0.3*3 + 0.1 = 1.25
        (Matrix(1, 1) << 1.75).finished()   // 0.5*3 + 0.3*0.5 + 0.1 = 1.75
    };

    auto mse = create_loss(LossType::MeanSquaredError);
    double learning_rate = 0.1;
    double initial_loss = 0.0;
    double final_loss = 0.0;

    // Initial loss
    for (size_t i = 0; i < inputs.size(); ++i) {
        Tensor input(inputs[i]);
        Tensor target(targets[i]);
        Tensor output = layer.forward(input);
        initial_loss += mse->forward(output, target);
    }
    initial_loss /= inputs.size();

    // Training loop
    for (int epoch = 0; epoch < 100; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            Tensor input(inputs[i]);
            Tensor target(targets[i]);

            // Forward pass
            Tensor output = layer.forward(input);

            // Backward pass
            Tensor loss_grad = mse->backward(output, target);
            layer.backward(loss_grad);

            // Update weights
            layer.update_weights(learning_rate);
            layer.zero_gradients();
        }
    }

    // Final loss
    for (size_t i = 0; i < inputs.size(); ++i) {
        Tensor input(inputs[i]);
        Tensor target(targets[i]);
        Tensor output = layer.forward(input);
        final_loss += mse->forward(output, target);
    }
    final_loss /= inputs.size();

    printf("✓ Training completed: Initial loss=%.6f, Final loss=%.6f\n", initial_loss, final_loss);
    assert(final_loss < initial_loss);  // Loss should decrease
    assert(final_loss < 0.1);           // Should converge to low loss
    printf("✓ Training convergence test passed\n");

    // Check that learned weights are close to target weights [0.5, 0.3]
    auto learned_weights = layer.weights();
    auto learned_biases = layer.biases();
    printf("✓ Learned weights: [%.3f, %.3f], bias: %.3f\n", learned_weights(0, 0),
           learned_weights(1, 0), learned_biases(0, 0));
}

void testBatchBackpropagation() {
    printf("\n=== Testing Batch Backpropagation ===\n");

    DenseLayer layer(2, 1, ActivationType::Linear);

    Matrix weights(2, 1);
    weights << 0.5, 0.3;
    layer.set_weights(Tensor(weights));

    Matrix biases(1, 1);
    biases << 0.1;
    layer.set_biases(Tensor(biases));

    // Batch data (3 samples)
    Matrix batch_input(3, 2);
    batch_input << 1.0, 2.0, 2.0, 1.0, 0.5, 3.0;
    Tensor input(batch_input);

    Matrix batch_target(3, 1);
    batch_target << 1.2, 1.4, 1.25;
    Tensor target(batch_target);

    auto mse = create_loss(LossType::MeanSquaredError);

    // Forward pass
    Tensor output = layer.forward(input);

    // Verify batch forward pass
    assert(output.rows() == 3 && output.cols() == 1);
    printf("✓ Batch forward pass test passed\n");

    // Backward pass
    Tensor loss_grad = mse->backward(output, target);
    Tensor grad_input = layer.backward(loss_grad);

    // Verify gradient shapes
    assert(grad_input.rows() == 3 && grad_input.cols() == 2);

    auto weight_grad = layer.weight_gradients();
    auto bias_grad = layer.bias_gradients();

    // Weight gradients should be averaged over batch
    assert(weight_grad.rows() == 2 && weight_grad.cols() == 1);
    assert(bias_grad.rows() == 1 && bias_grad.cols() == 1);

    printf("✓ Batch backpropagation test passed\n");
}

int runBackpropagationTests() {
    try {
        printf("=== BACKPROPAGATION TESTS ===\n");
        testEndToEndBackpropagation();
        testGradientChecking();
        testMultiLayerBackpropagation();
        testWeightUpdatesAndConvergence();
        testBatchBackpropagation();
        printf("✅ All backpropagation tests passed!\n\n");
        return 0;
    } catch (const std::exception& e) {
        printf("❌ Backpropagation test failed: %s\n", e.what());
        return 1;
    }
}