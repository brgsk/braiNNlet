//
// Created by Bartosz Roguski on 07/06/2025.
//
#include "core/nn/DenseLayer.hpp"
#include "core/nn/Loss.hpp"

#include <cassert>
#include <cmath>
#include <vector>

using namespace nn;

void testSimpleNetworkTraining()
{
    printf("\n=== Testing Simple Network Training ===\n");

    // Create a simple 2-layer network: 2 -> 3 -> 1
    DenseLayer layer1(2, 3, ActivationType::ReLU);
    DenseLayer layer2(3, 1, ActivationType::Linear);

    // Set known weights for reproducible results
    Matrix weights1(2, 3);
    weights1 << 0.1, 0.2, -0.1, 0.15, -0.1, 0.2;
    layer1.set_weights(Tensor(weights1));

    Matrix biases1(1, 3);
    biases1 << 0.01, -0.02, 0.015;
    layer1.set_biases(Tensor(biases1));

    Matrix weights2(3, 1);
    weights2 << 0.3, 0.4, -0.2;
    layer2.set_weights(Tensor(weights2));

    Matrix biases2(1, 1);
    biases2 << 0.1;
    layer2.set_biases(Tensor(biases2));

    // Training data: simple linear relationship
    std::vector<Matrix> inputs = {
        (Matrix(1, 2) << 1.0, 2.0).finished(),
        (Matrix(1, 2) << 2.0, 1.0).finished(),
        (Matrix(1, 2) << 0.5, 3.0).finished(),
        (Matrix(1, 2) << 3.0, 0.5).finished()};

    std::vector<Matrix> targets = {
        (Matrix(1, 1) << 0.8).finished(),
        (Matrix(1, 1) << 0.6).finished(),
        (Matrix(1, 1) << 0.9).finished(),
        (Matrix(1, 1) << 0.4).finished()};

    auto mse = create_loss(LossType::MeanSquaredError);
    double learning_rate = 0.01;
    double initial_loss = 0.0;
    double final_loss = 0.0;

    // Calculate initial loss
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        Tensor input(inputs[i]);
        Tensor target(targets[i]);

        Tensor hidden = layer1.forward(input);
        Tensor output = layer2.forward(hidden);
        initial_loss += mse->forward(output, target);
    }
    initial_loss /= inputs.size();

    // Training loop
    for (int epoch = 0; epoch < 50; ++epoch)
    {
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            Tensor input(inputs[i]);
            Tensor target(targets[i]);

            // Forward pass
            Tensor hidden = layer1.forward(input);
            Tensor output = layer2.forward(hidden);

            // Backward pass
            Tensor loss_grad = mse->backward(output, target);
            Tensor grad_hidden = layer2.backward(loss_grad);
            layer1.backward(grad_hidden);

            // Update weights
            layer2.update_weights(learning_rate);
            layer1.update_weights(learning_rate);

            // Zero gradients
            layer2.zero_gradients();
            layer1.zero_gradients();
        }
    }

    // Calculate final loss
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        Tensor input(inputs[i]);
        Tensor target(targets[i]);

        Tensor hidden = layer1.forward(input);
        Tensor output = layer2.forward(hidden);
        final_loss += mse->forward(output, target);
    }
    final_loss /= inputs.size();

    printf("✓ Training completed: Initial loss=%.6f, Final loss=%.6f\n", initial_loss, final_loss);
    assert(final_loss < initial_loss); // Loss should decrease
    printf("✓ Network training convergence test passed\n");
}

void testBatchTraining()
{
    printf("\n=== Testing Batch Training ===\n");

    DenseLayer layer(2, 1, ActivationType::Linear);

    // Set initial weights
    Matrix weights(2, 1);
    weights << 0.5, 0.3;
    layer.set_weights(Tensor(weights));

    Matrix biases(1, 1);
    biases << 0.1;
    layer.set_biases(Tensor(biases));

    // Batch data (4 samples)
    Matrix batch_input(4, 2);
    batch_input << 1.0, 2.0, 2.0, 1.0, 0.5, 3.0, 3.0, 0.5;

    Matrix batch_target(4, 1);
    batch_target << 1.2, 1.4, 1.25, 1.75;

    Tensor input(batch_input);
    Tensor target(batch_target);

    auto mse = create_loss(LossType::MeanSquaredError);

    // Forward pass
    Tensor output = layer.forward(input);
    assert(output.rows() == 4 && output.cols() == 1);

    // Compute loss
    double loss = mse->forward(output, target);
    printf("✓ Batch forward pass completed (loss = %.6f)\n", loss);

    // Backward pass
    Tensor loss_grad = mse->backward(output, target);
    Tensor grad_input = layer.backward(loss_grad);

    // Verify gradient shapes
    assert(grad_input.rows() == 4 && grad_input.cols() == 2);
    assert(layer.weight_gradients().rows() == 2 && layer.weight_gradients().cols() == 1);
    assert(layer.bias_gradients().rows() == 1 && layer.bias_gradients().cols() == 1);

    printf("✓ Batch training test passed\n");
}

void testDifferentActivationFunctions()
{
    printf("\n=== Testing Different Activation Functions ===\n");

    // Test network with different activations
    DenseLayer relu_layer(2, 2, ActivationType::ReLU);
    DenseLayer sigmoid_layer(2, 2, ActivationType::Sigmoid);
    DenseLayer tanh_layer(2, 2, ActivationType::Tanh);
    DenseLayer linear_layer(2, 2, ActivationType::Linear);

    Matrix input_data(1, 2);
    input_data << 1.0, -0.5;
    Tensor input(input_data);

    // Test that all activations produce different outputs
    Tensor relu_output = relu_layer.forward(input);
    Tensor sigmoid_output = sigmoid_layer.forward(input);
    Tensor tanh_output = tanh_layer.forward(input);
    Tensor linear_output = linear_layer.forward(input);

    // Outputs should be different (probabilistically)
    assert((relu_output.data() - sigmoid_output.data()).norm() > 1e-10);
    assert((sigmoid_output.data() - tanh_output.data()).norm() > 1e-10);
    assert((tanh_output.data() - linear_output.data()).norm() > 1e-10);

    printf("✓ Different activation functions test passed\n");
}

void testDifferentLossFunctions()
{
    printf("\n=== Testing Different Loss Functions ===\n");

    auto mse = create_loss(LossType::MeanSquaredError);
    auto bce = create_loss(LossType::BinaryCrossEntropy);
    auto ce = create_loss(LossType::CrossEntropy);

    // Test MSE
    Matrix preds1(1, 2);
    preds1 << 1.0, 2.0;
    Matrix targets1(1, 2);
    targets1 << 0.5, 2.5;

    double mse_loss = mse->forward(Tensor(preds1), Tensor(targets1));
    assert(mse_loss > 0);
    printf("✓ MSE loss computation test passed\n");

    // Test BCE
    Matrix preds2(1, 2);
    preds2 << 0.8, 0.3;
    Matrix targets2(1, 2);
    targets2 << 1.0, 0.0;

    double bce_loss = bce->forward(Tensor(preds2), Tensor(targets2));
    assert(bce_loss > 0);
    printf("✓ BCE loss computation test passed\n");

    // Test CE
    Matrix preds3(1, 3);
    preds3 << 2.0, 1.0, 0.1;
    Matrix targets3(1, 3);
    targets3 << 0.0, 1.0, 0.0;

    double ce_loss = ce->forward(Tensor(preds3), Tensor(targets3));
    assert(ce_loss > 0);
    printf("✓ CE loss computation test passed\n");

    // All losses should be different
    assert(std::abs(mse_loss - bce_loss) > 1e-10);
    assert(std::abs(bce_loss - ce_loss) > 1e-10);
    printf("✓ Different loss functions test passed\n");
}

void testGradientFlow()
{
    printf("\n=== Testing Gradient Flow ===\n");

    // Create a 3-layer network to test gradient flow
    DenseLayer layer1(2, 3, ActivationType::Linear);
    DenseLayer layer2(3, 2, ActivationType::Linear);
    DenseLayer layer3(2, 1, ActivationType::Linear);

    Matrix input_data(1, 2);
    input_data << 1.0, 2.0;
    Tensor input(input_data);

    Matrix target_data(1, 1);
    target_data << 0.5;
    Tensor target(target_data);

    // Forward pass
    Tensor hidden1 = layer1.forward(input);
    Tensor hidden2 = layer2.forward(hidden1);
    Tensor output = layer3.forward(hidden2);

    auto mse = create_loss(LossType::MeanSquaredError);
    double loss = mse->forward(output, target);

    // Backward pass
    Tensor loss_grad = mse->backward(output, target);
    Tensor grad2 = layer3.backward(loss_grad);
    Tensor grad1 = layer2.backward(grad2);
    Tensor grad_input = layer1.backward(grad1);

    // Check that all layers have non-zero gradients
    assert(layer1.weight_gradients().norm() > 1e-10);
    assert(layer2.weight_gradients().norm() > 1e-10);
    assert(layer3.weight_gradients().norm() > 1e-10);

    assert(layer1.bias_gradients().norm() > 1e-10);
    assert(layer2.bias_gradients().norm() > 1e-10);
    assert(layer3.bias_gradients().norm() > 1e-10);

    printf("✓ Gradient flow test passed (loss = %.6f)\n", loss);
}

void testNumericalStability()
{
    printf("\n=== Testing Numerical Stability ===\n");

    // Test with very small and very large values
    DenseLayer layer(2, 2, ActivationType::Sigmoid);

    Matrix extreme_input(1, 2);
    extreme_input << 100.0, -100.0;
    Tensor input(extreme_input);

    Tensor output = layer.forward(input);

    // Output should be finite
    for (int i = 0; i < output.rows(); ++i)
    {
        for (int j = 0; j < output.cols(); ++j)
        {
            assert(std::isfinite(output(i, j)));
        }
    }

    printf("✓ Numerical stability test passed\n");

    // Test loss with extreme predictions
    auto bce = create_loss(LossType::BinaryCrossEntropy);

    Matrix extreme_preds(1, 2);
    extreme_preds << 1e-15, 1.0 - 1e-15;
    Matrix targets(1, 2);
    targets << 0.0, 1.0;

    double loss = bce->forward(Tensor(extreme_preds), Tensor(targets));
    assert(std::isfinite(loss));

    printf("✓ Loss numerical stability test passed\n");
}

void testMemoryManagement()
{
    printf("\n=== Testing Memory Management ===\n");

    // Test that objects can be created and destroyed without issues
    {
        DenseLayer layer(100, 50, ActivationType::ReLU);
        Matrix large_input = Matrix::Random(10, 100);
        Tensor input(large_input);

        Tensor output = layer.forward(input);
        assert(output.rows() == 10 && output.cols() == 50);
    } // layer should be destroyed here

    {
        auto mse = create_loss(LossType::MeanSquaredError);
        Matrix preds = Matrix::Random(100, 10);
        Matrix targets = Matrix::Random(100, 10);

        double loss = mse->forward(Tensor(preds), Tensor(targets));
        assert(std::isfinite(loss));
    } // mse should be destroyed here

    printf("✓ Memory management test passed\n");
}

void testComplexWorkflow()
{
    printf("\n=== Testing Complex Workflow ===\n");

    // Simulate a more complex training scenario
    DenseLayer layer1(3, 5, ActivationType::ReLU);
    DenseLayer layer2(5, 3, ActivationType::ReLU);
    DenseLayer layer3(3, 1, ActivationType::Linear);

    auto mse = create_loss(LossType::MeanSquaredError);
    double learning_rate = 0.01;

    // Generate some synthetic data
    std::vector<Tensor> train_inputs;
    std::vector<Tensor> train_targets;

    for (int i = 0; i < 10; ++i)
    {
        Matrix input_data = Matrix::Random(1, 3);
        Matrix target_data = Matrix::Random(1, 1);

        train_inputs.push_back(Tensor(input_data));
        train_targets.push_back(Tensor(target_data));
    }

    // Training for multiple epochs
    for (int epoch = 0; epoch < 5; ++epoch)
    {
        double epoch_loss = 0.0;

        for (size_t i = 0; i < train_inputs.size(); ++i)
        {
            // Forward pass
            Tensor h1 = layer1.forward(train_inputs[i]);
            Tensor h2 = layer2.forward(h1);
            Tensor output = layer3.forward(h2);

            // Compute loss
            epoch_loss += mse->forward(output, train_targets[i]);

            // Backward pass
            Tensor loss_grad = mse->backward(output, train_targets[i]);
            Tensor grad2 = layer3.backward(loss_grad);
            Tensor grad1 = layer2.backward(grad2);
            layer1.backward(grad1);

            // Update weights
            layer3.update_weights(learning_rate);
            layer2.update_weights(learning_rate);
            layer1.update_weights(learning_rate);

            // Zero gradients
            layer3.zero_gradients();
            layer2.zero_gradients();
            layer1.zero_gradients();
        }

        epoch_loss /= train_inputs.size();
        printf("✓ Epoch %d completed (avg loss = %.6f)\n", epoch + 1, epoch_loss);
    }

    printf("✓ Complex workflow test passed\n");
}

int runIntegrationTests()
{
    try
    {
        printf("=== INTEGRATION TESTS ===\n");
        testSimpleNetworkTraining();
        testBatchTraining();
        testDifferentActivationFunctions();
        testDifferentLossFunctions();
        testGradientFlow();
        testNumericalStability();
        testMemoryManagement();
        testComplexWorkflow();
        printf("✅ All integration tests passed!\n\n");
        return 0;
    }
    catch (const std::exception &e)
    {
        printf("❌ Integration test failed: %s\n", e.what());
        return 1;
    }
}
