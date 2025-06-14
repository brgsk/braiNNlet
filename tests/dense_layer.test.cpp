//
// Created by Bartosz Roguski on 07/06/2025.
//
#include "core/nn/DenseLayer.hpp"

#include <cassert>
#include <cmath>

using namespace nn;

void testDenseLayerCreation()
{
    printf("\n=== Testing Dense Layer Creation ===\n");

    DenseLayer layer(2, 3, ActivationType::ReLU);

    // Test dimensions
    assert(layer.input_size() == 2);
    assert(layer.output_size() == 3);
    assert(layer.activation().type() == ActivationType::ReLU);
    printf("✓ Layer dimensions test passed\n");

    // Test that weights and biases are initialized
    auto weights = layer.weights();
    auto biases = layer.biases();
    assert(weights.rows() == 2 && weights.cols() == 3);
    assert(biases.rows() == 1 && biases.cols() == 3);
    printf("✓ Weight and bias initialization test passed\n");

    // Test that weights are not all zero (random initialization)
    double weight_sum = weights.sum();
    assert(std::abs(weight_sum) > 1e-10); // Should not be zero
    printf("✓ Random weight initialization test passed\n");

    // Test layer properties
    assert(layer.has_parameters() == true);
    assert(layer.parameter_count() == 2 * 3 + 3); // weights + biases
    printf("✓ Layer properties test passed\n");
}

void testDenseLayerForward()
{
    printf("\n=== Testing Dense Layer Forward Pass ===\n");

    DenseLayer layer(2, 2, ActivationType::Linear);

    // Set known weights and biases for predictable output
    Matrix weights(2, 2);
    weights << 0.5, 0.3, 0.2, 0.4;
    layer.set_weights(Tensor(weights));

    Matrix biases(1, 2);
    biases << 0.1, 0.2;
    layer.set_biases(Tensor(biases));

    // Test input
    Matrix input_data(1, 2);
    input_data << 2.0, 3.0;
    Tensor input(input_data);

    Tensor output = layer.forward(input);

    // Expected output: [2, 3] * [[0.5, 0.3], [0.2, 0.4]] + [0.1, 0.2]
    //                = [1.0+0.6, 0.6+1.2] + [0.1, 0.2] = [1.7, 2.0]
    Matrix expected(1, 2);
    expected << 1.7, 2.0;

    assert((output.data() - expected).norm() < 1e-10);
    printf("✓ Forward pass calculation test passed\n");
}

void testDenseLayerBackward()
{
    printf("\n=== Testing Dense Layer Backward Pass ===\n");

    DenseLayer layer(2, 2, ActivationType::Linear);

    // Set known weights
    Matrix weights(2, 2);
    weights << 0.5, 0.3, 0.2, 0.4;
    layer.set_weights(Tensor(weights));

    Matrix biases(1, 2);
    biases << 0.1, 0.2;
    layer.set_biases(Tensor(biases));

    // Forward pass first to set up internal state
    Matrix input_data(1, 2);
    input_data << 2.0, 3.0;
    Tensor input(input_data);
    layer.forward(input);

    // Test backward pass
    Matrix grad_output_data(1, 2);
    grad_output_data << 1.0, 1.0;
    Tensor grad_output(grad_output_data);

    Tensor grad_input = layer.backward(grad_output);

    // Expected gradient w.r.t. input: grad_output * weights^T
    // [1, 1] * [[0.5, 0.2], [0.3, 0.4]] = [0.8, 0.6]
    Matrix expected_grad_input(1, 2);
    expected_grad_input << 0.8, 0.6;

    assert((grad_input.data() - expected_grad_input).norm() < 1e-10);
    printf("✓ Backward pass input gradient test passed\n");

    // Check that weight and bias gradients are computed
    auto weight_grad = layer.weight_gradients();
    auto bias_grad = layer.bias_gradients();

    // Weight gradients: input^T * grad_output = [[2], [3]] * [1, 1] = [[2, 2], [3, 3]]
    Matrix expected_weight_grad(2, 2);
    expected_weight_grad << 2.0, 2.0, 3.0, 3.0;

    // Bias gradients: sum(grad_output) = [1, 1]
    Matrix expected_bias_grad(1, 2);
    expected_bias_grad << 1.0, 1.0;

    assert((weight_grad.data() - expected_weight_grad).norm() < 1e-10);
    assert((bias_grad.data() - expected_bias_grad).norm() < 1e-10);
    printf("✓ Weight and bias gradient computation test passed\n");
}

void testDenseLayerWithActivation()
{
    printf("\n=== Testing Dense Layer with ReLU Activation ===\n");

    DenseLayer layer(2, 2, ActivationType::ReLU);

    // Set weights that will produce some negative values
    Matrix weights(2, 2);
    weights << 0.5, -0.3, -0.2, 0.4;
    layer.set_weights(Tensor(weights));

    Matrix biases(1, 2);
    biases << -0.1, 0.2;
    layer.set_biases(Tensor(biases));

    // Test input
    Matrix input_data(1, 2);
    input_data << 2.0, 3.0;
    Tensor input(input_data);

    Tensor output = layer.forward(input);

    // Expected before activation: [2, 3] * [[0.5, -0.3], [-0.2, 0.4]] + [-0.1, 0.2]
    //                           = [1.0-0.6, -0.6+1.2] + [-0.1, 0.2] = [0.3, 0.8]
    // After ReLU: [0.3, 0.8] (both positive, so unchanged)
    Matrix expected(1, 2);
    expected << 0.3, 0.8;

    assert((output.data() - expected).norm() < 1e-10);
    printf("✓ Forward pass with ReLU test passed\n");

    // Test backward pass with ReLU
    Matrix grad_output_data(1, 2);
    grad_output_data << 1.0, 1.0;
    Tensor grad_output(grad_output_data);

    Tensor grad_input = layer.backward(grad_output);

    // ReLU derivative is 1 for positive values, 0 for negative
    // Both outputs were positive, so activation gradient is [1, 1]
    // Input gradient: [1, 1] * [[0.5, -0.2], [-0.3, 0.4]] = [0.2, 0.2]
    Matrix expected_grad_input(1, 2);
    expected_grad_input << 0.2, 0.2;

    assert((grad_input.data() - expected_grad_input).norm() < 1e-10);
    printf("✓ Backward pass with ReLU test passed\n");
}

void testDenseLayerParameterUpdates()
{
    printf("\n=== Testing Dense Layer Parameter Updates ===\n");

    DenseLayer layer(2, 2, ActivationType::Linear);

    // Set initial weights and biases
    Matrix weights(2, 2);
    weights << 1.0, 2.0, 3.0, 4.0;
    layer.set_weights(Tensor(weights));

    Matrix biases(1, 2);
    biases << 0.5, 1.0;
    layer.set_biases(Tensor(biases));

    // Forward and backward pass to compute gradients
    Matrix input_data(1, 2);
    input_data << 1.0, 2.0;
    Tensor input(input_data);

    layer.forward(input);

    Matrix grad_output_data(1, 2);
    grad_output_data << 0.1, 0.2;
    Tensor grad_output(grad_output_data);

    layer.backward(grad_output);

    // Update parameters with learning rate 0.1
    double learning_rate = 0.1;
    layer.update_weights(learning_rate);

    // Expected weight gradients: [[1], [2]] * [0.1, 0.2] = [[0.1, 0.2], [0.2, 0.4]]
    // New weights: [[1, 2], [3, 4]] - 0.1 * [[0.1, 0.2], [0.2, 0.4]] = [[0.99, 1.98], [2.98, 3.96]]
    Matrix expected_weights(2, 2);
    expected_weights << 0.99, 1.98, 2.98, 3.96;

    // Expected bias gradients: [0.1, 0.2]
    // New biases: [0.5, 1.0] - 0.1 * [0.1, 0.2] = [0.49, 0.98]
    Matrix expected_biases(1, 2);
    expected_biases << 0.49, 0.98;

    auto updated_weights = layer.weights();
    auto updated_biases = layer.biases();

    assert((updated_weights.data() - expected_weights).norm() < 1e-10);
    assert((updated_biases.data() - expected_biases).norm() < 1e-10);
    printf("✓ Parameter update test passed\n");
}

void testDenseLayerGradientZeroing()
{
    printf("\n=== Testing Dense Layer Gradient Zeroing ===\n");

    DenseLayer layer(2, 2, ActivationType::Linear);

    // Forward and backward pass to create gradients
    Matrix input_data(1, 2);
    input_data << 1.0, 2.0;
    Tensor input(input_data);

    layer.forward(input);

    Matrix grad_output_data(1, 2);
    grad_output_data << 1.0, 1.0;
    Tensor grad_output(grad_output_data);

    layer.backward(grad_output);

    // Verify gradients exist
    auto weight_grad = layer.weight_gradients();
    auto bias_grad = layer.bias_gradients();
    assert(weight_grad.norm() > 1e-10);
    assert(bias_grad.norm() > 1e-10);
    printf("✓ Gradients created test passed\n");

    // Zero gradients
    layer.zero_gradients();

    weight_grad = layer.weight_gradients();
    bias_grad = layer.bias_gradients();

    assert(weight_grad.norm() < 1e-10);
    assert(bias_grad.norm() < 1e-10);
    printf("✓ Gradient zeroing test passed\n");
}

void testDenseLayerBatchProcessing()
{
    printf("\n=== Testing Dense Layer Batch Processing ===\n");

    DenseLayer layer(2, 2, ActivationType::Linear);

    // Set known weights and biases
    Matrix weights(2, 2);
    weights << 0.5, 0.3, 0.2, 0.4;
    layer.set_weights(Tensor(weights));

    Matrix biases(1, 2);
    biases << 0.1, 0.2;
    layer.set_biases(Tensor(biases));

    // Batch input (2 samples)
    Matrix batch_input(2, 2);
    batch_input << 1.0, 2.0, 3.0, 4.0;
    Tensor input(batch_input);

    Tensor output = layer.forward(input);

    // Expected output for sample 1: [1, 2] * [[0.5, 0.3], [0.2, 0.4]] + [0.1, 0.2] = [1.0, 1.3]
    // Expected output for sample 2: [3, 4] * [[0.5, 0.3], [0.2, 0.4]] + [0.1, 0.2] = [2.4, 2.7]
    Matrix expected(2, 2);
    expected << 1.0, 1.3, 2.4, 2.7;

    assert((output.data() - expected).norm() < 1e-10);
    printf("✓ Batch forward pass test passed\n");

    // Test batch backward pass
    Matrix batch_grad_output(2, 2);
    batch_grad_output << 1.0, 1.0, 0.5, 0.5;
    Tensor grad_output(batch_grad_output);

    Tensor grad_input = layer.backward(grad_output);

    // Expected input gradients:
    // Sample 1: [1, 1] * [[0.5, 0.2], [0.3, 0.4]] = [0.8, 0.6]
    // Sample 2: [0.5, 0.5] * [[0.5, 0.2], [0.3, 0.4]] = [0.4, 0.3]
    Matrix expected_grad_input(2, 2);
    expected_grad_input << 0.8, 0.6, 0.4, 0.3;

    assert((grad_input.data() - expected_grad_input).norm() < 1e-10);
    printf("✓ Batch backward pass test passed\n");
}

void testDenseLayerInitialization()
{
    printf("\n=== Testing Dense Layer Initialization Methods ===\n");

    DenseLayer layer(3, 2, ActivationType::ReLU);

    // Test Xavier initialization
    layer.xavier_init();
    auto xavier_weights = layer.weights();

    // Check that weights are in reasonable range for Xavier init
    double limit = std::sqrt(6.0 / (3 + 2)); // sqrt(6/(fan_in + fan_out))
    bool in_range = true;
    for (int i = 0; i < xavier_weights.rows(); ++i)
    {
        for (int j = 0; j < xavier_weights.cols(); ++j)
        {
            if (std::abs(xavier_weights(i, j)) > limit)
            {
                in_range = false;
                break;
            }
        }
    }
    assert(in_range);
    printf("✓ Xavier initialization test passed\n");

    // Test He initialization
    layer.he_init();
    auto he_weights = layer.weights();

    // Check that weights are different from Xavier (probabilistically)
    assert((he_weights.data() - xavier_weights.data()).norm() > 1e-10);
    printf("✓ He initialization test passed\n");
}

void testDenseLayerEdgeCases()
{
    printf("\n=== Testing Dense Layer Edge Cases ===\n");

    // Test single input/output
    DenseLayer single_layer(1, 1, ActivationType::Linear);

    Matrix single_input(1, 1);
    single_input << 5.0;
    Tensor input(single_input);

    Tensor output = single_layer.forward(input);
    assert(output.rows() == 1 && output.cols() == 1);
    printf("✓ Single input/output test passed\n");

    // Test large layer
    DenseLayer large_layer(10, 5, ActivationType::ReLU);
    assert(large_layer.parameter_count() == 10 * 5 + 5); // 55 parameters
    printf("✓ Large layer test passed\n");

    // Test layer name
    std::string name = large_layer.name();
    assert(name.find("Dense") != std::string::npos);
    assert(name.find("10") != std::string::npos);
    assert(name.find("5") != std::string::npos);
    assert(name.find("ReLU") != std::string::npos);
    printf("✓ Layer name test passed\n");
}

int runDenseLayerTests()
{
    try
    {
        printf("=== DENSE LAYER TESTS ===\n");
        testDenseLayerCreation();
        testDenseLayerForward();
        testDenseLayerBackward();
        testDenseLayerWithActivation();
        testDenseLayerParameterUpdates();
        testDenseLayerGradientZeroing();
        testDenseLayerBatchProcessing();
        testDenseLayerInitialization();
        testDenseLayerEdgeCases();
        printf("✅ All dense layer tests passed!\n\n");
        return 0;
    }
    catch (const std::exception &e)
    {
        printf("❌ Dense layer test failed: %s\n", e.what());
        return 1;
    }
}