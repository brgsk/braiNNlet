//
// Created by Bartosz Roguski on 07/06/2025.
//
#include <cassert>
#include <cmath>

#include "src/core/nn/activations.hpp"

void testReLUActivation() {
    printf("\n=== Testing ReLU Activation ===\n");

    auto relu = create_activation(ActivationType::ReLU);

    // Test forward pass
    Matrix input(2, 3);
    input << -1, 0, 1, -2, 3, -0.5;

    Tensor input_tensor(input);
    Tensor output = relu->forward(input_tensor);

    // Expected: negative values should be 0, positive unchanged
    Matrix expected(2, 3);
    expected << 0, 0, 1, 0, 3, 0;

    assert((output.data() - expected).norm() < 1e-10);
    printf("✓ ReLU forward pass test passed\n");

    // Test backward pass - should be 0 for negative inputs, 1 for positive
    Tensor grad_output(2, 3);
    grad_output.fill(1.0);  // Gradient from next layer

    Tensor grad_input = relu->backward(grad_output);

    Matrix expected_grad(2, 3);
    expected_grad << 0, 0, 1, 0, 1, 0;

    assert((grad_input.data() - expected_grad).norm() < 1e-10);
    printf("✓ ReLU backward pass test passed\n");

    // Test properties
    assert(relu->name() == "ReLU");
    assert(relu->type() == ActivationType::ReLU);
    printf("✓ ReLU properties test passed\n");
}

void testLinearActivation() {
    printf("\n=== Testing Linear Activation ===\n");

    auto linear = create_activation(ActivationType::Linear);

    Matrix input(2, 2);
    input << 1.5, -2.3, 0.7, 4.1;

    Tensor input_tensor(input);
    Tensor output = linear->forward(input_tensor);

    // Linear should return input unchanged
    assert((output.data() - input).norm() < 1e-10);
    printf("✓ Linear forward pass test passed\n");

    // Test backward pass - should return gradient unchanged
    Tensor grad_output(2, 2);
    grad_output.fill(2.5);

    Tensor grad_input = linear->backward(grad_output);

    // Should return the gradient unchanged
    assert((grad_input.data() - grad_output.data()).norm() < 1e-10);
    printf("✓ Linear backward pass test passed\n");

    // Test properties
    assert(linear->name() == "Linear");
    assert(linear->type() == ActivationType::Linear);
    printf("✓ Linear properties test passed\n");
}

void testTanhActivation() {
    printf("\n=== Testing Tanh Activation ===\n");

    auto tanh_act = create_activation(ActivationType::Tanh);

    Matrix input(2, 2);
    input << 0, 1, -1, 2;

    Tensor input_tensor(input);
    Tensor output = tanh_act->forward(input_tensor);

    // Expected tanh values
    Matrix expected(2, 2);
    expected << std::tanh(0), std::tanh(1), std::tanh(-1), std::tanh(2);

    assert((output.data() - expected).norm() < 1e-10);
    printf("✓ Tanh forward pass test passed\n");

    // Test backward pass - derivative is 1 - tanh^2(x)
    Tensor grad_output(2, 2);
    grad_output.fill(1.0);

    Tensor grad_input = tanh_act->backward(grad_output);

    // Expected derivative: 1 - tanh^2(x)
    Matrix expected_grad(2, 2);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            double tanh_val = std::tanh(input(i, j));
            expected_grad(i, j) = 1.0 - tanh_val * tanh_val;
        }
    }

    assert((grad_input.data() - expected_grad).norm() < 1e-10);
    printf("✓ Tanh backward pass test passed\n");

    // Test properties
    assert(tanh_act->name() == "Tanh");
    assert(tanh_act->type() == ActivationType::Tanh);
    printf("✓ Tanh properties test passed\n");
}

void testSigmoidActivation() {
    printf("\n=== Testing Sigmoid Activation ===\n");

    auto sigmoid = create_activation(ActivationType::Sigmoid);

    Matrix input(2, 2);
    input << 0, 1, -1, 2;

    Tensor input_tensor(input);
    Tensor output = sigmoid->forward(input_tensor);

    // Expected sigmoid values: 1 / (1 + exp(-x))
    Matrix expected(2, 2);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            expected(i, j) = 1.0 / (1.0 + std::exp(-input(i, j)));
        }
    }

    assert((output.data() - expected).norm() < 1e-10);
    printf("✓ Sigmoid forward pass test passed\n");

    // Test backward pass - derivative is sigmoid(x) * (1 - sigmoid(x))
    Tensor grad_output(2, 2);
    grad_output.fill(1.0);

    Tensor grad_input = sigmoid->backward(grad_output);

    // Expected derivative: sigmoid(x) * (1 - sigmoid(x))
    Matrix expected_grad(2, 2);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            double sig_val = 1.0 / (1.0 + std::exp(-input(i, j)));
            expected_grad(i, j) = sig_val * (1.0 - sig_val);
        }
    }

    assert((grad_input.data() - expected_grad).norm() < 1e-10);
    printf("✓ Sigmoid backward pass test passed\n");

    // Test properties
    assert(sigmoid->name() == "Sigmoid");
    assert(sigmoid->type() == ActivationType::Sigmoid);
    printf("✓ Sigmoid properties test passed\n");
}

void testActivationFactoryFunctions() {
    printf("\n=== Testing Activation Factory Functions ===\n");

    // Test string to activation type
    assert(activation_from_string("ReLU") == ActivationType::ReLU);
    assert(activation_from_string("relu") == ActivationType::ReLU);
    assert(activation_from_string("Linear") == ActivationType::Linear);
    assert(activation_from_string("linear") == ActivationType::Linear);
    assert(activation_from_string("Tanh") == ActivationType::Tanh);
    assert(activation_from_string("tanh") == ActivationType::Tanh);
    assert(activation_from_string("Sigmoid") == ActivationType::Sigmoid);
    assert(activation_from_string("sigmoid") == ActivationType::Sigmoid);
    printf("✓ String to activation type conversion test passed\n");

    // Test activation type to string
    assert(activation_to_string(ActivationType::ReLU) == "ReLU");
    assert(activation_to_string(ActivationType::Linear) == "Linear");
    assert(activation_to_string(ActivationType::Tanh) == "Tanh");
    assert(activation_to_string(ActivationType::Sigmoid) == "Sigmoid");
    printf("✓ Activation type to string conversion test passed\n");

    // Test unknown string handling
    try {
        activation_from_string("Unknown");
        assert(false);  // Should throw
    } catch (const std::invalid_argument&) {
        printf("✓ Unknown activation string handling test passed\n");
    }

    // Test factory function creates correct types
    auto relu = create_activation(ActivationType::ReLU);
    auto linear = create_activation(ActivationType::Linear);
    auto tanh_act = create_activation(ActivationType::Tanh);
    auto sigmoid = create_activation(ActivationType::Sigmoid);

    assert(relu->type() == ActivationType::ReLU);
    assert(linear->type() == ActivationType::Linear);
    assert(tanh_act->type() == ActivationType::Tanh);
    assert(sigmoid->type() == ActivationType::Sigmoid);
    printf("✓ Factory function test passed\n");
}

void testActivationEdgeCases() {
    printf("\n=== Testing Activation Edge Cases ===\n");

    auto relu = create_activation(ActivationType::ReLU);
    auto sigmoid = create_activation(ActivationType::Sigmoid);

    // Test with very large values
    Matrix large_input(1, 2);
    large_input << 100, -100;

    Tensor large_tensor(large_input);

    // ReLU should handle large values fine
    Tensor relu_output = relu->forward(large_tensor);
    assert(relu_output(0, 0) == 100 && relu_output(0, 1) == 0);
    printf("✓ ReLU large values test passed\n");

    // Sigmoid should be numerically stable
    Tensor sigmoid_output = sigmoid->forward(large_tensor);
    assert(sigmoid_output(0, 0) > 0.99);  // Should be close to 1
    assert(sigmoid_output(0, 1) < 0.01);  // Should be close to 0
    printf("✓ Sigmoid numerical stability test passed\n");

    // Test with zero-sized tensors
    Tensor empty_tensor(0, 0);
    Tensor empty_output = relu->forward(empty_tensor);
    assert(empty_output.rows() == 0 && empty_output.cols() == 0);
    printf("✓ Empty tensor handling test passed\n");
}

int runActivationTests() {
    try {
        printf("=== ACTIVATION FUNCTION TESTS ===\n");
        testReLUActivation();
        testLinearActivation();
        testTanhActivation();
        testSigmoidActivation();
        testActivationFactoryFunctions();
        testActivationEdgeCases();
        printf("✅ All activation function tests passed!\n\n");
        return 0;
    } catch (const std::exception& e) {
        printf("❌ Activation test failed: %s\n", e.what());
        return 1;
    }
}