//
// Created by Bartosz Roguski on 07/06/2025.
//
// Main test runner for the braiNNlet neural network library
//

#include <cstdio>

// Test function declarations from separate test files
extern int runTensorTests();
extern int runActivationTests();
extern int runLossTests();
extern int runDenseLayerTests();
extern int runIntegrationTests();

int main() {
    printf("🧠 braiNNlet Neural Network Library - Test Suite\n");
    printf("================================================\n\n");

    int total_failures = 0;

    // Run tensor tests
    printf("1. Running Tensor Tests...\n");
    total_failures += runTensorTests();

    // Run activation function tests
    printf("2. Running Activation Function Tests...\n");
    total_failures += runActivationTests();

    // Run loss function tests
    printf("3. Running Loss Function Tests...\n");
    total_failures += runLossTests();

    // Run dense layer tests
    printf("4. Running Dense Layer Tests...\n");
    total_failures += runDenseLayerTests();

    // Run integration tests
    printf("5. Running Integration Tests...\n");
    total_failures += runIntegrationTests();

    // Summary
    printf("================================================\n");
    if (total_failures == 0) {
        printf("🎉 ALL TESTS PASSED! The braiNNlet library is working correctly.\n");
        printf("Total test suites run: 5\n");
        printf("Status: ✅ SUCCESS\n");
        return 0;
    } else {
        printf("❌ SOME TESTS FAILED!\n");
        printf("Number of test suites with failures: %d\n", total_failures);
        printf("Status: ❌ FAILURE\n");
        return 1;
    }
}