//
// Created by Bartosz Roguski on 07/06/2025.
//
#include "src/core/nn/tensor.hpp"
#include <cassert>
#include <cmath>

void testTensorCreation() {
    printf("\n=== Testing Tensor Creation ===\n");
    
    Tensor t1(2, 2, 1);  // Should be zero-initialized
    Matrix expected = Matrix::Zero(2, 2);
    assert((t1.getData() - expected).norm() < 1e-10);
    printf("✓ Zero initialization test passed\n");
    
    Matrix m = Matrix::Ones(2, 2);
    Tensor t2(m);
    assert((t2.getData() - Matrix::Ones(2, 2)).norm() < 1e-10);
    printf("✓ Matrix constructor test passed\n");
}

void testTensorOperations() {
    printf("\n=== Testing Tensor Operations ===\n");
    
    Matrix m1 = Matrix::Ones(2, 2);
    Matrix m2 = Matrix::Constant(2, 2, 2.0);
    
    Tensor t1(m1);
    Tensor t2(m2);
    
    // Test addition
    Tensor sum = t1 + t2;
    assert((sum.getData() - Matrix::Constant(2, 2, 3.0)).norm() < 1e-10);
    printf("✓ Addition test passed\n");
    
    // Test multiplication
    Tensor prod = t1 * t2;
    assert((prod.getData() - Matrix::Constant(2, 2, 2.0)).norm() < 1e-10);
    printf("✓ Element-wise multiplication test passed\n");
    
    // Test division
    Tensor div = t2 / t1;
    assert((div.getData() - Matrix::Constant(2, 2, 2.0)).norm() < 1e-10);
    printf("✓ Element-wise division test passed\n");
}

void testScalarOperations() {
    printf("\n=== Testing Scalar Operations ===\n");
    
    Matrix m = Matrix::Ones(2, 2);
    Tensor t(m);
    
    // Test scalar multiplication
    Tensor prod = t * 3.0;
    assert((prod.getData() - Matrix::Constant(2, 2, 3.0)).norm() < 1e-10);
    printf("✓ Scalar multiplication test passed\n");
    
    // Test scalar division
    Tensor div = t / 2.0;
    assert((div.getData() - Matrix::Constant(2, 2, 0.5)).norm() < 1e-10);
    printf("✓ Scalar division test passed\n");
    
    // Test compound operators
    t *= 2.0;
    assert((t.getData() - Matrix::Constant(2, 2, 2.0)).norm() < 1e-10);
    printf("✓ Compound multiplication test passed\n");
    
    t /= 4.0;
    assert((t.getData() - Matrix::Constant(2, 2, 0.5)).norm() < 1e-10);
    printf("✓ Compound division test passed\n");
}

int main() {
    try {
        testTensorCreation();
        testTensorOperations();
        testScalarOperations();
        printf("\n✅ All tests passed!\n");
    } catch (const std::exception& e) {
        printf("\n❌ Test failed: %s\n", e.what());
        return 1;
    }
    return 0;
}
