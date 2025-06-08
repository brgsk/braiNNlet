//
// Created by Bartosz Roguski on 07/06/2025.
//
#include "src/core/nn/tensor.hpp"

#include <cassert>
#include <cmath>

void testTensorCreation() {
    printf("\n=== Testing Tensor Creation ===\n");

    Tensor t1(2, 2);  // Should be zero-initialized
    Matrix expected = Matrix::Zero(2, 2);
    assert((t1.data() - expected).norm() < 1e-10);
    printf("✓ Zero initialization test passed\n");

    Matrix m = Matrix::Ones(2, 2);
    Tensor t2(m);
    assert((t2.data() - Matrix::Ones(2, 2)).norm() < 1e-10);
    printf("✓ Matrix constructor test passed\n");

    // Test shape access
    assert(t1.rows() == 2 && t1.cols() == 2);
    auto shape = t1.shape();
    assert(shape.first == 2 && shape.second == 2);
    printf("✓ Shape access test passed\n");
}

void testElementAccess() {
    printf("\n=== Testing Element Access ===\n");

    Matrix m = Matrix::Zero(2, 2);
    m(0, 1) = 5.0;
    Tensor t(m);

    assert(t(0, 1) == 5.0);
    t(1, 0) = 3.0;
    assert(t(1, 0) == 3.0);
    printf("✓ Element access test passed\n");
}

void testTensorOperations() {
    printf("\n=== Testing Tensor Operations ===\n");

    Matrix m1 = Matrix::Ones(2, 2);
    Matrix m2 = Matrix::Constant(2, 2, 2.0);

    Tensor t1(m1);
    Tensor t2(m2);

    // Test addition
    Tensor sum = t1 + t2;
    assert((sum.data() - Matrix::Constant(2, 2, 3.0)).norm() < 1e-10);
    printf("✓ Addition test passed\n");

    // Test hadamard (element-wise multiplication)
    Tensor prod = t1.hadamard(t2);
    assert((prod.data() - Matrix::Constant(2, 2, 2.0)).norm() < 1e-10);
    printf("✓ Hadamard (element-wise) multiplication test passed\n");

    // Test scalar division
    Tensor div = t2 / 2.0;
    assert((div.data() - Matrix::Constant(2, 2, 1.0)).norm() < 1e-10);
    printf("✓ Scalar division test passed\n");

    // Test element-wise division
    Tensor div_elem = t2 / t1;
    assert((div_elem.data() - Matrix::Constant(2, 2, 2.0)).norm() < 1e-10);
    printf("✓ Element-wise division test passed\n");

    // Test compound assignments
    Tensor t3 = t1;
    t3 += t2;
    assert((t3.data() - Matrix::Constant(2, 2, 3.0)).norm() < 1e-10);
    printf("✓ Compound addition test passed\n");

    t3 -= t1;
    assert((t3.data() - Matrix::Constant(2, 2, 2.0)).norm() < 1e-10);
    printf("✓ Compound subtraction test passed\n");
}

void testMatrixOperations() {
    printf("\n=== Testing Matrix Operations ===\n");

    // Test matrix multiplication
    Matrix a(2, 3);
    a << 1, 2, 3, 4, 5, 6;

    Matrix b(3, 2);
    b << 1, 2, 3, 4, 5, 6;

    Tensor ta(a);
    Tensor tb(b);

    Tensor result = ta * tb;
    Matrix expected(2, 2);
    expected << 22, 28, 49, 64;

    assert((result.data() - expected).norm() < 1e-10);
    printf("✓ Matrix multiplication test passed\n");

    // Test transpose
    Tensor transposed = ta.transpose();
    assert(transposed.rows() == 3 && transposed.cols() == 2);
    assert(transposed(0, 0) == 1 && transposed(0, 1) == 4);
    printf("✓ Transpose test passed\n");
}

void testBroadcasting() {
    printf("\n=== Testing Broadcasting ===\n");

    // Test bias addition: (2, 3) + (1, 3)
    Matrix data(2, 3);
    data << 1, 2, 3, 4, 5, 6;

    Matrix bias(1, 3);
    bias << 10, 20, 30;

    Tensor t_data(data);
    Tensor t_bias(bias);

    Tensor result = t_data.broadcast_add(t_bias);
    Matrix expected(2, 3);
    expected << 11, 22, 33, 14, 25, 36;

    assert((result.data() - expected).norm() < 1e-10);
    printf("✓ Broadcasting addition test passed\n");
}

void testReductions() {
    printf("\n=== Testing Reduction Operations ===\n");

    Matrix m(2, 3);
    m << 1, 2, 3, 4, 5, 6;

    Tensor t(m);

    // Test sum all
    double sum_all = t.sum();
    assert(std::abs(sum_all - 21.0) < 1e-10);
    printf("✓ Sum all test passed\n");

    // Test sum along axis 0 (columns)
    Tensor sum_cols = t.sum(0);
    assert(std::abs(sum_cols(0, 0) - 5.0) < 1e-10);
    assert(std::abs(sum_cols(0, 1) - 7.0) < 1e-10);
    assert(std::abs(sum_cols(0, 2) - 9.0) < 1e-10);
    printf("✓ Sum along columns test passed\n");

    // Test mean
    double mean_all = t.mean();
    assert(std::abs(mean_all - 3.5) < 1e-10);
    printf("✓ Mean test passed\n");

    // Test norm
    double norm = t.norm();
    assert(std::abs(norm - std::sqrt(91.0)) < 1e-10);
    printf("✓ Norm test passed\n");
}

void testFunctionApplication() {
    printf("\n=== Testing Function Application ===\n");

    Matrix m = Matrix::Ones(2, 2);
    Tensor t(m);

    // Test ReLU-like function
    auto relu = [](double x) { return std::max(0.0, x); };
    Tensor result = t.apply(relu);
    assert((result.data() - Matrix::Ones(2, 2)).norm() < 1e-10);

    // Test square function
    auto square = [](double x) { return x * x; };
    Tensor squared = t.apply(square);
    assert((squared.data() - Matrix::Ones(2, 2)).norm() < 1e-10);

    printf("✓ Function application test passed\n");
}

void testReshape() {
    printf("\n=== Testing Reshape ===\n");

    Matrix m(2, 3);
    m << 1, 2, 3, 4, 5, 6;

    Tensor t(m);
    t.resize(3, 2);

    // Test that dimensions are correct
    assert(t.rows() == 3 && t.cols() == 2);
    printf("✓ Reshape dimensions test passed\n");

    // Test that we can set new values after resize
    t.zero();
    t(0, 0) = 10;
    t(1, 1) = 20;
    assert(t(0, 0) == 10 && t(1, 1) == 20);

    printf("✓ Reshape test passed\n");
}

void testScalarOperations() {
    printf("\n=== Testing Scalar Operations ===\n");

    Matrix m = Matrix::Ones(2, 2);
    Tensor t(m);

    // Test scalar multiplication
    Tensor prod = t * 3.0;
    assert((prod.data() - Matrix::Constant(2, 2, 3.0)).norm() < 1e-10);
    printf("✓ Scalar multiplication test passed\n");

    // Test scalar division
    Tensor div = t / 2.0;
    assert((div.data() - Matrix::Constant(2, 2, 0.5)).norm() < 1e-10);
    printf("✓ Scalar division test passed\n");

    // Test compound operators
    t *= 2.0;
    assert((t.data() - Matrix::Constant(2, 2, 2.0)).norm() < 1e-10);
    printf("✓ Compound multiplication test passed\n");

    t /= 4.0;
    assert((t.data() - Matrix::Constant(2, 2, 0.5)).norm() < 1e-10);
    printf("✓ Compound division test passed\n");
}

int runTensorTests() {
    try {
        printf("=== TENSOR TESTS ===\n");
        testTensorCreation();
        testElementAccess();
        testTensorOperations();
        testMatrixOperations();
        testBroadcasting();
        testReductions();
        testFunctionApplication();
        testReshape();
        testScalarOperations();
        printf("✅ All tensor tests passed!\n\n");
        return 0;
    } catch (const std::exception& e) {
        printf("❌ Tensor test failed: %s\n", e.what());
        return 1;
    }
}