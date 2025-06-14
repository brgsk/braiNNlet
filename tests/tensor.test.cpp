//
// Created by Bartosz Roguski on 07/06/2025.
//
#include "core/nn/Tensor.hpp"

#include <cassert>
#include <cmath>

using namespace nn;

void testTensorCreation() {
    printf("\n=== Testing Tensor Creation ===\n");

    // Test default constructor and size constructor
    Tensor t1(2, 2);
    assert(t1.rows() == 2 && t1.cols() == 2);
    assert(t1.size() == 4);
    printf("✓ Size constructor test passed\n");

    // Test matrix constructor
    Matrix m = Matrix::Ones(2, 2);
    Tensor t2(m);
    assert((t2.data() - Matrix::Ones(2, 2)).norm() < 1e-10);
    printf("✓ Matrix constructor test passed\n");

    // Test move constructor
    Matrix m2 = Matrix::Constant(2, 2, 3.0);
    Tensor t3(std::move(m2));
    assert((t3.data() - Matrix::Constant(2, 2, 3.0)).norm() < 1e-10);
    printf("✓ Move constructor test passed\n");
}

void testElementAccess() {
    printf("\n=== Testing Element Access ===\n");

    Matrix m = Matrix::Zero(2, 2);
    m(0, 1) = 5.0;
    Tensor t(m);

    // Test const access
    assert(t(0, 1) == 5.0);
    assert(t(0, 0) == 0.0);
    printf("✓ Const element access test passed\n");

    // Test non-const access
    t(1, 0) = 3.0;
    assert(t(1, 0) == 3.0);
    printf("✓ Non-const element access test passed\n");
}

void testBasicOperations() {
    printf("\n=== Testing Basic Operations ===\n");

    Matrix m1 = Matrix::Ones(2, 2);
    Matrix m2 = Matrix::Constant(2, 2, 2.0);

    Tensor t1(m1);
    Tensor t2(m2);

    // Test addition
    Tensor sum = t1 + t2;
    assert((sum.data() - Matrix::Constant(2, 2, 3.0)).norm() < 1e-10);
    printf("✓ Addition test passed\n");

    // Test subtraction
    Tensor diff = t2 - t1;
    assert((diff.data() - Matrix::Constant(2, 2, 1.0)).norm() < 1e-10);
    printf("✓ Subtraction test passed\n");

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

void testScalarOperations() {
    printf("\n=== Testing Scalar Operations ===\n");

    Matrix m = Matrix::Ones(2, 2);
    Tensor t(m);

    // Test scalar multiplication
    Tensor prod = t * 3.0;
    assert((prod.data() - Matrix::Constant(2, 2, 3.0)).norm() < 1e-10);
    printf("✓ Scalar multiplication test passed\n");

    // Test scalar multiplication (commutative)
    Tensor prod2 = 3.0 * t;
    assert((prod2.data() - Matrix::Constant(2, 2, 3.0)).norm() < 1e-10);
    printf("✓ Commutative scalar multiplication test passed\n");

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

void testUtilityFunctions() {
    printf("\n=== Testing Utility Functions ===\n");

    Tensor t(2, 2);

    // Test zero
    t.fill(5.0);
    t.zero();
    assert((t.data() - Matrix::Zero(2, 2)).norm() < 1e-10);
    printf("✓ Zero function test passed\n");

    // Test fill
    t.fill(3.5);
    assert((t.data() - Matrix::Constant(2, 2, 3.5)).norm() < 1e-10);
    printf("✓ Fill function test passed\n");

    // Test random (just check it's not all zeros)
    t.random(-1.0, 1.0);
    assert(t.data().norm() > 1e-10);
    printf("✓ Random function test passed\n");
}

void testStatistics() {
    printf("\n=== Testing Statistics ===\n");

    Matrix m(2, 3);
    m << 1, 2, 3, 4, 5, 6;
    Tensor t(m);

    // Test sum
    double sum_all = t.sum();
    assert(std::abs(sum_all - 21.0) < 1e-10);
    printf("✓ Sum test passed\n");

    // Test mean
    double mean_all = t.mean();
    assert(std::abs(mean_all - 3.5) < 1e-10);
    printf("✓ Mean test passed\n");

    // Test norm
    double norm = t.norm();
    assert(std::abs(norm - std::sqrt(91.0)) < 1e-10);
    printf("✓ Norm test passed\n");
}

void testShapeOperations() {
    printf("\n=== Testing Shape Operations ===\n");

    Matrix m(2, 3);
    m << 1, 2, 3, 4, 5, 6;
    Tensor t(m);

    // Test resize
    t.resize(3, 2);
    assert(t.rows() == 3 && t.cols() == 2);
    printf("✓ Resize test passed\n");

    // Test that we can set new values after resize
    t.zero();
    t(0, 0) = 10;
    t(1, 1) = 20;
    assert(t(0, 0) == 10 && t(1, 1) == 20);
    printf("✓ Post-resize functionality test passed\n");
}

void testSerialization() {
    printf("\n=== Testing Serialization ===\n");

    Matrix m(2, 2);
    m << 1, 2, 3, 4;
    Tensor t(m);

    // Test to_vector (Eigen uses column-major storage)
    std::vector<double> vec = t.to_vector();
    assert(vec.size() == 4);
    assert(vec[0] == 1 && vec[1] == 3 && vec[2] == 2 && vec[3] == 4);
    printf("✓ To vector test passed\n");

    // Test from_vector
    Tensor t2;
    std::vector<double> test_vec = {5, 6, 7, 8, 9, 10};
    t2.from_vector(test_vec, 2, 3);
    assert(t2.rows() == 2 && t2.cols() == 3);
    assert(t2(0, 0) == 5 && t2(1, 2) == 10);
    printf("✓ From vector test passed\n");
}

void testEdgeCases() {
    printf("\n=== Testing Edge Cases ===\n");

    // Test empty tensor
    Tensor empty(0, 0);
    assert(empty.rows() == 0 && empty.cols() == 0 && empty.size() == 0);
    printf("✓ Empty tensor test passed\n");

    // Test single element tensor
    Tensor single(1, 1);
    single(0, 0) = 42.0;
    assert(single(0, 0) == 42.0);
    assert(single.sum() == 42.0);
    printf("✓ Single element tensor test passed\n");

    // Test dimension mismatch handling (should throw)
    try {
        Tensor t1(2, 2);
        Tensor t2(3, 3);
        Tensor result = t1 + t2;  // Should throw
        assert(false);            // Should not reach here
    } catch (const std::invalid_argument &) {
        printf("✓ Dimension mismatch handling test passed\n");
    }
}

int runTensorTests() {
    try {
        printf("=== TENSOR TESTS ===\n");
        testTensorCreation();
        testElementAccess();
        testBasicOperations();
        testMatrixOperations();
        testScalarOperations();
        testUtilityFunctions();
        testStatistics();
        testShapeOperations();
        testSerialization();
        testEdgeCases();
        printf("✅ All tensor tests passed!\n\n");
        return 0;
    } catch (const std::exception &e) {
        printf("❌ Tensor test failed: %s\n", e.what());
        return 1;
    }
}