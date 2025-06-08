#include <iostream>

#include "src/core/nn/dense_layer.hpp"
#include "src/core/nn/tensor.hpp"

int main() {
    std::cout << "\nCreate tensor..." << std::endl;
    Tensor tensor(10, 10);
    std::cout << tensor << std::endl;

    std::cout << "\nUpdate tensor..." << std::endl;
    tensor.data() = Matrix::Ones(10, 10);
    std::cout << tensor << std::endl;

    std::cout << "\nCreate tensor2 from Matrix..." << std::endl;
    Tensor tensor2(Matrix::Zero(10, 10));
    std::cout << tensor2 << std::endl;

    std::cout << "\nUpdate tensor2..." << std::endl;
    tensor2.data() = Matrix::Ones(10, 10);
    std::cout << tensor2 << std::endl;

    std::cout << "\nMultiply tensor2 by 2..." << std::endl;
    tensor2 *= 2;
    std::cout << tensor2 << std::endl;

    std::cout << "\nAdd tensor2 and tensor..." << std::endl;
    Tensor result = tensor2 + tensor;
    std::cout << result << std::endl;

    std::cout << "\nHadamard (element-wise) multiply tensor2 and tensor..." << std::endl;
    Tensor result2 = tensor2.hadamard(tensor);
    std::cout << result2 << std::endl;

    std::cout << "\nDivide tensor2 by 2..." << std::endl;
    tensor2 /= 2;
    std::cout << tensor2 << std::endl;

    std::cout << "\n=== Testing New Tensor Operations ===" << std::endl;

    // Test matrix multiplication
    std::cout << "\nTesting matrix multiplication..." << std::endl;
    Matrix a(2, 3);
    a << 1, 2, 3, 4, 5, 6;
    Matrix b(3, 2);
    b << 1, 2, 3, 4, 5, 6;

    Tensor ta(a);
    Tensor tb(b);

    std::cout << "Matrix A:" << std::endl;
    std::cout << ta << std::endl;
    std::cout << "Matrix B:" << std::endl;
    std::cout << tb << std::endl;

    Tensor matmul_result = ta * tb;
    std::cout << "A * B (matrix multiplication):" << std::endl;
    std::cout << matmul_result << std::endl;

    // Test transpose
    std::cout << "\nTranspose of A:" << std::endl;
    Tensor transposed = ta.transpose();
    std::cout << transposed << std::endl;

    // Test broadcasting
    std::cout << "\nTesting broadcasting (bias addition)..." << std::endl;
    Matrix data(2, 3);
    data << 1, 2, 3, 4, 5, 6;
    Matrix bias(1, 3);
    bias << 10, 20, 30;

    Tensor t_data(data);
    Tensor t_bias(bias);

    std::cout << "Data:" << std::endl;
    std::cout << t_data << std::endl;
    std::cout << "Bias:" << std::endl;
    std::cout << t_bias << std::endl;

    Tensor broadcast_result = t_data.broadcast_add(t_bias);
    std::cout << "Data + Bias (broadcast):" << std::endl;
    std::cout << broadcast_result << std::endl;

    std::cout << "\n=== Neural Network Layer Testing ===" << std::endl;
    std::cout << "\nCreate dense layer..." << std::endl;
    DenseLayer layer(4, 3);  // 4 inputs, 3 outputs

    std::cout << "\nTesting forward pass..." << std::endl;
    // Create input tensor (1 sample, 4 features)
    Matrix inputMatrix(1, 4);
    inputMatrix << 1.0, 2.0, 3.0, 4.0;
    Tensor input(inputMatrix);

    std::cout << "Input:" << std::endl;
    std::cout << input << std::endl;

    // Forward pass
    Tensor output = layer.forward(input);
    std::cout << "Output after forward pass:" << std::endl;
    std::cout << output << std::endl;

    // Test with batch of inputs (2 samples, 4 features each)
    std::cout << "\nTesting batch forward pass..." << std::endl;
    Matrix batchMatrix(2, 4);
    batchMatrix << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0;
    Tensor batchInput(batchMatrix);

    std::cout << "Batch input:" << std::endl;
    std::cout << batchInput << std::endl;

    Tensor batchOutput = layer.forward(batchInput);
    std::cout << "Batch output after forward pass:" << std::endl;
    std::cout << batchOutput << std::endl;

    return 0;
}
