//
// Created by Bartosz Roguski on 07/06/2025.
//
#include "src/core/nn/tensor.hpp"
#include "src/core/nn/layer.hpp"
#include <random>

int main() {
    printf("\nCreate tensor...\n");
    Tensor tensor(10, 10, 10);
    tensor.printTensor();

    printf("\nUpdate tensor...\n");
    tensor.updateTensor(Matrix::Ones(10, 10));
    tensor.printTensor();

    printf("\nCreate tensor2 from Matrix...\n");
    Tensor tensor2(Matrix::Zero(10, 10));
    tensor2.printTensor();

    printf("\nUpdate tensor2...\n");
    tensor2.updateTensor(Matrix::Ones(10, 10));
    tensor2.printTensor();

    printf("\nMultiply tensor2 by 2...\n");
    tensor2 *= 2;
    tensor2.printTensor();

    printf("\nAdd tensor2 and tensor...\n");
    Tensor result = tensor2 + tensor;
    result.printTensor();

    printf("\nHadamard (element-wise) multiply tensor2 and tensor...\n");
    Tensor result2 = tensor2.hadamard(tensor);
    result2.printTensor();

    printf("\nDivide tensor2 by 2...\n");
    tensor2 /= 2;
    tensor2.printTensor();

    printf("\n=== Testing New Tensor Operations ===\n");
    
    // Test matrix multiplication
    printf("\nTesting matrix multiplication...\n");
    Matrix a(2, 3);
    a << 1, 2, 3,
         4, 5, 6;
    Matrix b(3, 2);
    b << 1, 2,
         3, 4,
         5, 6;
    
    Tensor ta(a);
    Tensor tb(b);
    
    printf("Matrix A:\n");
    ta.printTensor();
    printf("Matrix B:\n");
    tb.printTensor();
    
    Tensor matmul_result = ta.matmul(tb);
    printf("A * B (matrix multiplication):\n");
    matmul_result.printTensor();
    
    // Test transpose
    printf("\nTranspose of A:\n");
    Tensor transposed = ta.transpose();
    transposed.printTensor();
    
    // Test broadcasting
    printf("\nTesting broadcasting (bias addition)...\n");
    Matrix data(2, 3);
    data << 1, 2, 3,
            4, 5, 6;
    Matrix bias(1, 3);
    bias << 10, 20, 30;
    
    Tensor t_data(data);
    Tensor t_bias(bias);
    
    printf("Data:\n");
    t_data.printTensor();
    printf("Bias:\n");
    t_bias.printTensor();
    
    Tensor broadcast_result = t_data.broadcast_add(t_bias);
    printf("Data + Bias (broadcast):\n");
    broadcast_result.printTensor();

    printf("\n=== Neural Network Layer Testing ===\n");
    printf("\nCreate layer...\n");
    Layer layer(3, 4);  // 3 neurons, 4 inputs
    layer.printLayer();

    printf("\nTesting forward pass...\n");
    // Create input tensor (1 sample, 4 features)
    Matrix inputMatrix(1, 4);
    inputMatrix << 1.0, 2.0, 3.0, 4.0;
    Tensor input(inputMatrix);
    
    printf("Input:\n");
    input.printTensor();
    
    // Forward pass
    Tensor output = layer.forward(input);
    printf("Output after forward pass:\n");
    output.printTensor();

    // Test with batch of inputs (2 samples, 4 features each)
    printf("\nTesting batch forward pass...\n");
    Matrix batchMatrix(2, 4);
    batchMatrix << 1.0, 2.0, 3.0, 4.0,
                   5.0, 6.0, 7.0, 8.0;
    Tensor batchInput(batchMatrix);
    
    printf("Batch input:\n");
    batchInput.printTensor();
    
    Tensor batchOutput = layer.forward(batchInput);
    printf("Batch output after forward pass:\n");
    batchOutput.printTensor();

    return 0;
}
