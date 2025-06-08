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

    printf("\nMultiply tensor2 and tensor...\n");
    Tensor result2 = tensor2 * tensor;
    result2.printTensor();

    printf("\nDivide tensor2 by 2...\n");
    tensor2 /= 2;
    tensor2.printTensor();

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
