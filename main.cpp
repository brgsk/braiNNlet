#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "src/core/nn/network.hpp"
#include "src/core/nn/tensor.hpp"

void printSeparator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void printSubSection(const std::string& title) {
    std::cout << "\n--- " << title << " ---" << std::endl;
}

// Generate synthetic data for regression: y = 0.5*x1 + 0.3*x2 + 0.1*x3 + 0.2
std::pair<std::vector<Matrix>, std::vector<Matrix>> generateRegressionData(int samples) {
    std::vector<Matrix> inputs;
    std::vector<Matrix> targets;

    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dis(-2.0, 2.0);

    for (int i = 0; i < samples; ++i) {
        Matrix input(1, 3);
        input << dis(gen), dis(gen), dis(gen);

        // Target: linear combination with some noise
        double target = 0.5 * input(0, 0) + 0.3 * input(0, 1) + 0.1 * input(0, 2) + 0.2;
        target += 0.05 * dis(gen);  // Add small noise

        Matrix target_matrix(1, 1);
        target_matrix << target;

        inputs.push_back(input);
        targets.push_back(target_matrix);
    }

    return {inputs, targets};
}

// Generate synthetic data for binary classification: XOR-like problem
std::pair<std::vector<Matrix>, std::vector<Matrix>> generateClassificationData(int samples) {
    std::vector<Matrix> inputs;
    std::vector<Matrix> targets;

    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    for (int i = 0; i < samples; ++i) {
        Matrix input(1, 2);
        input << dis(gen), dis(gen);

        // XOR-like target: positive if x1*x2 > 0, negative otherwise
        double target = (input(0, 0) * input(0, 1) > 0) ? 1.0 : 0.0;

        Matrix target_matrix(1, 1);
        target_matrix << target;

        inputs.push_back(input);
        targets.push_back(target_matrix);
    }

    return {inputs, targets};
}

void demonstrateBasicTensorOperations() {
    printSeparator("BASIC TENSOR OPERATIONS");

    printSubSection("Matrix Creation and Operations");
    Matrix a(2, 3);
    a << 1, 2, 3, 4, 5, 6;
    Matrix b(3, 2);
    b << 1, 2, 3, 4, 5, 6;

    Tensor ta(a);
    Tensor tb(b);

    std::cout << "Matrix A (2x3):\n" << ta << std::endl;
    std::cout << "\nMatrix B (3x2):\n" << tb << std::endl;

    Tensor matmul = ta * tb;
    std::cout << "\nA * B (matrix multiplication):\n" << matmul << std::endl;

    printSubSection("Broadcasting Operations");
    Matrix data(2, 3);
    data << 1, 2, 3, 4, 5, 6;
    Matrix bias(1, 3);
    bias << 10, 20, 30;

    Tensor t_data(data);
    Tensor t_bias(bias);

    std::cout << "Data:\n" << t_data << std::endl;
    std::cout << "\nBias:\n" << t_bias << std::endl;

    Tensor broadcast_result = t_data.broadcast_add(t_bias);
    std::cout << "\nData + Bias (broadcast):\n" << broadcast_result << std::endl;
}

void demonstrateNetworkBuilding() {
    printSeparator("NETWORK BUILDING AND INSPECTION");

    printSubSection("Creating a Multi-Layer Network");
    Network network;

    // Build a network: 3 -> 8 -> 4 -> 1
    network.add_layer(8, ActivationType::ReLU);    // Hidden layer 1
    network.add_layer(4, ActivationType::Tanh);    // Hidden layer 2
    network.add_layer(1, ActivationType::Linear);  // Output layer

    network.set_loss_function(LossType::MeanSquaredError);

    std::cout << network.summary() << std::endl;

    printSubSection("Network Validation");
    std::cout << "Network is valid: " << (network.is_valid() ? "Yes" : "No") << std::endl;
    if (!network.is_valid()) {
        std::cout << "Validation error: " << network.validation_error() << std::endl;
    }

    printSubSection("Testing Forward Pass");
    Matrix test_input(1, 3);
    test_input << 1.0, 2.0, 3.0;
    Tensor input(test_input);

    std::cout << "Input:\n" << input << std::endl;

    Tensor output = network.forward(input);
    std::cout << "\nNetwork output:\n" << output << std::endl;

    // Test with batch
    printSubSection("Batch Processing");
    Matrix batch_input(2, 3);
    batch_input << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    Tensor batch(batch_input);

    std::cout << "Batch input:\n" << batch << std::endl;

    Tensor batch_output = network.forward(batch);
    std::cout << "\nBatch output:\n" << batch_output << std::endl;
}

void demonstrateRegressionTraining() {
    printSeparator("REGRESSION TRAINING DEMONSTRATION");

    printSubSection("Generating Training Data");
    auto [train_inputs, train_targets] = generateRegressionData(100);
    auto [test_inputs, test_targets] = generateRegressionData(20);

    std::cout << "Training samples: " << train_inputs.size() << std::endl;
    std::cout << "Test samples: " << test_inputs.size() << std::endl;

    // Show a few samples
    std::cout << "\nFirst 3 training samples:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        std::cout << "Input: [" << std::fixed << std::setprecision(3) << train_inputs[i](0, 0)
                  << ", " << train_inputs[i](0, 1) << ", " << train_inputs[i](0, 2)
                  << "] -> Target: " << train_targets[i](0, 0) << std::endl;
    }

    printSubSection("Building and Training Network");
    Network network;
    network.add_layer(8, ActivationType::ReLU);
    network.add_layer(4, ActivationType::ReLU);
    network.add_layer(1, ActivationType::Linear);
    network.set_loss_function(LossType::MeanSquaredError);

    std::cout << "\n" << network.summary() << std::endl;

    double learning_rate = 0.01;
    int epochs = 50;

    std::cout << "Training parameters:" << std::endl;
    std::cout << "  Learning rate: " << learning_rate << std::endl;
    std::cout << "  Epochs: " << epochs << std::endl;

    // Training loop
    std::vector<double> losses;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;

        for (size_t i = 0; i < train_inputs.size(); ++i) {
            // Forward pass
            Tensor input(train_inputs[i]);
            Tensor target(train_targets[i]);
            Tensor output = network.forward(input);

            // Compute loss
            double loss = network.compute_loss(output, target);
            total_loss += loss;

            // Backward pass
            Tensor loss_grad = network.compute_loss_gradient(output, target);
            network.backward(loss_grad);

            // Update weights
            network.update_weights(learning_rate);
            network.zero_gradients();
        }

        double avg_loss = total_loss / train_inputs.size();
        losses.push_back(avg_loss);

        if (epoch % 10 == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << std::setw(3) << epoch << " - Loss: " << std::fixed
                      << std::setprecision(6) << avg_loss << std::endl;
        }
    }

    printSubSection("Evaluating on Test Set");
    double test_loss = 0.0;
    std::cout << "\nTest predictions (first 5 samples):" << std::endl;
    std::cout << "Input -> Predicted | Target | Error" << std::endl;

    for (size_t i = 0; i < std::min(size_t(5), test_inputs.size()); ++i) {
        Tensor input(test_inputs[i]);
        Tensor target(test_targets[i]);
        Tensor prediction = network.forward(input);

        double pred_val = prediction(0, 0);
        double target_val = target(0, 0);
        double error = std::abs(pred_val - target_val);

        test_loss += network.compute_loss(prediction, target);

        std::cout << "[" << std::fixed << std::setprecision(2) << test_inputs[i](0, 0) << ","
                  << test_inputs[i](0, 1) << "," << test_inputs[i](0, 2) << "] -> "
                  << std::setprecision(3) << pred_val << " | " << target_val << " | " << error
                  << std::endl;
    }

    std::cout << "\nTest Loss: " << std::fixed << std::setprecision(6)
              << test_loss / test_inputs.size() << std::endl;
}

void demonstrateClassificationTraining() {
    printSeparator("BINARY CLASSIFICATION DEMONSTRATION");

    printSubSection("Generating Classification Data");
    auto [train_inputs, train_targets] = generateClassificationData(200);
    auto [test_inputs, test_targets] = generateClassificationData(50);

    std::cout << "Training samples: " << train_inputs.size() << std::endl;
    std::cout << "Test samples: " << test_inputs.size() << std::endl;

    // Show class distribution
    int class_0 = 0, class_1 = 0;
    for (const auto& target : train_targets) {
        if (target(0, 0) > 0.5)
            class_1++;
        else
            class_0++;
    }
    std::cout << "Class distribution - Class 0: " << class_0 << ", Class 1: " << class_1
              << std::endl;

    printSubSection("Building Classification Network");
    Network network;
    network.add_layer(6, ActivationType::ReLU);
    network.add_layer(4, ActivationType::ReLU);
    network.add_layer(1, ActivationType::Sigmoid);
    network.set_loss_function(LossType::BinaryCrossEntropy);

    std::cout << "\n" << network.summary() << std::endl;

    double learning_rate = 0.1;
    int epochs = 100;

    std::cout << "Training parameters:" << std::endl;
    std::cout << "  Learning rate: " << learning_rate << std::endl;
    std::cout << "  Epochs: " << epochs << std::endl;

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;

        for (size_t i = 0; i < train_inputs.size(); ++i) {
            Tensor input(train_inputs[i]);
            Tensor target(train_targets[i]);
            Tensor output = network.forward(input);

            double loss = network.compute_loss(output, target);
            total_loss += loss;

            Tensor loss_grad = network.compute_loss_gradient(output, target);
            network.backward(loss_grad);
            network.update_weights(learning_rate);
            network.zero_gradients();
        }

        if (epoch % 20 == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << std::setw(3) << epoch << " - Loss: " << std::fixed
                      << std::setprecision(6) << total_loss / train_inputs.size() << std::endl;
        }
    }

    printSubSection("Classification Results");
    int correct = 0;
    std::cout << "\nTest predictions (first 10 samples):" << std::endl;
    std::cout << "Input -> Prob | Pred | Target | Correct" << std::endl;

    for (size_t i = 0; i < std::min(size_t(10), test_inputs.size()); ++i) {
        Tensor input(test_inputs[i]);
        Tensor target(test_targets[i]);
        Tensor prediction = network.forward(input);

        double prob = prediction(0, 0);
        int pred_class = prob > 0.5 ? 1 : 0;
        int target_class = target(0, 0) > 0.5 ? 1 : 0;
        bool is_correct = pred_class == target_class;

        if (is_correct)
            correct++;

        std::cout << "[" << std::fixed << std::setprecision(2) << test_inputs[i](0, 0) << ","
                  << test_inputs[i](0, 1) << "] -> " << std::setprecision(3) << prob << " | "
                  << pred_class << " | " << target_class << " | " << (is_correct ? "✓" : "✗")
                  << std::endl;
    }

    // Calculate full test accuracy
    correct = 0;
    for (size_t i = 0; i < test_inputs.size(); ++i) {
        Tensor input(test_inputs[i]);
        Tensor target(test_targets[i]);
        Tensor prediction = network.forward(input);

        int pred_class = prediction(0, 0) > 0.5 ? 1 : 0;
        int target_class = target(0, 0) > 0.5 ? 1 : 0;
        if (pred_class == target_class)
            correct++;
    }

    double accuracy = double(correct) / test_inputs.size() * 100.0;
    std::cout << "\nTest Accuracy: " << std::fixed << std::setprecision(1) << accuracy << "% ("
              << correct << "/" << test_inputs.size() << ")" << std::endl;
}

void demonstrateNetworkManipulation() {
    printSeparator("NETWORK MANIPULATION AND UTILITIES");

    printSubSection("Dynamic Network Building");
    Network network;

    std::cout << "Initial network:" << std::endl;
    std::cout << network.summary() << std::endl;

    std::cout << "\nAdding layers dynamically:" << std::endl;
    network.add_layer(4, ActivationType::ReLU);
    std::cout << "Added layer 1: " << network.summary() << std::endl;

    network.add_layer(3, ActivationType::Tanh);
    std::cout << "Added layer 2: " << network.summary() << std::endl;

    network.add_layer(1, ActivationType::Linear);
    std::cout << "Added layer 3: " << network.summary() << std::endl;

    printSubSection("Layer Access and Inspection");
    std::cout << "Network has " << network.layer_count() << " layers" << std::endl;

    for (int i = 0; i < network.layer_count(); ++i) {
        const Layer& layer = network.layer(i);
        std::cout << "Layer " << i << ": " << layer.name() << " (" << layer.input_size() << " -> "
                  << layer.output_size() << ")"
                  << " - " << layer.parameter_count() << " parameters" << std::endl;
    }

    printSubSection("Training Mode Control");
    std::cout << "Network is in training mode: " << (network.is_training() ? "Yes" : "No")
              << std::endl;

    network.set_training(false);
    std::cout << "Set to inference mode..." << std::endl;
    std::cout << "Network is in training mode: " << (network.is_training() ? "Yes" : "No")
              << std::endl;

    network.set_training(true);
    std::cout << "Set back to training mode..." << std::endl;
    std::cout << "Network is in training mode: " << (network.is_training() ? "Yes" : "No")
              << std::endl;
}

int main() {
    std::cout << "\n🧠 braiNNlet Neural Network Library - Comprehensive Demo" << std::endl;
    std::cout << "========================================================" << std::endl;

    try {
        demonstrateBasicTensorOperations();
        demonstrateNetworkBuilding();
        demonstrateRegressionTraining();
        demonstrateClassificationTraining();
        demonstrateNetworkManipulation();

        printSeparator("DEMO COMPLETED SUCCESSFULLY");
        std::cout << "🎉 All demonstrations completed successfully!" << std::endl;
        std::cout << "The braiNNlet library is ready for use in your projects." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\n❌ Error during demonstration: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
