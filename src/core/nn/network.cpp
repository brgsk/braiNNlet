#include "network.hpp"

#include <sstream>
#include <stdexcept>

Network::Network() : loss_function_(create_loss(LossType::MeanSquaredError)) {}

void Network::add_layer(int neurons, ActivationType activation) {
    add_layer(LayerConfig(neurons, activation));
}

void Network::add_layer(const LayerConfig& config) {
    if (config.neurons <= 0) {
        throw std::invalid_argument("Layer must have at least 1 neuron");
    }
    int input_size = 0;
    if (!layers_.empty()) {
        input_size = layers_.back()->output_size();
    } else {
        // First layer - input size will be determined at runtime
        // For now, we'll defer creation until we know the input size
        input_size = config.neurons;  // This will be corrected later
    }

    auto layer = std::make_unique<DenseLayer>(input_size, config.neurons, config.activation);
    layers_.push_back(std::move(layer));
};

void Network::remove_layer(int index) {
    if (layers_.empty()) {
        throw std::runtime_error("Cannot remove layer from empty network");
    }

    if (index == -1) {
        index = layers_.size() - 1;
    }

    if (index < 0 || index >= layers_.size()) {
        throw std::invalid_argument("Layer index out of range");
    }

    layers_.erase(layers_.begin() + index);
}

void Network::clear() {
    layers_.clear();
}

Tensor Network::forward(const Tensor& input) {
    if (layers_.empty()) {
        throw std::runtime_error("Cannot forward through empty network");
    }

    // Handle first layer input size adjustment
    if (layers_[0]->input_size() != input.cols()) {
        // Recreate first layer with correct input size
        auto* dense_layer = static_cast<DenseLayer*>(layers_[0].get());
        int output_size = dense_layer->output_size();
        ActivationType activation = dense_layer->activation_type();

        layers_[0] = std::make_unique<DenseLayer>(input.cols(), output_size, activation);
    }

    Tensor current_output = input;
    for (auto& layer : layers_) {
        current_output = layer->forward(current_output);
    }

    return current_output;
}

Tensor Network::backward(const Tensor& loss_gradient) {
    if (layers_.empty()) {
        throw std::runtime_error("Cannot backward through empty network");
    }

    Tensor current_gradient = loss_gradient;

    // Backward pass through layers in reverse order
    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
        current_gradient = layers_[i]->backward(current_gradient);
    }

    return current_gradient;
}

void Network::set_loss_function(LossType loss_type) {
    loss_function_ = create_loss(loss_type);
}

double Network::compute_loss(const Tensor& predictions, const Tensor& targets) {
    if (!loss_function_) {
        throw std::runtime_error("No loss function set");
    }
    return loss_function_->forward(predictions, targets);
}

Tensor Network::compute_loss_gradient(const Tensor& predictions, const Tensor& targets) {
    if (!loss_function_) {
        throw std::runtime_error("No loss function set");
    }
    return loss_function_->backward(predictions, targets);
}

void Network::update_weights(double learning_rate) {
    for (auto& layer : layers_) {
        layer->update_weights(learning_rate);
    }
}

void Network::zero_gradients() {
    for (auto& layer : layers_) {
        layer->zero_gradients();
    }
}

int Network::parameter_count() const {
    int total = 0;
    for (const auto& layer : layers_) {
        total += layer->parameter_count();
    }
    return total;
}

std::string Network::summary() const {
    std::stringstream ss;
    ss << "Network Summary:\n";
    ss << "================\n";

    if (layers_.empty()) {
        ss << "Empty network\n";
        return ss.str();
    }

    for (size_t i = 0; i < layers_.size(); ++i) {
        const auto& layer = layers_[i];
        ss << "Layer " << i << ": " << layer->name() << "\n";
        ss << "  Parameters: " << layer->parameter_count() << "\n";
    }

    ss << "================\n";
    ss << "Total parameters: " << parameter_count() << "\n";
    ss << "Loss function: " << (loss_function_ ? loss_function_->name() : "None") << "\n";

    return ss.str();
}

const Layer& Network::layer(int index) const {
    if (index < 0 || index >= static_cast<int>(layers_.size())) {
        throw std::out_of_range("Layer index out of range");
    }
    return *layers_[index];
}

Layer& Network::layer(int index) {
    if (index < 0 || index >= static_cast<int>(layers_.size())) {
        throw std::out_of_range("Layer index out of range");
    }
    return *layers_[index];
}

bool Network::is_valid() const {
    try {
        validate_network();
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

std::string Network::validation_error() const {
    try {
        validate_network();
        return "";
    } catch (const std::exception& e) {
        return e.what();
    }
}

void Network::set_training(bool training) {
    training_ = training;
    for (auto& layer : layers_) {
        layer->set_training(training);
    }
}

bool Network::is_training() const {
    return training_;
}

void Network::validate_network() const {
    if (layers_.empty()) {
        throw std::runtime_error("Network has no layers");
    }

    // Check layer connectivity
    for (size_t i = 1; i < layers_.size(); ++i) {
        int prev_output = layers_[i - 1]->output_size();
        int curr_input = layers_[i]->input_size();

        if (prev_output != curr_input) {
            throw std::runtime_error("Layer " + std::to_string(i - 1) + " output size (" +
                                     std::to_string(prev_output) + ") doesn't match layer " +
                                     std::to_string(i) + " input size (" +
                                     std::to_string(curr_input) + ")");
        }
    }

    if (!loss_function_) {
        throw std::runtime_error("No loss function set");
    }
}