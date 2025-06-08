#include <iostream>
#include "neuron.hpp"

void Neuron::setWeight(int w) {
    weight=w;
};

void Neuron::setBias(int b) {
    bias=b;
};

void Neuron::printNeuron() {
    std::cout << "Neuron:" << std::endl;
    std::cout << "\tweight = " << weight << std::endl;
    std::cout << "\tbias = " << bias << std::endl;
}
