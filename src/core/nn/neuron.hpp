#pragma once

#include <iostream>

class Neuron {
public:
    int weight;
    int bias;

    void setWeight(int weight);
    void setBias(int bias);

    void printNeuron();

    // Constructor and destructor
    Neuron() {
        weight=0;
        bias=0;
    };

    virtual ~Neuron() {
        std::cout << "Neuron destructor" << std::endl;
    };
};