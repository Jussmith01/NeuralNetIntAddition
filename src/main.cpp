#include <iostream>
#include <vector>

#include "tools/binaryconversion.hpp"
#include "neuralnet.h"

int main(int argc, char *argv[])
{
    int value;
    std::vector<float> input = ProduceBinaryVector(2);
    std::vector<float> output;

    //Begin Neural Network Computation
    NeuralNetwork nn(input);
    nn.ComputeLayers();
    nn.GetOutput(output);

    value = ProduceIntegerFromBinary(output);
    std::cout << value << " COST: " << nn.CalculateCost(input) << std::endl;

    input = ProduceBinaryVector(-34585646);
    nn.NewTrainingData(input);
    nn.ComputeLayers();
    nn.GetOutput(output);

    value = ProduceIntegerFromBinary(output);
    std::cout << value << " COST: " << nn.CalculateCost(input) << std::endl;

    input = ProduceBinaryVector(2);
    nn.NewTrainingData(input);
    nn.ComputeLayers();
    nn.GetOutput(output);

    value = ProduceIntegerFromBinary(output);
    std::cout << value << " COST: " << nn.CalculateCost(input) << std::endl;

    nn.Clear();

    return 0;
};
