#include <iostream>
#include <iomanip>
#include <vector>

#include "tools/binaryconversion.hpp"
#include "neuralnet.h"

int main(int argc, char *argv[])
{
    if (argv[1]==NULL || argv[2]==NULL || argv[3]==NULL)
    {
        std::cout << "Error: Missing arguments!" << std::endl;
        std::cout << "Syntax: ./NeuralNetIntAddition [eta] [tss] [hls]" << std::endl;
        std::cout << "   eta: The learning rate" << std::endl;
        std::cout << "   tss: The training set size" << std::endl;
        std::cout << "   hls: The hidden layer size" << std::endl;

        exit(1);
    }

    double eta = atof(argv[1]);
    int tss = atoi(argv[2]);
    int hls = atoi(argv[3]);

    std::cout << "eta: " << eta << " tss: " << tss  << " hls: " << hls << std::endl;

    std::vector<double> input;
    std::vector<double> desired;
    std::vector<double> output;

    RandomInt irandgen; // Train with a set of 10000
    std::vector<int> irand(tss);
    irandgen.FillVector(irand,0,100000);
    irandgen.Clear();

    NeuralNetwork nn(32,hls,32,eta);

    int ep=0;
    //int i=100;
    while (ep<5000)
    {
        for (auto&& i : irand)
        {
            //Begin Neural Network Computation
            input = ProduceBinaryVector(i);
            desired = ProduceBinaryVector(i+1);

            nn.NewTrainingData(input,desired);
            nn.ComputeLayers();
            nn.ComputeDerivatives();
            nn.ResetForNewTrainingData();
        }

        nn.CompleteTrainingSet();
        ++ep;
    }

    input = ProduceBinaryVector(1000);
    desired = ProduceBinaryVector(1001);

    nn.NewTrainingData(input,desired);
    nn.ComputeLayers();

    nn.GetOutput(output);

    std::cout << "Input:  ";
    for (auto&& op : input)
        std::cout << " " << round(op);
    std::cout << std::endl;

    std::cout << "Desire: ";
    for (auto&& op : desired)
        std::cout << " " << round(op);
    std::cout << std::endl;

    std::cout << "Output: ";
    for (auto&& op : output)
        std::cout << " " << round(op);
    std::cout << std::endl;

    std::cout << "Output: ";
    for (auto&& op : output)
        std::cout << std::setprecision(5) << " " << op;
    std::cout << std::endl;

    int value = ProduceIntegerFromBinary(output);
    std::cout << value << std::endl;

    nn.Clear();

    return 0;
};
