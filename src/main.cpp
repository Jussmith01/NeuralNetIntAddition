#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

#include "tools/binaryconversion.hpp"
#include "neuralnet.h"

int main(int argc, char *argv[])
{
    if (argv[1]==NULL || argv[2]==NULL || argv[3]==NULL || argv[4]==NULL || argv[5]==NULL)
    {
        std::cout << "Error: Missing arguments!" << std::endl;
        std::cout << "Syntax: ./NeuralNetIntAddition [eta] [tss] [hls]" << std::endl;
        std::cout << "   eta: The learning rate" << std::endl;
        std::cout << "   tss: The Training Set Size" << std::endl;
        std::cout << "   ess: The tEsting Set Size" << std::endl;
        std::cout << "   hls: The hidden layer size" << std::endl;
        std::cout << "   con : Convergence of average cost" << std::endl;

        exit(1);
    }

    double eta = atof(argv[1]);
    int tss = atoi(argv[2]);
    int ess = atoi(argv[3]);
    int hls = atoi(argv[4]);
    double con = atof(argv[5]);

    std::cout << "eta: " << eta << " tss: " << tss  << " ess: " << ess << " hls: " << hls << " con: " << con << std::endl;


    std::vector<double> input;
    std::vector<double> desired;
    std::vector<double> output;

    RandomInt irandgen; // Train with a set of 10000
    std::vector<int> irand(tss);
    irandgen.FillVector(irand,-1000000,1000000);
    irandgen.Clear();

    NeuralNetwork nn(32,hls,32,eta);

    int ep=0;
    double avgcost = 100.0;

    while (avgcost>con)
    {
        //std::cout << "\n |---------STARING EPOCH " << ep << "----------|\n";
        //std::cout << "\n Randomizing Training Data...\n";
        std::random_shuffle(irand.begin(),irand.end());

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

        std::cout << " Epoch " << ep << " - ";
        avgcost = nn.CompleteTrainingSet();

        ++ep;
    }

    irand.clear();

    std::cout << " |------Testing Set-------|\n";
    RandomInt itestrandgen; // Train with a set of 10000
    std::vector<int> irandtest(ess);
    itestrandgen.FillVector(irandtest,-1000000,1000000);
    itestrandgen.Clear();

    int correct = 0;
    for (int i=0;i<ess;++i)
    {
        input = ProduceBinaryVector(irandtest[i]);
        desired = ProduceBinaryVector(irandtest[i]+1);

        nn.NewTrainingData(input,desired);
        nn.ComputeLayers();

        nn.GetOutput(output);
        int value = ProduceIntegerFromBinary(output);

        if (value == irandtest[i]+1)
            ++correct;
    }

    std::cout << "Accuracy:" << correct/double(ess) << std::endl;

    /*std::cout << "Input:  ";
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
    std::cout << std::endl;*/

    nn.Clear();

    return 0;
};
