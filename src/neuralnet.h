#ifndef NEURAL_NET_C
#define NEURAL_NET_C

#include <vector>
#include <cstring>

#include "tools/random.hpp"
#include "neuralnetlayer.hpp"

class NeuralNetwork
{
    std::vector<double> ila; // Input Layer Activation
    std::vector<double> ola; // Output Layer Activation
    std::vector<double> de; // Desired outcome

    double eta;
    double avgCost;
    long int cntr;

    NeuralNetLayer nnl1; // Hidden layer 1
    //NeuralNetLayer nnl2; // Hidden layer 2
    NeuralNetLayer nno; // The output layer

    NeuralNetwork() {};

public:

    //NeuralNetwork(std::vector<float> &ila,std::vector<float> &de,float eta)
    NeuralNetwork(int iptN,int midN,int outN,float eta)
    {
        //this->ila=ila;
        //this->de=de;
        this->eta=eta;

        nnl1.Init(iptN,midN,"Hidden1bgraph.dat");
        //nnl2.Init(midN,midN);
        nno.Init(midN,outN,"Output1bgraph.dat");

        cntr=0;
        avgCost=0;
    };

    void NewTrainingData(std::vector<double> &ila,std::vector<double> &de)
    {
        this->ila=ila;
        this->de=de;
    };

    void ComputeLayers()
    {
        std::cout << "\n Computing Activation\n";
        std::cout << "  Compute Activation Hidden Layer 1\n";
        nnl1.ComputeActivation(ila);
        //nnl2.ComputeActivation(nnl1.GetActivation());
        std::cout << "  Compute Activation of Output layer\n";
        nno.ComputeActivation(nnl1.GetActivation());// Getting Nanners????!!!!

        ola=nno.GetActivation();

        avgCost+=CalculateCost();
        ++cntr;
    };

    void ComputeDerivatives()
    {
        // Backpropagate
        std::cout << "\n Backpropagation\n";
        std::cout << "  Output Errors\n";
        nno.ComputeInitalError(de);
        //nnl2.ComputeError(nno);
        std::cout << "  Hiddle Layer 1 Errors\n";
        nnl1.ComputeError(nno);

        // Calculate Derivatives
        std::cout << "\n Compute Derivatives\n";
        std::cout << "  Hidden Layer 1 Derivatives\n";
        nnl1.ComputeDerivatives(ila);
        //nnl2.ComputeDerivatives(nnl1.GetActivation());
        std::cout << "  Output Layer Derivatives\n";
        nno.ComputeDerivatives(nnl1.GetActivation());
    };

    double CalculateCost()
    {
        float cost = 0.0;

        for (int i=0;i<int(ola.size());++i)
        {
            float diff = de[i] - nno.GetActivation()[i];
            cost+=diff*diff;
        };

        return 0.5 * cost;
    };

    void GetOutput(std::vector<double> &ola)
    {
        if (!ola.empty())
            ola.clear();

        ola=this->ola;
    };

    void ResetForNewTrainingData()
    {
        std::cout << "\n Resetting for new training data\n";

        this->ila.clear();
        this->de.clear();
        this->ola.clear();
    };

    void CompleteTrainingSet()
    {
        std::cout << "\n Training Epoch\n";
        std::cout << "  avgCost of training set: " << avgCost/double(cntr) << std::endl;
        //ResetForNewTrainingData();

        nnl1.EndTrainingSet(eta);
        //nnl2.EndTrainingSet(eta);
        nno.EndTrainingSet(eta);

        cntr=0;
        avgCost=0.0;
    };

    void Clear()
    {
        nnl1.Clear();
        //nnl2.Clear();
        nno.Clear();
        ila.clear();
        ola.clear();
    };
};

#endif
