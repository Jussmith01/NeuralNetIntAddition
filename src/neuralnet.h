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

    double prevcost;

    double eta;
    double avgCost;
    long int cntr;

    NeuralNetLayer nnl1; // Hidden layer 1
    NeuralNetLayer nnl2; // Hidden layer 2
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
        nnl2.Init(midN,midN,"Hidden2bgraph.dat");
        nno.Init(midN,outN,"Output1bgraph.dat");

        cntr=0;
        avgCost=0;
        prevcost=1000.0;
    };

    void NewTrainingData(std::vector<double> &ila,std::vector<double> &de)
    {
        this->ila=ila;
        this->de=de;
    };

    void ComputeLayers()
    {
        nnl1.ComputeActivation(ila);
        nnl2.ComputeActivation(nnl1.GetActivation());
        nno.ComputeActivation(nnl2.GetActivation());// Getting Nanners????!!!!

        ola=nno.GetActivation();

        avgCost+=CalculateCost();
        ++cntr;
    };

    void ComputeDerivatives()
    {
        // Backpropagate
        nno.ComputeInitalError(de);
        nnl2.ComputeError(nno);
        nnl1.ComputeError(nnl2);

        // Calculate Derivatives
        nnl1.ComputeDerivatives(ila);
        nnl2.ComputeDerivatives(nnl1.GetActivation());
        nno.ComputeDerivatives(nnl2.GetActivation());
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

        this->ila.clear();
        this->de.clear();
        this->ola.clear();
    };

    double CompleteTrainingSet()
    {
        double currentcost = avgCost/double(cntr);
        std::cout << "avgCost of training set: " << currentcost << " ";
        //ResetForNewTrainingData();

        if (prevcost-currentcost < 0.0)
        {
            Halfeta();
            std::cout << "eta now: " << eta;
        }

        std::cout << std::endl;
        prevcost=currentcost;

        nnl1.EndTrainingSet(eta);
        nnl2.EndTrainingSet(eta);
        nno.EndTrainingSet(eta);

        cntr=0;
        avgCost=0.0;

        return currentcost;
    };

    void Clear()
    {
        nnl1.Clear();
        //nnl2.Clear();
        nno.Clear();
        ila.clear();
        ola.clear();
    };

    void Halfeta()
    {
        eta=0.95*eta;
    }
};

#endif
