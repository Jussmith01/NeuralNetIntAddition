#ifndef NEURAL_NET_C
#define NEURAL_NET_C

#include <vector>
#include <cstring>
#include <sstream>

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

    std::vector<NeuralNetLayer> nnl; // Neural network hidden layers

    NeuralNetLayer nno; // The output layer

    NeuralNetwork() {};

public:

    //NeuralNetwork(std::vector<float> &ila,std::vector<float> &de,float eta)
    NeuralNetwork(int iptN,int midN,int outN,int Nlayers,float eta)
    {
        //this->ila=ila;
        //this->de=de;
        this->eta=eta;

        nnl.resize(Nlayers);

        nnl.front().Init(iptN,midN);
        for (int i = 1; i < int(nnl.size());++i)
            nnl[i].Init(midN,midN);

        nno.Init(midN,outN);

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

        nnl.front().ComputeActivation(ila);
        for (int i = 1; i < int(nnl.size());++i)
            nnl[i].ComputeActivation(nnl[i-1].GetActivation());

        nno.ComputeActivation(nnl.back().GetActivation());// Getting Nanners????!!!!

        ola=nno.GetActivation();

        avgCost+=CalculateCost();
        ++cntr;
    };

    void ComputeDerivatives()
    {
        // Backpropagate
        nno.ComputeInitalError(de);

        nnl.back().ComputeError(nno);

        for (int i = nnl.size()-2; i >= 0;i--)
            nnl[i].ComputeError(nnl[i+1]);

        // Calculate Derivatives
        nnl.front().ComputeDerivatives(ila);

        for ( int i = 1; i < int(nnl.size()) ; ++i )
            nnl[i].ComputeDerivatives(nnl[i-1].GetActivation());

        nno.ComputeDerivatives(nnl.back().GetActivation());
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

        prevcost=currentcost;

        for (auto&& n : nnl)
            n.EndTrainingSet(eta);

        nno.EndTrainingSet(eta);

        cntr=0;
        avgCost=0.0;

        return currentcost;
    };

    void Clear()
    {
        nnl.clear();
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
