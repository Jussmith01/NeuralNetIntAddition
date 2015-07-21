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

    NeuralNetLayer nnl;
    NeuralNetLayer nno;

    NeuralNetwork() {};

public:

    //NeuralNetwork(std::vector<float> &ila,std::vector<float> &de,float eta)
    NeuralNetwork(int iptN,int midN,int outN,float eta)
    {
        //this->ila=ila;
        //this->de=de;
        this->eta=eta;

        nnl.Init(iptN,midN);
        nno.Init(midN,outN);

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
        nnl.ComputeActivation(ila);
        nno.ComputeActivation(nnl.GetActivation());// Getting Nanners????!!!!

        ola=nno.GetActivation();

        avgCost+=CalculateCost();
        ++cntr;
    };

    void ComputeDerivatives()
    {
        nno.ComputeInitalError(de);
        nnl.ComputeError(nno);

        nnl.ComputeDerivatives(ila);
        nno.ComputeDerivatives(nnl.GetActivation());
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

    void CompleteTrainingSet()
    {
        std::cout << " avgCost of training set: " << avgCost/double(cntr) << std::endl;
        //ResetForNewTrainingData();

        nnl.EndTrainingSet(eta);
        nno.EndTrainingSet(eta);

        cntr=0;
        avgCost=0.0;
    };

    void Clear()
    {
        nnl.Clear();
        nno.Clear();
        ila.clear();
        ola.clear();
    };
};

#endif