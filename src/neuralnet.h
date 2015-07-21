#ifndef NEURAL_NET_C
#define NEURAL_NET_C

#include <vector>
#include <cstring>

#include "tools/random.hpp"
#include "neuralnetlayer.hpp"

class NeuralNetwork
{
    std::vector<float> ila; // Input Layer Activation
    std::vector<float> ola; // Output Layer Activation

    NeuralNetLayer nnl;
    NeuralNetLayer nno;

    NeuralNetwork() {};

public:

    NeuralNetwork(std::vector<float> &ila)
    {
        this->ila.resize(ila.size());
        memcpy(&this->ila[0],&ila[0],ila.size()*sizeof(float));

        nnl.Init(32,100);
        nno.Init(100,32);
    };

    void NewTrainingData(std::vector<float> &ila)
    {
        memcpy(&this->ila[0],&ila[0],ila.size()*sizeof(float));
        ola.clear();
    };

    void ComputeLayers()
    {
        nnl.ComputeActivation(ila);
        nno.ComputeActivation(nnl.GetActivation());

        ola=nno.GetActivation();
    };

    float CalculateCost(std::vector<float> &de)
    {
        float cost = 0.0;

        for (int i=0;i<int(ola.size());++i)
        {
            float diff = de[i] - ola[i];
            cost+=diff*diff;
        };

        return 0.5 * cost;
    };

    void GetOutput(std::vector<float> &ola)
    {
        if (!ola.empty())
            ola.clear();

        ola=this->ola;
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
