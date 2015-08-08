#include <iostream>
#include <iomanip>
#include <fstream>
#include <limits>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>

#include "tools/micro_timer.h"
#include "tools/binaryconversion.hpp"
#include "neuralnet.h"
#include "tools/csvreader.hpp"

int main(int argc, char *argv[])
{
    if (argv[1]==NULL || argv[2]==NULL || argv[3]==NULL || argv[4]==NULL || argv[5]==NULL || argv[6]==NULL || argv[7]==NULL)
    {
        std::cout << "Error: Missing arguments!" << std::endl;
        std::cout << "Syntax: ./NeuralNetIntAddition [eta] [tss] [ess] [hls] [nns] [con] [fn]" << std::endl;
        std::cout << "   eta: The learning rate" << std::endl;
        std::cout << "   tss: The Training Set Size" << std::endl;
        std::cout << "   ess: The tEsting Set Size" << std::endl;
        std::cout << "   hls: The hidden layer size" << std::endl;
        std::cout << "   nnd: The Neural Net Depth" << std::endl;
        std::cout << "   con : Convergence of average cost" << std::endl;
        std::cout << "   fn : CSV data file name" << std::endl;

        exit(1);
    }

    double eta = atof(argv[1]);
    int tss = atoi(argv[2]);
    int ess = atoi(argv[3]);
    int hls = atoi(argv[4]);
    int nnd = atoi(argv[5]);
    double con = atof(argv[6]);
    std::string filename(argv[7]);

    std::cout << "eta: " << eta << " tss: " << tss  << " ess: " << ess << " hls: " << hls << " nnd: " << nnd << " con: " << con << " filename: " << filename <<  std::endl;

    // Working vectors
    std::vector<double> input(3);
    std::vector<double> desired(9);
    std::vector<double> output(9);

    // Open File
    std::fstream file(filename.c_str());

    // Build index
    std::vector<int> irand(tss);
    for (int i=0;i<tss;++i)
    {
        irand[i]=i;
    }

    // Float scalar
    double scalari(7.0);
    double scalarf(9.5);
    double shiftf(0.5);

    // Define network
    NeuralNetwork nn(12,hls,9,nnd,eta);

    int ep=0;
    double avgcost = 100.0;

    // Define timer
    microTimer mt;

    while (avgcost > con || ep > 1500)
    {
        mt.start_point();

        std::random_shuffle(irand.begin(),irand.end());

        for (auto&& i : irand)
        {
            //Begin Neural Network Computation
            std::string data(GotoLine(file,i));
            csvreader(data,input,3,14,scalarf,shiftf);
            csvreader(data,desired,15,23,scalarf,shiftf);

            nn.NewTrainingData(input,desired);
            nn.ComputeLayers();
            nn.ComputeDerivatives();
            nn.ResetForNewTrainingData();
        }

        std::cout << " Epoch " << ep << " - ";
        avgcost = nn.CompleteTrainingSet();

        ++ep;

        mt.end_point();
        std::cout << mt.get_generic_print_string(" ") << std::endl;
        mt.reset();
    }

    irand.clear();

    std::cout << " |------Testing Set-------|\n";
    RandomInt itestrandgen; // Train with a set of 10000
    std::vector<int> irandtest(ess);
    itestrandgen.FillVector(irandtest,-1000000,1000000);
    itestrandgen.Clear();

    std::vector<double> iforce;

    std::ofstream am1g("am1force.graph");
    std::ofstream pm6g("pm6force.graph");
    std::ofstream nncg("nncforce.graph");
    std::ofstream am1pm6g("am1pm6force.graph");
    std::ofstream pm6nncg("pm6nncforce.graph");

    am1g.setf( std::ios::fixed, std::ios::floatfield );
    pm6g.setf( std::ios::fixed, std::ios::floatfield );
    nncg.setf( std::ios::fixed, std::ios::floatfield );
    am1pm6g.setf( std::ios::fixed, std::ios::floatfield );
    pm6nncg.setf( std::ios::fixed, std::ios::floatfield );

    for (int i=29000;i<29999;++i)
    {
        //Begin Neural Network Computation
        std::string data(GotoLine(file,i));
        csvreader(data,input,3,14,scalarf,shiftf);
        csvreader(data,iforce,6,14,scalarf,shiftf);
        csvreader(data,desired,15,23,scalarf,shiftf);

        nn.NewTrainingData(input,desired);
        nn.ComputeLayers();

        nn.GetOutput(output);


        for (int j=0;j<int(output.size());++j)
        {
            am1pm6g << std::setprecision(7) << iforce[j]  << "  " << desired[j] << std::endl;
            pm6nncg << std::setprecision(7) << desired[j] << "  " << output[j] << std::endl;
            am1g << std::setprecision(7) << iforce[j] << std::endl;
            pm6g << std::setprecision(7) << desired[j] << std::endl;
            nncg << std::setprecision(7) << output[j] << std::endl;
        }
    }

    am1g.close();
    pm6g.close();
    nncg.close();
    am1pm6g.close();
    pm6nncg.close();

    //std::cout << "Accuracy:" << correct/double(ess) << std::endl;

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
