#ifndef RANDOM_C
#define RANDOM_C

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <stdio.h>
#include <time.h>
#include <vector>
#include <random>
#include <omp.h>
#include <math.h>


/*-----------------------------------------------
  ***********************************************
  |Class for Generating a Flt from a Normal Dist|
  ***********************************************
Example Use:
NormRandomReal NR(w,i); //int w=num seeds
                        //int i=thread seed

float someflt = GenRandReal(float mean,float std)
                   //mean is mean, duh
                   //std is standard deviation
-------------------------------------------------*/
class NormRandomReal
{
        std::default_random_engine generator;
        std::vector<int> array;
        int index;

        public:
        NormRandomReal(){};
        NormRandomReal(int w,int seed){Setup(w,seed);};

        void Setup(int w,int i)
        {
                time_t Time;
                time(&Time);
                int seedOffset=(int)Time+clock();

                array.resize(w);

                int t = (int)omp_get_wtime()+i;
                std::seed_seq seed = {seedOffset,t,i};
                seed.generate(array.begin(),array.end());//Seed the generator
                index = 0;
        };

        double GenRandReal(double mean,double stdv)
        {
                //std::cout << " " << array[index];
                generator.seed(array[index]);//Seed the generator
                std::normal_distribution<double> distribution(mean,stdv);//Setup the distribution
                double RN = (double)distribution(generator);//Denerate the random number
                //std::cout << "RandomNumber: " << RN << std::endl;
                ++index;//Increase seed offset
                return RN;
        };

        void FillVector(std::vector<double> &vec,double mean,double stdv,int seed)
        {
            int N = vec.size();
            Setup(N,seed);

            //std::cout << " Generating (" << N  << ") random numbers." << std::endl;

            for (auto&& v : vec)
            {
                v=GenRandReal(mean,stdv);
                //std::cout << v << " ";
            }
            //std::cout << "\n";
        };

        void Clear()
        {
            array.clear();
        };
};

/*----------------------------------------
  ***************************************
  |  Class for generating random Ints   |
  ***************************************
Example Use:
RandomInt RI(w,i); //int w=num seeds
		   //int i=thread seed

int someInt = GenRandInt(high,low)
		   //Parameters give the
		   //range of the int rtn'd
------------------------------------------*/
class RandomInt
{
        std::default_random_engine generator;
        std::vector<int> array;
        int index;

        public:
        RandomInt(){};
        RandomInt(int w,int i){Setup(w,i);};

        void Setup(int w,int i)
        {
                time_t Time;
                time(&Time);
                int seedOffset=(int)Time;

                array.resize(w);

                int t = (int)omp_get_wtime()+i;
                std::seed_seq seed = {seedOffset,t,i+100};
                seed.generate(array.begin(),array.end());//Seed the generator
                index = 0;
        };

        int GenRandInt(int high,int low)
        {
                generator.seed(array[index]);//Seed the generator
                std::uniform_int_distribution<int> distribution(low,high);//Setup the distribution
                int RN = (int)distribution(generator);//Denerate the random number
                //std::cout << "RandomNumber: " << RN << std::endl;
                ++index;//Increase seed offset
                return RN;
        };

        void FillVector(std::vector<int> &vec,int low,int high)
        {
            int N = vec.size();
            Setup(N,63463);

            for (auto&& v : vec)
                v=GenRandInt(high,low);
        };

        void Clear()
        {
            array.clear();
        };
};

#endif
