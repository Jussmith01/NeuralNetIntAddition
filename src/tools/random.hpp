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
                int seedOffset=(int)Time;

                array.resize(w);

                int t = (int)omp_get_wtime()+i;
                std::seed_seq seed = {seedOffset,t,i+100};
                seed.generate(array.begin(),array.end());//Seed the generator
                index = 0;
        };

        float GenRandReal(float mean,float stdv)
        {
                generator.seed(array[index]);//Seed the generator
                std::normal_distribution<float> distribution(mean,stdv);//Setup the distribution
                float RN = (float)distribution(generator);//Denerate the random number
                //std::cout << "RandomNumber: " << RN << std::endl;
                ++index;//Increase seed offset
                return RN;
        };

        void FillVector(std::vector<float> &vec,float mean,float stdv)
        {
            int N = vec.size();
            Setup(N,1728);

            for (auto&& v : vec)
                v=GenRandReal(mean,stdv);
        };

        void Clear()
        {
            array.clear();
        };
};


#endif
