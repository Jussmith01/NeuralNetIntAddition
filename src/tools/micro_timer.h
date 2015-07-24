#ifndef MICROTIMER_HPP
#define MICROTIMER_HPP

#include <string>
#include <sstream>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <iostream>

//________________________________________________________________________//
//      *************************************************************     //
//                               Timer Class
//                  Holds timer variables and class functions
//      *************************************************************     //
class microTimer
{
    //--------------------------
    //Private Class Declarations
    //--------------------------

    double start_wall_time; //Holds start time
    long int wt_count; //Holds end time
    double accumtime;

    clock_t start_clock_time;
    long int ct_count;
    double accumclock;

    //------------------------------
    //Private Member Class Functions
    //------------------------------

    //Intakes a value of time in seconds and returns a string formmatted as:
    //Computation Wall Time: d days h hours m minutes s seconds
    std::string mk_time_string(double time_val);

public:
    microTimer()
    {
        Init();
    };

    //-----------------------------
    //Public Member Class Functions
    //-----------------------------
    // Initialie the class
    void Init(void);

    //Sets the start time for the timer
    void start_point(void);

    //Sets the end time for the timer
    void end_point(void);

    //Prints run_timer = end_time - start_time
    //template<typename T>
    void print_wall_time(std::string message,std::ostream &output);

    //Prints the clock time
    //template<typename T>
    void print_clock_time(std::string message,std::ostream &output);

    // Generic Time Print
    //template<typename T>
    void print_generic(std::string message,std::ostream &output);

    std::string get_generic_print_string(std::string message);

    //Resets the timer if needed
    void reset(void);
};

#endif

