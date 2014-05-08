//
//  fileoptions.h
//  mcWHAM
//


#ifndef mcWHAM_fileoptions_h
#define mcWHAM_fileoptions_h

#include <algorithm>
#include "WHAMfunctions.h"

#endif

/* Standard help message */
void userinfo(double T, double begin, double end, double bin_size, int dof, double tol, int niter, bool bVerbose);

/* Option parsing */
char* getCmdOption(char ** begin, char ** end, const std::string & option);

struct range
{
    double * values;
    int n_values;
};

range getMultiCmdOption(char ** begin, char ** end, const std::string & option);

/* Checking for options to parse though */
bool cmdOptionExists(char** begin, char** end, const std::string & option);
   
/* How many lines */
int get_n_lines( char * input );

void nD_WHAM_grouping( TorsionExperiment * experiment, int dof, int n_files, int first_experiment, int **&group_trajectories, bool bVerbose );


void compare_len( int a, std::string alabel, int b, std::string blabel );


