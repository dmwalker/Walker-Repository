//
//  fileoptions.cpp
//  mcWHAM
//

#include "fileoptions.h"

/* Standard help message */
void userinfo(double T, double begin, double end, double bin_size, int dof, double tol, int niter, bool bVerbose)
{
    std::cout << "\n   This is a multidimensional Weighted-Histrogram Analysis program writting in C++.  For more information about the algorithm employed, refer to 'BEFEUS: Bayesian Estimation of Free Energy for Umbrella Sampling' by Dr. Daniel L. Ensign.\n    Currently, this build can only handle up to 3 degrees of freedom, but has only been tested up to 2.\n   NOTE 1: Be sure to make sure the file list format correctly adheres to what is listed under the --f option information.  The trajectory # needs to be unique for each individual simulation.  This parameter is used to group different degrees of freedom to the same biased trajectory.  The numbering can start at 0 -or- 1.\n   Note 2:  When using the --fb and --fe flags, the selection will show up at the begining and the end of the run, but the number of frames in each trajectory will still be listed as the total number of frames for tha trajectory, rather than however many were used." << std::endl << std::endl;
    std::cout << "   --f <file list>              : File containing a list of trajectories and\n\t\t\t\t  information about the umbrella potentials applied." << std::endl;
    std::cout << "Format: <trajectory #> <coordinate file> <coordinate bias value (degrees)> <flat range (degrees)> <force constant (kJ mol^-1 rad^-2)> <temperature (K)>" << std::endl;
    std::cout << " ---------------------------------------------------------------------------" << std::endl;
    std::cout << "   --o <out file>               : Output file name" << std::endl;
    std::cout << "   --fb <first frame>           : First frame to read." << std::endl;
    std::cout << "   --fe <last frame>            : Last frame to read." << std::endl;
    std::cout << "   --t <temperature>            : Default = " << T << " K" << std::endl;
    std::cout << "   --s <coordinate start>       : Default = " << begin << " degrees" << std::endl;
    std::cout << "   --e <coordinate end>         : Default = " << end << " degrees" << std::endl;
    std::cout << "   --b <bin size>               : Default = " << bin_size << std::endl;
    std::cout << "   --dof <degrees of freedom>   : Default = " << dof << std::endl;
    std::cout << "   --tol <tolerance>            : Default = " << tol << std::endl;
    std::cout << "   --iter <iterations>          : Default = " << niter << std::endl;
    std::cout << "   --v                          : Be a little more verbose." << std::endl;
    std::cout << "                                  Default = " << bVerbose << std::endl;
    
    return;
};

/* Option parsing */
char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
};

range getMultiCmdOption(char ** begin, char ** end, const std::string & option)
{
    double * values=new double [10]; // arbitrarily large; hopefully 100 degrees of freedom will never be needed
    int n=0;
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        for (int i=0;i<10;i++ )
        {
            if (itr != end && strncmp(*itr,"--",2) != 0 )
            {
                sscanf(*itr, "%lf", &values[n]);
                ++itr;
                n++;
            }
            else { break; };
        }
        range value;
        value.values=values;
        value.n_values=n;
        //for (int i=0;i<value.n_values;i++){std::cout<<value.n_values<<" "<<values[i]<<std::endl;}
        
        return value;
    }
    std::cout << option << " was flagged with no entry following, exiting..." << std::endl;
    exit(1);
};

/* Checking for options to parse though */
bool cmdOptionExists(char** begin, char** end, const std::string & option)
{
    return std::find(begin, end, option) != end;
};

/* How many lines */
int get_n_lines( char * input )
{
    std::string line;
    int lines=0;
    std::ifstream file( input );
    if (file.is_open())
    {
        while (file.good())
        {
            getline( file, line );
            if ( line == "\n" || line == "" ) { break; }
            lines++;
        }
    }
    return lines;
};

void nD_WHAM_grouping( TorsionExperiment * experiment, int dof, int n_files, int first_experiment, int **&group_trajectories, bool bVerbose )
{
    int n_trajectories = n_files/dof;
    int this_many;
    int experiment_index=0;
    std::string grammar;
    if (dof > 1 ) {grammar="s";}
    else {grammar="";}
    
    group_trajectories=new int * [n_trajectories];
    
    for (int i=first_experiment; i<(first_experiment+n_trajectories); i++)
    {
        if (bVerbose)
        {
            std::cout <<"Checking trajectory " << i << " for each degree of freedom." << std::endl;
        }
        this_many=0;
        group_trajectories[experiment_index]=new int [dof];
        for (int this_dof=0; this_dof<dof; this_dof++)
        {
            for (int n=0; n<n_files; n++)
            {
                if (experiment[n].Experiment() == i && experiment[n].DoF() == this_dof)
                {
                    if (bVerbose)
                    {
                        std::cout << experiment[n].Datafile() << " ";
                    }
                    group_trajectories[experiment_index][this_dof]=n;
                    this_many++;
                }
            }
            if (bVerbose){std::cout << std::endl;}
        }
        if (bVerbose){ std::cout << this_many << " DoF file" << grammar << " found for trajectory " << i << std::endl;}
        if ( this_many != dof )
        {
            std::cout << "WARNING: Could not find as many trajectories as degrees of freedom.  Exiting..." << std::endl;
            exit(1);
        }
        experiment_index++;
    }
};


void compare_len( int a, std::string alabel, int b, std::string blabel )
{
    if ( a != b )
    {
        std::cout << "WARNING: The number of " << alabel << ", " << a << ", and the number of " << blabel << ", " << b << ", are not the same. Exiting..." << std::endl;
        exit(1);
    }
};


