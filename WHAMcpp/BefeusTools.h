#ifndef mcWHAM_BefeusTools_h
#define mcWHAM_BefeusTools_h
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>
#include <iterator>
#include <cstdlib>
#include <cstring>

#endif


struct vec
{
    double x, y;
};

vec convert( double angle );

double periodicity ( double angle );

double angle_difference(double a, double b);

double remove_periodicity( double angle );

class Experiment
{
    double * coordinates;
    int * bincounts, * frame_bins;
    int frame, nbins;
public:
    int nFrames();
    int nBins();
    double * Coord();
    int * BinCounts();
    int * BinID();
    
    void readtrajectory( std::string trajectory, double begin, double end, double bin_sizes, bool bVerbose );
    
    void histogram( double * array, double begin, double end, double bin_size, bool bVerbose );
};


class TorsionExperiment
{
    int experiment, this_dof;
    double phi0, deltaphi, fc, t, beta, begin, end, bin_size;
    std::string xvgfile;
    long double * potentials;
    long double * omega;
    bool precision_error;
public:    
    int DoF();
    int Experiment();
    double Phi0();
    double DeltaPhi();
    double FC();
    double T();
    std::string Datafile();
    long double * Omega();
    long double * Potential();
    
    class Experiment trajectory;

    void read_filelist( std::string line, int previous_experiment, int previous_dof, double *begins, double *ends, double *bin_sizes, double kb, bool bVerbose);
        
            
    void weight_dihedral_restraint_potentials();
    
    long double weight_dihedral_restraint_potential( double phi );
    
    long double trapezoidal_integration( double a, double b );
    
    void assign_omega();
};

