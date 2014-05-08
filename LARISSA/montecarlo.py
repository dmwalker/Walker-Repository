#!/usr/bin/python
# David M. Walker
# The University of Texas at Austin
# 11/17/2010
#
# montecarlo.py
#
# Script for MC sampling of model spectra
# TODO 11/17/2010:
#   add a burn-in so it need not be done manually
#   move the actual MC to another module for encapsulation and extensibility
  
import sys
from optparse import OptionParser
from SpectrumModels import *

from numpy import array, exp, loadtxt, ones, zeros
from numpy.random import normal, random
try:
    from pylab import figure, plot, savefig, show, histogram
    havepylab = True
except:
    havepylab = False

from copy import copy

# set up the option parser
parser = OptionParser()

# decide how to minimize -- for now only one of these is possible upon one invokation
parser.add_option("-n", "--niter", dest="niter", action="store", type="int", default=100, help="Number of Monte Carlo iterations" )
parser.add_option("-w", "--step-width", dest="stepwidth", action="store", type=float, default=0.01, help="Step width for proposal steps" )

# Model and data stuff
parser.add_option("-i", "--initial-state", dest="initialfile", action="store", type="string", default=None, help="User-supplied initial MC state in a flat file (REQUIRED)" )
parser.add_option("-s", "--spectrum", dest="spectrumfile", action="store", type="string", default=None, help="Spectrum file (REQUIRED)" )
parser.add_option("-F", "--function", dest="function", action="store", type="string", default=None, help="Function to use (right now: use the marginal p3gaussian, no other option)" )

# general settings
parser.add_option("-H", "--long-help", dest="longhelp", action="store_true", default=False, help="Display long help" )
parser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False, help="Print verbosely" )
parser.add_option("--testmode", dest="testmode", action="store_true", default=False, help="Run in test mode; print initial state only" )

# output
parser.add_option("--output-result", dest="outputfile", action="store", default=None, help="File to write output (stdout is the default) (not functional)" )
parser.add_option("--output-trajectory", dest="trajectoryfile", action="store", default=None, help="File to write the trajectory of model coordinates (default is not to output) (not functional)")
parser.add_option("--plot", dest="plotbool", action="store_true", default=False, help="Show a plot of the final fit (requires pylab)" )

# parse command line arguments
(options, args) = parser.parse_args()
niter = options.niter
stepwidth = options.stepwidth
initialfile = options.initialfile
spectrumfile = options.spectrumfile
function = options.function
longhelp = options.longhelp
verbose = options.verbose 
testmode = options.testmode 
plotbool = options.plotbool

# deal with special cases
if verbose :
    print "If you need help, try adding the --help or --long-help options."

if longhelp or len(sys.argv) == 1 or spectrumfile is None or initialfile is None:
    scriptname = sys.argv[0]
    print "Long help"
    parser.print_help()
    
    print "This is the long help."
    print "A spectrum file and an initial state file (ie, from the output of the minimization"
    print "step) must be supplied. An example:" 
    print "\t%s -s spectrum.txt -i initial.txt"% scriptname
    print "Note that the default step width (-w option) is likely too large to generate useful"
    print "results. This parameter should be tuned until the acceptance rate is between about"
    print "25% and 50%."
    print 
    print "About the algorithm: Currently, this script generates and independence chain Monte"
    print "Carlo simulation. A new state is selected from an uncorrelated multivariate Gaussian"
    print "with the initial state as the mean and the width parameter (-w) as the standard"
    print "deviation."

    if verbose and spectrumfile :
        print "verbose is SET"
    else :
        sys.exit()

# open the spectrum
try:
    frequencies, signals = loadtxt( spectrumfile ).transpose()
except ValueError:
    print "ERROR: Problem reading %s" % spectrumfile
    print "Perhaps the contents of the file is not numeric?"
    sys.exit()
except IOError :
    print "ERROR: %s not found" % spectrumfile
    sys.exit()
except : raise

# open the initial guesses file
try:
    if initialfile: initialState = loadtxt( initialfile )
except ValueError :
    print "ERROR: Problem reading %s" % initialfile
    print "Perhaps the contents of the file is not numeric?"
    sys.exit()
except IOError :
    print "ERROR: %s not found" % initialfile
    sys.exit()
except : raise

# Scale frequencies and signals to lie on 0,1 since it seems to make the algorithms
# more stable.
# This is probably a bad way to do this since inverting it requires knowledge
# of the model function.
minfreq = frequencies.min()
maxfreq = frequencies.max()
frequencies -= minfreq
frequencies /= maxfreq-minfreq

minsig = signals.min()
maxsig = signals.max()
signals -= minsig
signals /= maxsig - minsig

# Set up the model function
model = p3gaussianM( data=(frequencies, signals), guess=initialState, verbose=verbose )
proposal = p3gaussianM( data=(frequencies, signals), guess=initialState, verbose=verbose )
original = p3gaussianM( data=(frequencies, signals), guess=initialState, verbose=verbose )

if testmode :
    print "*** initial state"
    model.showstate()
    sys.exit()

if verbose:
    print
    model.showstate()
    print "RUNNING MONTE CARLO"
    print "Running Metropolis-Hastings targeted to %d acceptance" % niter

nparams = len(model.parameters)
naccepted = 0
nrejected = 0
iteration = 0
nreturn = 0 
samples = [ model.parameters, ]
functions=[ model.calcf(), ]
posvalues=[ model.position, ]
widvalues=[ model.width, ]
while naccepted < niter and iteration < 10*niter :
    currentEnergy = model.energy

    # propose a new model 
    # choose a new model that is another independently chosen Gaussian distance
    # from the original model
    proposal.parameters = original.parameters + normal(0,stepwidth,nparams)

    proposalEnergy = proposal.energy 
    print "Iteration", iteration,
    r = random() 
    boltzfactor = exp( currentEnergy - proposalEnergy ) 
    if r < boltzfactor :
        naccepted += 1
        model.parameters = copy(proposal.parameters)
        print "accepted", 
    else :
        print "rejected", 
        nrejected += 1
    print "transition probability", boltzfactor,
    print "r =", r,
    print "energy %f -> %f" % ( currentEnergy, proposalEnergy ),
    print "accepted", naccepted
    samples.append( model.parameters )
    functions.append( model.calcf() )
    posvalues.append( model.position )
    widvalues.append( model.width )
    iteration += 1

try :
    print "Accepted %d of %d trials (%f)" % ( naccepted, naccepted+nrejected, naccepted/float(naccepted+nrejected) )
except ZeroDivisionError :
    print "Accepted all transitions (this is Bad)"


# compute the noise level
print "Results (scaled units):"
samples=array(samples)
print "mean =", samples.mean(0)
print "std =", samples.std(0)
residuals = signals - functions[-1]
print "residual mean =", residuals.mean()
print "residual std =", residuals.std() 
print "residual prec (1/std)", 1/residuals.std()

print "Results (unscaled units):"
positions = array(posvalues)*(maxfreq-minfreq) + minfreq
meanpos = positions.mean()
stdpos = sqrt( (positions**2).mean() - meanpos**2 )
widths = array(widvalues)*(maxfreq-minfreq)
meanwid = widths.mean()
stdwid = sqrt( (widths**2).mean() - meanwid**2 )
print "mean frequency =", meanpos, "std frequency =", stdpos
print "mean width =", meanwid, "std width =", stdwid

if plotbool and havepylab:
    figure(0)
    for function in functions :
        plot( frequencies, function )
    plot( frequencies, signals, "k-" )
    figure(1)
    plot( array(Bvalues)*(maxfreq-minfreq) + minfreq )
    show()

