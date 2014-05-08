#!/usr/bin/python
"""
David M. Walker
The University of Texas at Austin
11/17/2010

mc-run.py
Run on the command line with -h or -H for help.

TODO
- modularize the modifications to the spectrum (rescaling, etc.)
- modularize generation of eps files; there is a lot of overlap; also calling
    figure(N) is going to get clunky very soon
- modularize generation of flat text files
- modularize the log file/message printing and combine with 'verbose' setting
"""  

import sys
from optparse import OptionParser
from SpectrumModels import *
from MCAlgorithms import *

#from numpy import array, exp, loadtxt, ones, savetxt, zeros
#from numpy.random import normal, random
from numpy import loadtxt
from pylab import figure, plot, savefig, show

def modify_txt_extension( refname, extension ):
    extension = "-mc" + extension
    if refname[-4:] == ".txt":
        newname = refname.replace( ".txt", extension )
    else : newname = refname + extension
    return newname

# set up the option parser
parser = OptionParser()

# decide how to minimize -- for now only one of these is possible upon one invokation
parser.add_option("-n", "--niter", dest="niter", action="store", type="int", default=100, help="Number of Monte Carlo iterations" )
parser.add_option("-k", "--step-width", dest="stepwidth", action="store", type=float, default=0.01, help="Step width for proposal steps (default 0.01, which may not give a suitable acceptance ratio)" )
parser.add_option("-w", dest="stepwidth_dep", action="store", type=float, default=0, help="Deprecated step width flag (use -k)" )

# Model and data stuff
parser.add_option("-G", "--initial-state", dest="initialfile", action="store", type="string", default=None, help="User-supplied initial MC state in a flat file (REQUIRED)" )
parser.add_option("-s", "--spectrum", dest="spectrumfile", action="store", type="string", default=None, help="Spectrum file (REQUIRED)" )
parser.add_option("-F", "--function", dest="function", action="store", type="string", default=None, help="Function to use (right now: use the marginal p3gaussian, no other option)" )

# general settings
parser.add_option("-H", "--long-help", dest="longhelp", action="store_true", default=False, help="Display long help" )
parser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False, help="Print verbosely" )
parser.add_option("--testmode", dest="testmode", action="store_true", default=False, help="Run in test mode; print initial state only" )

# output
parser.add_option("--plot", dest="plotbool", action="store_true", default=False, help="Show a plot of the final fit (requires pylab)" )

# parse command line arguments
(options, args) = parser.parse_args()
niter = options.niter
stepwidth = options.stepwidth
stepwidth_dep = options.stepwidth_dep
if stepwidth_dep > 0: stepwidth = stepwidth_dep
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
    print "ERROR: Spectrum file %s not found" % spectrumfile
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
    print "ERROR: Initial guess file %s not found" % initialfile
    sys.exit()
except : raise

# Scale frequencies and signals to lie on 0,1 since it seems to make the algorithms
# more stable.
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

# Some pre-MC checking and messaging
if testmode :
    print "*** initial state"
    model.showstate()
    sys.exit()

if verbose:
    print
    model.showstate()
    print "RUNNING MONTE CARLO"
    print "Running Metropolis-Hastings targeted to %d acceptance" % niter

results = ""
results += "Command line: "
for elem in sys.argv : results += elem + " "
results += "\n\n"
results += "Settings:\n"
results += "niter %d\n" % niter 
results += "stepwidth %f\n" % stepwidth 
results += "initialfile %s\n" % initialfile
results += "spectrumfile %s\n" % spectrumfile 
results += "function %s\n" % function 

# Run MC
mc = GaussianIndependenceChain( model, stepwidth, niter, niter, verbose=verbose ) 

# Deal with results
# Set up a model function using the mean parameters
samples = mc.samples
samplemean = samples.mean(0)
samplestd = samples.std(0)
avgmodel = p3gaussianM( data=(frequencies, signals), guess=samples.mean(0), verbose=verbose )

results += "\n"
results += "MC Results:\n"

# Convert back to frequency units
positions = mc.positions*(maxfreq-minfreq) + minfreq
positionmean = positions.mean()
positionstd = positions.std()
widths = mc.widths*(maxfreq-minfreq)
widthmean = widths.mean()
widthstd = widths.std()
fwhm = 2*sqrt( 2*log(2) )*widthmean
fwhmerr = 2*sqrt( 2*log(2) )*widthstd 

results += "mean frequency = %4.3f std frequency = %e\n" % ( positionmean, positionstd )
results += "mean width = %4.3f std width = %e\n" % ( widthmean, widthstd )
results += "fwhm = %4.3f err = %e\n" % ( fwhm, fwhmerr )

naccept = mc.naccepted
nreject = mc.nrejected
niterations = naccept + nreject
acceptanceRatio = mc.acceptanceratio * 100
results += "Accepted %d of %d trials (%1.4f %%)\n" % ( naccept, niterations, acceptanceRatio )

results += "mean parameters = " 
for value in samplemean: results += "%6e " % value
results += "\n"
results += "std parameters = "
for value in samplestd: results += "%6e " % value
results += "\n"

if verbose: print results
logfile = modify_txt_extension( spectrumfile, ".log" )
file = open( logfile, "w" )
file.write( results )
file.close()
if verbose: print "Wrote log file to '%s'" % logfile

# Do plotting
if plotbool:
    figure(0)
    for farray in functions :
        plot( frequencies, farray )
    plot( frequencies, signals, "k-" )
    figure(1)
    plot( positions )
    show()

# To generate a baseline-corrected spectrum, subtract the baseline of the average
# model from MC. Note we are still working with rescaled absorbances. 
baseline = avgmodel.baseline
correctedsignal = signals - baseline

# Generate a PostScript plot of the peak position.
positionsfile = modify_txt_extension( spectrumfile, ".pos.eps" )
figure(1)
plot(positions)
savefig(positionsfile)
if verbose: print "Wrote peak position trajectory to '%s'" % positionsfile

# Generate a PostScript plot using the corrected baseline. First we need to rescale
# the frequencies. 
frequencies = frequencies*(maxfreq-minfreq) + minfreq
figure(2)
plot(frequencies, correctedsignal)
correctedfile = modify_txt_extension( spectrumfile, ".cor.eps" )
savefig(correctedfile)
if verbose: print "Wrote baseline corrected spectrum plot to '%s'" % correctedfile

# Write the corrected spectrum as a flat text file
corrspectrumfile = modify_txt_extension( spectrumfile, ".cor.txt" )
fitdata = array((frequencies, correctedsignal)).transpose()
savetxt( corrspectrumfile, fitdata )
if verbose: print "Wrote baseline corrected spectrum to '%s'" % corrspectrumfile
