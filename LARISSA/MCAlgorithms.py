#!/usr/bin/python
"""
David M. Walker
The University of Texas at Austin
11/18/2010

MCAlgorithms.py

Algorithms for Monte Carlo sampling of a spectrum fit posterior distribution.
The algorithms assume the following interface with 'model' objects, such as
those in SpectrumModels.py:

Methods:
model.calcf(self, frequency=None) 
    Returns the value of the model function using the parameters defining the
    current state of the object. Default is to calculate the model function at 
    all frequencies in the data.

model.guessparameters(self)
    Guesses an initial set of parameters for the model from the input 
    frequencies and signals. Used if no initial state is supplied to the model
    function. Guessing in this way is probably not a good idea for Monte
    Carlo; an optimization step should be run to get an initial state.

model.showstate(self) 
    Print information about the model's current state

model.__len__(self)
    The number of parameters in the model

Attributes:
model.baseline
    The basline part of the signal, as defined by the model and the current state's 
    parameters

model.energy
    Energy at the current state (arbitrary units). 'Energy' means the negative natural logarithm of the probability.

model.force
    Force at the current state (arbitrary units)

model.parameters
    List of model parameters

model.position
    The position parameter of the model in the current state

model.width
    The width parameter of the model in the current state
"""
  
from numpy import array, exp, nan, savetxt, zeros
from numpy.random import normal, random
from copy import copy

class MCAlgorithm( object ):
    """
    The MCAgorithm base class. Classes derived from MCAlgorithm interface with 
    objects such as those defined in SpectrumModels.py.
    """
    def __init__( self, model, stepsize, target_naccept, maxiter=None, verbose=False ):
        self.verbose = verbose
        self.model = model
        self.nparams = len(model)
        self.target_naccept = target_naccept
        if maxiter is None: self._maxiter = 10*self.target_naccept
        elif maxiter < self.target_naccept:
            if self.verbose:
                print "WARNING: maxiter is less than naccept"
                print "Setting maxiter to target_naccept=%d" % ( self.target_naccept )
            self._maxiter= self.naccept
        else: self._maxiter = maxiter
        self.stepsize = stepsize

        self._proposal = copy(self.model)
        self._proposal.parameters = zeros( self.nparams )

        self._naccepted = 0
        self._nrejected = 0
        self._samples = [ model.parameters, ]
        self._functions = [ model.calcf(), ]
        self._positions = [ model.position, ]
        self._widths = [ model.width, ]

        self._init_more()

        self._sample()

    def _init_more( self ): pass

    def _getnaccepted( self ):
        """The number of accepted MC moves."""
        return self._naccepted
    naccepted = property( _getnaccepted )

    def _getnrejected( self ):
        """The number of rejected MC moves."""
        return self._nrejected
    nrejected = property( _getnrejected )

    def _getacceptanceratio( self ):
        """The acceptance ratio."""
        total = float( self.naccepted + self.nrejected )
        try :
            ratio = self.naccepted/total
        except ZeroDivisionError:
            ratio = nan
        return ratio
    acceptanceratio = property( _getacceptanceratio )

    def _getsamples( self ):
        """The samples from MC iteration."""
        return array(self._samples)
    samples = property( _getsamples )

    def _getfunctions( self ):
        """The model functions evaluated at the samples of MC iteration."""
        return array(self._functions)
    functions = property( _getfunctions )

    def _getpositions( self ):
        """The peak positions evaluated at the samples of MC iteration."""
        return array(self._positions)
    positions = property( _getpositions )

    def _getwidths( self ):
        """The peak widths evaluated at the samples of MC iteration."""
        return array(self._widths)
    widths = property( _getwidths )

    def _propose(self):
        """The proposed move. The base class does not propose a move."""
        pass

    def _vprint( self, message ):
        if self.verbose: print message

    def _sample( self ):
        """The MC algorithm."""
        iteration = 0
        while self._naccepted < self.target_naccept and iteration < self._maxiter :
            currentEnergy = self.model.energy

            # Propose a new model; this alters self._proposal 
            self._propose() 
            proposalEnergy = self._proposal.energy 
            message = "Iteration %d " % iteration
            r = random() 
            probabilityRatio = exp( currentEnergy - proposalEnergy ) 
            if r < probabilityRatio:
                self._naccepted += 1
                self.model.parameters = copy(self._proposal.parameters)
                message += "accepted "
            else :
                message += "rejected "
                self._nrejected += 1
            message += "transition probability %6e " % probabilityRatio
            message += "r = %1.4f " % r
            message += "energy %f -> %f " % ( currentEnergy, proposalEnergy )
            message += "accepted %d" % self._naccepted
            self._vprint( message )

            self._samples.append( self.model.parameters )
            self._functions.append( self.model.calcf() )
            self._positions.append( self.model.position )
            self._widths.append( self.model.width )
            iteration += 1

class GaussianIndependenceChain( MCAlgorithm ):
    """
    A MC algorithm which proposes new states whose parameters are selected from
    a Gaussian with mean equal to the initial state and standard deviation defined
    by self.stepsize. For this to be a useful algorithm, the initial state must be
    very close to the posterior mode.
    """
    def _init_more( self ): self._originalParameters = copy(self.model.parameters)

    def _propose( self ):
        """
        Propose a new model that is a Gaussian step from the original state.
        """
        self._proposal.parameters = self._originalParameters + self._step()
    
    def _step( self ): return normal( 0, self.stepsize, self.nparams )
