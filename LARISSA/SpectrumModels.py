# SpectrumModels.py module
# David M. Walker
# The University of Texas at Austin
# 11/17/2010
# 
# TODO:
#   put the p3gaussian model here (where we explicitly simulate tau)
#   write the p3g2 model (2 gaussians)

from numpy import array, concatenate, exp, inf, log, polyfit, sqrt, zeros, ones, transpose, dot
from numpy.linalg import norm, inv

try:
    from pylab import plot, savefig, show
    plottable = True
except ImportError:
    plottable = False

class SpectrumModel( object ):
    """A generic spectrum model."""
    def __init__( self, data=[ [], [] ], verbose=False ):
        self.data = data
        self.parameters = []
        self.verbose = verbose
        self.width = 0
        self.position = 0

    def __len__( self ):
        return len(self.parameters)

    def _setdata( self, data ):
        self.frequencies = data[0]
        self.signals = data[1] 
        self.lfreq = len(self.frequencies)
        self.lsig = len(self.signals)
        if self.lfreq != self.lsig :
            raise DataMismatchError( self.lfreq, self.lsig )
        else :
            self.ldata = self.lfreq
    def _getdata( self ):
        return array( ( self.frequencies, self.signals ) ).transpose()
    data = property( _getdata, _setdata )
    
    def _setfrequencies( self, freq ):
        self._frequencies = array( freq, float )
    def _getfrequencies( self ):
        return self._frequencies
    frequencies = property( _getfrequencies, _setfrequencies )

    def _setsignals( self, sig ):
        self._signals = array(sig, float )
    def _getsignals( self ):
        return self._signals
    signals = property( _getsignals, _setsignals )

    def _getbaseline( self ): 
        """Return the baseline part of the model."""
        return 0
    baseline = property( _getbaseline )

    def plot( self, fname=None ):
        if plottable :
            fdata = self.calcf()
            plot( self.frequencies, self.signals )
            plot( self.frequencies, fdata )
            if fname: savefig(fname)
            else: show()
        elif verbose :
            print "Cannot plot, the pylab module is not available"

    def showstate( self ): 
        """Print out information about the current state of this object."""
        print "Generic object state 0"       
    def energy( self ): return 0
    def force( self ): return 0
    def calcf( self ): return 0
 
class p3gaussianM( SpectrumModel ) :
    """An object to represent the probability distribution of 
        p \propto [ sum_i (d_i-f(n_i))^2 ]^(-N/2)
    which is the marginal (removing t) of 
        p \propto t^(N-1) exp[(-(1/2)t^2 sum_i (d_i-f(n_i))^2 ] 
    where
        f(n) = a + b n + c n^2 + d n^3 + A exp( -(1/2) C^2 (n-B)^2 )
    and calculate given data d (points di) and frequencies n (points ni)
        Fa * S = N sum_i di-f(ni)
        Fb * S = N sum_i (di-f(ni))ni
        Fc * S = N sum_i (di-f(ni))ni^2
        Fd * S = N sum_i (di-f(ni))ni^3
        FA * S = N sum_i (di-f(ni)) Ei
        FB * S = N A C^2 sum_i (di-f(ni)) Ei (ni-B)
        FC * S = - N A C sum_i (di-f(ni)) Ei (ni-B)^2
    where 
        Ei = exp[ -(1/2) C^2 (ni-B)^2 ]
    and 
        S = sum_i ( di-f(ni) )^2
    """
    def __init__( self, data, guess=None, verbose=False ) :
        self.verbose = verbose

        # This _sets self.frequencies and self.signals
        self.data = data

        # Set the initial state. self.guessparameters() needs self.data to
        # function.
        try : a, b, c, d, A, B, C = guess
        except TypeError : 
            a, b, c, d, A, B, C = self.guessparameters() 
            if self.verbose: 
                print "Guess:", a,b,c,d,A,B,C
        self.a = float(a) 
        self.b = float(b) 
        self.c = float(c) 
        self.d = float(d) 
        self.A = float(A) 
        self.B = float(B) 
        self.C = float(C)

        self.frequencies2 = self.frequencies*self.frequencies
        self.frequencies3 = self.frequencies2*self.frequencies
        self.forcearray = zeros( 7, "float" )
 
    def _getparameters( self ) :
        return array((self.a, self.b, self.c, self.d, self.A, self.B, self.C ))
    def _setparameters( self, values ):
        self.a, self.b, self.c, self.d, self.A, self.B, self.C = values
    parameters = property(_getparameters, _setparameters, None, "Parameter list a, b, c, d, A, B, C" )

    def _geta( self ): return self._a
    def _seta( self, value) : self._a = value
    a = property( _geta, _seta, None, "Zeroth order polynomial coefficient" ) 

    def _getb( self ): return self._b
    def _setb( self, value) : self._b = value
    b = property( _getb, _setb, None, "First order polynomial coefficient" ) 

    def _getc( self ): return self._c
    def _setc( self, value) : self._c = value
    c = property( _getc, _setc, None, "Second order polynomial coefficient" ) 

    def _getd( self ): return self._d
    def _setd( self, value) : self._d = value
    d = property( _getd, _setd, None, "Third order polynomial coefficient" ) 

    def _getA( self ): return self._A
    def _setA( self, value) : self._A = value
    A = property( _getA, _setA, None, "Gaussian amplitude"  ) 
    
    def _getB( self ): return self._B
    def _setB( self, value) : self._B = value
    B = property( _getB, _setB, None, "Gaussian mean" ) 
    
    def _getC( self ): return self._C
    def _setC( self, value) : 
        self._C = float(value)
        try:
            self._invC = 1/float(value)
        except ZeroDivisionError:
            self._invC = inf
    C = property( _getC, _setC, None, "Gaussian precision" ) 
    
    def _getinvC( self ): return self._invC
    def _setinvC( self, value ):
        self._invC = float(value)
        try:
            self._C = 1/float(value)
        except ZeroDivisionError:
            self._C = inf
    invC = property( _getinvC, _setinvC, None, "Gaussian width" )

    def _getC2( self ): return self._C*self._C
    C2 = property( _getC2, None, None, "Gaussian precision squared" )

    def _getbaseline( self ): 
        """Return the baseline part of the model."""
        return self.calcpolynomial()
    baseline = property( _getbaseline )
        
    # Before we build the posterior distribution, we might need some good
    # guesses for the parameters.
    def guessparameters( self ):
        # For the polynomial part, we use numpy.polyfit to _get those coeffs.
        # We do this on the first 20% of the data and the last 20% of the data.
        twentypercent = int( 0.2 * self.ldata )
        eightypercent = int( 0.8 * self.ldata )
        freq20 = self.frequencies[ : twentypercent ]
        freq80 = self.frequencies[ eightypercent : ]
        sig20 = self.signals[ :twentypercent ]
        sig80 = self.signals[ eightypercent: ]
        d, c, b, a = polyfit( concatenate((freq20, freq80)), concatenate((sig20, sig80)), 3 )
        
        # For the gaussian let us assume
        # 1. The height A is 1/3 the signal range
        # 2. The location B is the center of the frequency data
        # 3. The width 1/C is 1/12 of the frequency range
        A = ( self.signals.max() - self.signals.min() )/3.0
        B = self.frequencies.mean()
        C = 12.0/( self.frequencies.max() - self.frequencies.min() )

        return a, b, c, d, A, B, C

    def calcgaussian( self, frequency=None ):
        """Calculates exp( -(1/2) C^2 (ni-B)^2 ) as a list if no frequency is given (default)"""
        if frequency :
            diff = frequency - self.B
        else :
            diff = self.frequencies - self.B
        return exp( -0.5 * diff * diff * self.C2 )

    def calcpolynomial( self, frequency=None ):
        """Calculates a + b ni + c ni^2 + d ni^3 as a list if no frequency is given (default)"""
        o0 = self.a
        if frequency :
            o1 = self.b * frequency
            o2 = self.c * frequency * frequency
            o3 = self.d * frequency * frequency * frequency
        else :
            o1 = self.b * self.frequencies
            o2 = self.c * self.frequencies2
            o3 = self.d * self.frequencies3
        return o0 + o1 + o2 + o3

    def calcf( self, frequency=None ):
        """Calculates polypart + A * gausspart as a list if no frequency is given (default)"""
        if frequency : return self.A * self.calcgaussian( frequency ) + self.calcpolynomial( frequency )
        else : return self.A * self.calcgaussian() + self.calcpolynomial()

    def _calcdatadiff( self ) :
        """Calculates ( di - f(ni) ) as a list"""
        return self.signals - self.calcf()

    def _calcfreqdiff( self ):
        """Calculates ( ni - B ) as a list"""
        return self.frequencies - self.B

    def _calcforcedenominator( self ):
        datadiff = self._calcdatadiff()
        return ( datadiff * datadiff ).mean()

    def _calcforce_a( self ):
        """Computes Fa * S"""
        return ones(len(self.frequencies))

    def _calcforce_b( self ):
        """Computes Fb * S"""
        return  (  self.frequencies )

    def _calcforce_c( self ):
        """Computes Fc * S"""
        return (  self.frequencies2 )

    def _calcforce_d( self ):
        """Computes Fd * S"""
        return (  self.frequencies3 )

    def _calcforce_A( self ):
        """Computes FA * S"""
        return(self.calcgaussian())


    def _calcforce_B( self ):
        """Computes FB * S"""
        return self.A * self.C2 * (   self.calcgaussian() * self._calcfreqdiff() )

    def _calcforce_C( self ):
        """Computes FC * S"""
        freqdiff = self._calcfreqdiff()
        freqdiff2 = freqdiff*freqdiff
        return - self.A * self.C * (   self.calcgaussian() * freqdiff2 )

    # Public methods and properties
    def _getforce( self ):
        """Force at the current state"""
        denominator = self._calcforcedenominator()
        da = self._calcforce_a()
        db = self._calcforce_b()
        dc = self._calcforce_c()
        dd = self._calcforce_d()
        dA = self._calcforce_A()
        dB = self._calcforce_B()
        dC = self._calcforce_C()

        derivArray = array( ( da, db, dc, dd, dA, dB, dC ) ).transpose()
        Beta  = array(self.signals - self.calcf()).transpose()
        numerator = dot(derivArray.transpose(),Beta)
        denominator = inv(dot(derivArray.transpose(),derivArray))

        
        return dot(denominator, numerator)

    force = property( _getforce )

    def _getenergy( self ):
        """Energy at the current state in units of N"""
        datadiff = self._calcdatadiff()
        cond = self.A > 0 and 0 < self.B < 1 and self.C > 0 
        if cond :
            energy = 0.5 * self.ldata * log( ( datadiff * datadiff ).sum() )
        else :
            energy = inf
        return energy
    energy = property( _getenergy )
  
    def showstate( self ) :
        """Print out information about the current state of this object."""
        print "Fa =", self._calcforce_a()
        print "Fb =", self._calcforce_b()
        print "Fc =", self._calcforce_c()
        print "Fd =", self._calcforce_d()
        print "FA =", self._calcforce_A()
        print "FB =", self._calcforce_B() 
        print "FC =", self._calcforce_C()
        print "parameters (a, b, c, d, A, B, C) = (", self.parameters, ")"
        print "force magnitude =", norm(self.force)

    def _getwidth( self ):
        """The width parameter of the model"""
        return self.invC
    width = property( _getwidth )

    def _getpos( self ):
        """The position parameter of the model"""
        return self.B
    position = property( _getpos )

class DataMismatchError(Exception):
    def __init__( self, ldata1, ldata2 ):      
        self.ldata1= ldata1
        self.ldata2=ldata2
        self.msg = "Got %d points in first data set and %d points in the second" % ( ldata1, ldata2 )
    def __str__( self ): return self.msg 
