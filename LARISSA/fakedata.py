#!/usr/bin/python
#from random import random
from numpy.random import random, normal
from numpy import array, exp, savetxt
from pylab import plot, show

def polyPlusGauss( point ):
    a = 1.0
    b = -0.5
    c = -0.3
    d = -0.2
    A = 1.0
    B = 0.5
    C = 10.0
    
    point2=point*point
    point3=point2*point
    poly = 1- 0.5 * point- 0.3*point2 - 0.2*point3

    diff = point - B
    diff2 = diff * diff
    C2 = C * C
    gauss = exp( -0.5 * C2 * diff2 ) 

    return poly + gauss

function = polyPlusGauss
xdata=[]
ydata=[]
for i in range(0,100) :
    point = i/100.
    fvalue = function(point)
    datavalue = fvalue + 0.3*random()
    xdata.append(point)
    ydata.append(datavalue)

plot( xdata, ydata )
show()

savetxt( "simulated.txt", array(( xdata, ydata )).transpose() ) 
